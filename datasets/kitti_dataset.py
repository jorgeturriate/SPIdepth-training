# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil
import gcsfs
import json

from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset

class GCSHandler:
    def __init__(self, data_path):
        self.is_gcs = data_path.startswith("gs://")
        if self.is_gcs:
            self.fs = gcsfs.GCSFileSystem()
            self.data_path = data_path.replace("gs://", "")
        else:
            self.fs = None
            self.data_path = data_path

    def open_image(self, path):
        if self.is_gcs:
            with self.fs.open(os.path.join(self.data_path, path), 'rb') as f:
                return pil.open(f).convert('RGB')
        else:
            return pil.open(os.path.join(self.data_path, path)).convert('RGB')

    def open_depth(self, path):
        if self.is_gcs:
            with self.fs.open(os.path.join(self.data_path, path), 'rb') as f:
                return pil.open(f)
        else:
            return pil.open(os.path.join(self.data_path, path))

    def get_file_path(self, *args):
        return os.path.join(*args)



class KITTIDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        self.data_path = args[0]  # asuming that data_path is in args[0]
        self.gcs = GCSHandler(self.data_path)
        super(KITTIDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
        # To normalize you need to scale the first row by 1 / image_width and the second row
        # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
        # If your principal point is far from the center you might need to disable the horizontal
        # flip augmentation.
        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
# K是相机内参 stereo_T是相机外参，也就是转换矩阵,4行4列，16维度
        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

        self.crop_box = kwargs.get('crop_box', False) # Use kwargs to get crop_box


        if self.crop_box:
            box_file = "train_car_boxes.json" if self.is_train else "val_car_boxes.json"
            root_dir= os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
            box_path = os.path.join(root_dir, "splits", "eigen_zhou", box_file)
            with open(box_path, 'r') as f:
                self.crop_boxes = json.load(f)
        else:
            self.crop_boxes = {}

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_rel_path = os.path.join(
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        if self.gcs.is_gcs:
            gcs_full_path = os.path.join(self.gcs.data_path, velo_rel_path)
            return self.gcs.fs.exists(gcs_full_path)
        else:
            local_path = os.path.join(self.data_path, velo_rel_path)
            return os.path.isfile(local_path)
        
    def apply_crop_and_resize(self, img, folder, frame_index, side):
        key = f"{folder} {frame_index} {side}"
        if key not in self.crop_boxes:
            return img.resize((self.width, self.height), pil.BILINEAR if img.mode == "RGB" else pil.NEAREST)

        x, y, w, h = self.crop_boxes[key]
        
        # Expand the box by 20% in all directions
        expansion_ratio = 0.2
        x_expand = int(w * expansion_ratio)
        y_expand = int(h * expansion_ratio)

        x0 = max(x - x_expand, 0)
        y0 = max(y - y_expand, 0)
        x1 = x + w + x_expand
        y1 = y + h + y_expand

        cropped = img.crop((x0, y0, x1, y1))
        resized = cropped.resize((self.width, self.height), pil.BILINEAR if img.mode == "RGB" else pil.NEAREST)
        return resized



    def get_color(self, folder, frame_index, side, do_flip):
        path = self.get_image_path(folder, frame_index, side)
        color = self.gcs.open_image(path)

        if self.crop_box:
            color = self.apply_crop_and_resize(color, folder, frame_index, side)

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color


class KITTIRAWDataset(KITTIDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(KITTIRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        return self.gcs.get_file_path(folder, "image_0{}/data".format(self.side_map[side]), f_str)

    def get_depth(self, folder, frame_index, side, do_flip):
        calib_path = self.gcs.get_file_path(folder.split("/")[0])

        velo_path = self.gcs.get_file_path(folder, "velodyne_points/data", "{:010d}.bin".format(frame_index))

        depth_gt = generate_depth_map(calib_path, velo_path, self.side_map[side], use_gcs=True, gcs=self.gcs.fs, gcs_root=self.gcs.data_path)
        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')
        
        if self.crop_box:
            # Convert to PIL Image for cropping
            depth_pil = pil.fromarray(depth_gt)
            depth_pil = self.apply_crop_and_resize(depth_pil, folder, frame_index, side)
            depth_gt = depth_pil.resize(self.full_res_shape, pil.NEAREST)
            depth_gt = np.array(depth_pil)


        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt


class KITTIOdomDataset(KITTIDataset):
    """KITTI dataset for odometry training and testing
    """
    def __init__(self, *args, **kwargs):
        super(KITTIOdomDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        return self.gcs.get_file_path("sequences/{:02d}".format(int(folder)), "image_{}".format(self.side_map[side]), f_str)


class KITTIDepthDataset(KITTIDataset):
    """KITTI dataset which uses the updated ground truth depth maps
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDepthDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        return self.gcs.get_file_path(folder, "image_0{}/data".format(self.side_map[side]), f_str)

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "{:010d}.png".format(frame_index)
        depth_path = self.gcs.get_file_path(folder, "proj_depth/groundtruth/image_0{}".format(self.side_map[side]), f_str)

        depth_gt = self.gcs.open_depth(depth_path)

        if self.crop_box:
            depth_gt = self.apply_crop_and_resize(depth_gt, folder, frame_index, side)

        depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
