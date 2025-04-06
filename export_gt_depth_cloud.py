# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os

import argparse
import numpy as np
import PIL.Image as pil
import io
import gcsfs

from utils import readlines
from kitti_utils import generate_depth_map


def export_gt_depths_kitti():

    parser = argparse.ArgumentParser(description='export_gt_depth_cloud')

    parser.add_argument('--bucket_name',
                        type=str,
                        help='GCS bucket name where KITTI data is stored',
                        required=True)
    parser.add_argument('--split',
                        type=str,
                        help='which split to export gt from',
                        required=True,
                        choices=["eigen", "eigen_benchmark"])
    opt = parser.parse_args()

     # Init GCS filesystem
    fs = gcsfs.GCSFileSystem()

    # e.g., "splits/eigen/test_files.txt"
    split_folder = os.path.join(os.path.dirname(__file__), "splits", opt.split)
    lines = readlines(os.path.join(split_folder, "test_files.txt"))

    print("Exporting ground truth depths for {}".format(opt.split))
    gt_depths = []

    for line in lines:
        folder, frame_id, _ = line.split()
        frame_id = int(frame_id)

        if opt.split == "eigen":
            calib_dir = f"{opt.bucket_name}/kitti/{folder.split('/')[0]}"
            velo_filename = f"{opt.bucket_name}/kitti/{folder}/velodyne_points/data/{frame_id:010d}.bin"

            # Download both calibration and velodyne files locally
            local_calib_dir = "/tmp/kitti_calib"
            local_velo_file = "/tmp/temp_velo.bin"

            os.makedirs(local_calib_dir, exist_ok=True)

            # Copy all calibration files (assumes they're small)
            for calib_file in ["calib_cam_to_cam.txt", "calib_velo_to_cam.txt"]:
                remote_calib_file = f"{calib_dir}/{calib_file}"
                local_file_path = os.path.join(local_calib_dir, calib_file)
                with fs.open(remote_calib_file, 'rb') as rf, open(local_file_path, 'wb') as lf:
                    lf.write(rf.read())

            # Copy velodyne file
            with fs.open(velo_filename, 'rb') as rf, open(local_velo_file, 'wb') as lf:
                lf.write(rf.read())

            gt_depth = generate_depth_map(local_calib_dir, local_velo_file, 2, True)

        elif opt.split == "eigen_benchmark":
            image_path = f"{opt.bucket_name}/kitti/{folder}/proj_depth/groundtruth/image_02/{frame_id:010d}.png"
            with fs.open(image_path, 'rb') as f:
                img = pil.open(io.BytesIO(f.read()))
            gt_depth = np.array(img).astype(np.float32) / 256

        gt_depths.append(gt_depth.astype(np.float32))

    output_path = os.path.join(split_folder, "gt_depths.npz")
    print("Saving to {}".format(output_path))
    np.savez_compressed(output_path, data=np.array(gt_depths, dtype="object"))


if __name__ == "__main__":
    export_gt_depths_kitti()

