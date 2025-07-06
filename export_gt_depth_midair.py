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
import skimage.transform

from utils import readlines



def export_gt_depths_kitti():

    parser = argparse.ArgumentParser(description='export_gt_depth')

    parser.add_argument('--data_path',
                        type=str,
                        help='path to the root of the KITTI data',
                        required=True)
    parser.add_argument('--split',
                        type=str,
                        help='which split to export gt from',
                        required=True,
                        choices=["eigen", "eigen_benchmark", 'midair'])
    opt = parser.parse_args()

    split_folder = os.path.join(os.path.dirname(__file__), "splits", opt.split)
    lines = readlines(os.path.join(split_folder, "val_files.txt"))

    print("Exporting ground truth depths for {}".format(opt.split))

    gt_depths = []
    for line in lines:

        #folder, frame_id, _ = line.split()
        folder, depth_folder, frame_id= line.split()
        frame_id = int(frame_id)

        
        gt_depth_path = os.path.join(opt.data_path, depth_folder, f"{frame_id:06d}.PNG")
        gt_depth= np.array(pil.open(gt_depth_path), dtype=np.uint16)
        disparity= gt_depth.view(dtype=np.float16).astype(np.float32)
        disparity[disparity == 0] = 0.01  # avoid division by 0
        gt_depth = 512. / disparity

        # Resize to 384x384 BEFORE appending
        gt_depth = skimage.transform.resize(gt_depth, (384, 384), order=0, preserve_range=True, mode='constant')


        gt_depths.append(gt_depth.astype(np.float32))

    output_path = os.path.join(split_folder, "gt_depths.npz")

    print("Saving to {}".format(opt.split))

    np.savez_compressed(output_path, data=np.array(gt_depths, dtype="object"))


if __name__ == "__main__":
    export_gt_depths_kitti()

