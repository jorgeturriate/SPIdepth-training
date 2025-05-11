import os
import torch
import json
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO
from datasets import KITTIRAWDataset
from torch.utils.data import DataLoader

def readlines(filename):
    with open(filename, 'r') as f:
        return f.read().splitlines()

def main(input_txt, output_txt, box_json, remaining_txt, data_path, num_images=1000, img_ext='.png'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load YOLOv5 (car = class 2 in COCO)
    model = YOLO("yolov5su.pt")  # Make sure this file is downloaded or pip install yolov5
    model.to(device)
    model.eval()

    # Load file paths
    all_filenames = readlines(input_txt)
    random.shuffle(all_filenames)  # for diversity

    dataset = KITTIRAWDataset(data_path, all_filenames, 192, 512, [0], 1, is_train=True, img_ext=img_ext)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    valid_paths = []
    car_boxes = {}
    used_indices = []

    for idx, inputs in tqdm(enumerate(dataloader), total=len(dataloader)):
        if len(valid_paths) >= num_images:
            break

        image = inputs[("color", 0, 0)][0].permute(1, 2, 0).numpy() * 255
        image = image.astype(np.uint8)

        results = model(image, verbose=False)[0]
        boxes = [b for b in results.boxes.data.cpu().numpy() if int(b[5]) == 2]  # class 2 = car

        if len(boxes) == 0 or len(boxes) > 4:
            used_indices.append(idx)
            continue

        boxes = sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
        main_box = boxes[0][:4].tolist()

        path_str = all_filenames[idx]
        valid_paths.append(path_str)
        car_boxes[path_str] = main_box
        used_indices.append(idx)

        # Print every 10 detections
        if len(valid_paths) % 5 == 0:
            print(f"[INFO] Found {len(valid_paths)} valid images with cars...")

    # Save selected paths
    with open(output_txt, 'w') as f:
        f.write('\n'.join(valid_paths))

    # Save bounding boxes
    with open(box_json, 'w') as f:
        json.dump(car_boxes, f, indent=2)

    # Save remaining paths (excluding selected and not selected ones)
    all_indices = set(range(len(all_filenames)))
    remaining_indices = sorted(all_indices - set(used_indices))
    remaining_paths = [all_filenames[i] for i in remaining_indices]
    with open(remaining_txt, 'w') as f:
        f.write('\n'.join(remaining_paths))

    print(f"\nSaved {len(valid_paths)} filtered image paths to {output_txt}")
    print(f"Saved bounding boxes to {box_json}")
    print(f"Saved remaining unprocessed paths to {remaining_txt}")

if __name__ == '__main__':
    # Example call in Colab:
    main(
        #input_txt="/content/SPIdepth-training/splits/eigen_zhou/train_files_original.txt",
        input_txt="/content/train_files_remaining2.txt",
        output_txt="/content/filtered_train_files3.txt",
        box_json="/content/car_boxes3.json",
        remaining_txt="/content/train_files_remaining.txt",
        data_path="gs://mde_data_bucket/kitti",
        num_images=500
    )