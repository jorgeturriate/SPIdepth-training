import os
import cv2
import argparse
import numpy as np

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = np.sqrt(np.mean((gt - pred) ** 2))
    rmse_log = np.sqrt(np.mean((np.log(gt) - np.log(pred)) ** 2))
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def main(pred_path, gt_path, image_format):
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    gt_files = sorted([f for f in os.listdir(gt_path) if f.endswith(('.npy', '.png'))])
    pred_files = sorted([f for f in os.listdir(pred_path) if f.endswith(('.npy', '.png'))])

    assert len(gt_files) == len(pred_files), "Mismatch between GT and predicted images"

    errors = []

    for gt_file, pred_file in zip(gt_files, pred_files):
        gt_f = os.path.join(gt_path, gt_file)
        pred_f = os.path.join(pred_path, pred_file)

        # Load images
        gt = np.load(gt_f) if gt_file.endswith(".npy") else cv2.imread(gt_f, cv2.IMREAD_UNCHANGED).astype(np.float32)
        pred = np.load(pred_f) if pred_file.endswith(".npy") else cv2.imread(pred_f, cv2.IMREAD_UNCHANGED).astype(np.float32)

        # Handle uint16 scaling if needed
        if image_format == "uint16":
            gt /= 256.0
            pred /= 256.0

        # Resize prediction if shape mismatch
        if gt.shape != pred.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Mask and scale
        mask = gt > MIN_DEPTH
        scale = np.median(gt[mask]) / np.median(pred[mask])
        pred *= scale

        # Clip
        pred = np.clip(pred, MIN_DEPTH, MAX_DEPTH)
        gt = np.clip(gt, MIN_DEPTH, MAX_DEPTH)

        errors.append(compute_errors(gt[mask], pred[mask]))

    # Print results
    mean_errors = np.mean(errors, axis=0)
    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate predicted depth against ground truth")
    parser.add_argument("--pred_path", type=str, required=True, help="Path to predicted depth maps")
    parser.add_argument("--gt_path", type=str, required=True, help="Path to ground truth depth maps")
    parser.add_argument("--format", type=str, default="uint16", choices=["uint16", "colormap"],
                        help="Format of input images if PNG: 'uint16' for raw depth, 'colormap' for visualization only")

    args = parser.parse_args()
    main(args.pred_path, args.gt_path, args.format)
