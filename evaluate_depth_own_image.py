import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from IPython.display import display, clear_output

MIN_DEPTH = 1e-3
MAX_DEPTH = 80


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

def visualize(gt, pred, idx, save_dir=None):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].imshow(gt, cmap='plasma', vmin=MIN_DEPTH, vmax=MAX_DEPTH)
    axs[0].set_title("Ground Truth")
    axs[0].axis('off')

    axs[1].imshow(pred, cmap='plasma', vmin=MIN_DEPTH, vmax=MAX_DEPTH)
    axs[1].set_title("Prediction (scaled)")
    axs[1].axis('off')

    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"compare_{idx:03d}.png"))
        plt.close()
    else:
        display(fig)
        plt.close()


def main(pred_path, gt_path, image_format="uint16", vis_every=20, save_vis=False):

    gt_files = sorted(glob(os.path.join(gt_path, '*')))
    pred_files = sorted(glob(os.path.join(pred_path, '*')))

    assert len(gt_files) == len(pred_files), f"Mismatch: {len(gt_files)} GT vs {len(pred_files)} predictions"

    errors = []

    print(f"Found {len(gt_files)} samples to evaluate.")
    for idx, (gt_file, pred_file) in enumerate(zip(gt_files, pred_files)):
        # Load GT
        gt = np.load(gt_file) if gt_file.endswith(".npy") else cv2.imread(gt_file, cv2.IMREAD_UNCHANGED).astype(np.float32)
        pred = np.load(pred_file) if pred_file.endswith(".npy") else cv2.imread(pred_file, cv2.IMREAD_UNCHANGED).astype(np.float32)

        # Handle uint16 scaling if needed
        if image_format == "uint16":
            gt /= 256.0
            #if the predictions disparities were stored after multiplied by 256
            #pred /= 256.0 
            pred /= 1000.0  # if saved like SPIdepth predictions (disp * 1000)

        # Resize prediction if shape mismatch
        if gt.shape != pred.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Mask and scale
        mask = gt > MIN_DEPTH
        if mask.sum() == 0:
            continue

        scale = np.median(gt[mask]) / np.median(pred[mask])
        pred *= scale

        # Clip
        pred = np.clip(pred, MIN_DEPTH, MAX_DEPTH)
        gt = np.clip(gt, MIN_DEPTH, MAX_DEPTH)

        errors.append(compute_errors(gt[mask], pred[mask]))
        # Visualize every N samples
        if vis_every > 0 and idx % vis_every == 0:
            visualize(gt, pred, idx, "/content/vis" if save_vis else None)

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
    parser.add_argument("--vis_every", type=int, default=20, help="Visualize every N images")
    parser.add_argument("--save_vis", default=False , action="store_true", help="Save comparison plots")


    args = parser.parse_args()
    main(args.pred_path, args.gt_path, args.format, args.vis_every, args.save_vis)
