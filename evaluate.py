from pathlib import Path
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def evaluate_line_completion(gt_dir, results_dir, input_dir="./Inputs", log_file="evaluation_comparison_log.txt"):
    """
    Evaluates both Corrupted Inputs and Algorithm Results against Ground Truth clean images.
    Computes PSNR, SSIM, MSE, Pixel Accuracy, Precision, Recall, and F1-Score.
    """
    gt_path = Path(gt_dir)
    res_path = Path(results_dir)
    in_path = Path(input_dir)
    log_path = Path(log_file)

    for path, name in [(gt_path, "Ground Truth"), (res_path, "Results"), (in_path, "Inputs")]:
        if not path.exists():
            print(f"Error: {name} directory '{path}' does not exist.")
            return

    gt_files = sorted(list(gt_path.glob("*.png")))
    if not gt_files:
        gt_files = sorted(list(gt_path.glob("*.jpg")))
        
    if not gt_files:
        print(f"No image files found in ground truth directory: {gt_path}")
        return

    print(f"Evaluating {len(gt_files)} images comparing Input vs Output against Ground Truth...")

    # Metrics storage for cumulative averages
    metrics = {
        "input": {"psnr": [], "ssim": [], "mse": [], "accuracy": [], "precision": [], "recall": [], "f1": []},
        "output": {"psnr": [], "ssim": [], "mse": [], "accuracy": [], "precision": [], "recall": [], "f1": []}
    }

    success_count = 0
    missing_count = 0

    with open(log_path, "w") as log:
        # Write CSV header
        log.write(
            "Filename,"
            "In_PSNR,In_SSIM,In_MSE,In_Accuracy,In_Precision,In_Recall,In_F1,"
            "Out_PSNR,Out_SSIM,Out_MSE,Out_Accuracy,Out_Precision,Out_Recall,Out_F1\n"
        )

        for gt_file in gt_files:
            res_file = res_path / gt_file.name
            in_file = in_path / gt_file.name

            # Fallback extension check if needed
            if not res_file.exists():
                res_file = res_path / gt_file.with_suffix(".jpg").name
            if not in_file.exists():
                in_file = in_path / gt_file.with_suffix(".jpg").name

            if not res_file.exists() or not in_file.exists():
                missing_count += 1
                continue

            # 1. Read images as grayscale
            gt_img = cv2.imread(str(gt_file), cv2.IMREAD_GRAYSCALE)
            res_img = cv2.imread(str(res_file), cv2.IMREAD_GRAYSCALE)
            in_img = cv2.imread(str(in_file), cv2.IMREAD_GRAYSCALE)

            if gt_img is None or res_img is None or in_img is None:
                continue

            # Ensure matching dimensions
            h, w = gt_img.shape
            if res_img.shape != (h, w):
                res_img = cv2.resize(res_img, (w, h))
            if in_img.shape != (h, w):
                in_img = cv2.resize(in_img, (w, h))

            total_pixels = h * w

            # Helper function to compute all metrics for a given image vs ground truth
            def compute_metrics(target_img):
                mse = np.mean((gt_img.astype(np.float64) - target_img.astype(np.float64)) ** 2)
                p = psnr(gt_img, target_img, data_range=255) if mse > 0 else 100.0
                s = ssim(gt_img, target_img, data_range=255)

                # Binary classification (Lines = foreground/black < 128, Background = white >= 128)
                gt_lines = (gt_img < 128)
                target_lines = (target_img < 128)

                tp = np.sum(gt_lines & target_lines)
                fp = np.sum(~gt_lines & target_lines)
                fn = np.sum(gt_lines & ~target_lines)
                tn = np.sum(~gt_lines & ~target_lines)

                accuracy = (tp + tn) / total_pixels
                precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

                return p, s, mse, accuracy, precision, recall, f1

            # Compute for Input vs GT and Output vs GT
            in_metrics = compute_metrics(in_img)
            res_metrics = compute_metrics(res_img)

            # Store in cumulative lists
            for key, val in zip(metrics["input"].keys(), in_metrics):
                metrics["input"][key].append(val)
            for key, val in zip(metrics["output"].keys(), res_metrics):
                metrics["output"][key].append(val)

            # Write row to log
            log.write(f"{gt_file.name},")
            log.write(",".join([f"{v:.4f}" for v in in_metrics]) + ",")
            log.write(",".join([f"{v:.4f}" for v in res_metrics]) + "\n")

            success_count += 1

    # 4. Compute and Print Cumulative Summary side-by-side
    print("=" * 70)
    print(" CUMULATIVE EVALUATION SUMMARY: INPUT vs. ALGORITHM OUTPUT")
    print("=" * 70)
    print(f"Successfully evaluated: {success_count} images")
    if missing_count > 0:
        print(f"Missing result/input files skipped: {missing_count}")
    print("-" * 70)
    print(f"{'Metric':<18} | {'Corrupted Input':<18} | {'Algorithm Output':<18} | {'Delta / Improvement':<18}")
    print("-" * 70)

    metric_names = [
        ("PSNR (dB)", "psnr", True),
        ("SSIM", "ssim", True),
        ("MSE", "mse", False),
        ("Accuracy", "accuracy", True),
        ("Precision", "precision", True),
        ("Recall", "recall", True),
        ("F1-Score", "f1", True)
    ]

    for label, key, higher_is_better in metric_names:
        in_val = np.mean(metrics["input"][key])
        out_val = np.mean(metrics["output"][key])
        
        if key == "mse":
            diff = in_val - out_val  # Lower MSE is better
            delta_str = f"-{diff:.4f}" if diff > 0 else f"+{-diff:.4f}"
        else:
            diff = out_val - in_val  # Higher is better
            delta_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"

        print(f"{label:<18} | {in_val:<18.4f} | {out_val:<18.4f} | {delta_str:<18}")

    print("=" * 70)

import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate line completion results")

    parser.add_argument("gt_dir", help="Path to ground truth directory")
    parser.add_argument("results_dir", help="Path to results directory")
    parser.add_argument("input_dir", help="Path to input directory")
    parser.add_argument("--log-file", default="evaluation_comparison_log.txt",
                        help="Path to output log file (default: evaluation_comparison_log.txt)")

    args = parser.parse_args()

    if os.path.exists(args.log_file):
        os.remove(args.log_file)

    evaluate_line_completion(
        gt_dir=args.gt_dir,
        results_dir=args.results_dir,
        input_dir=args.input_dir,
        log_file=args.log_file
    )

# Usage:
# python script.py ./Outputs_skeletonized ./Results_skeletonized ./Inputs_skeletonized
# python script.py ./Outputs_skeletonized ./Results_skeletonized ./Inputs_skeletonized --log-file evaluation_log.txt