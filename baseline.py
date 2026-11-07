from pathlib import Path
import cv2
import numpy as np
from scipy.signal import convolve2d
from skimage.morphology import skeletonize


def baseline_normal_line_completion(input_dir, output_dir, max_bridge_dist=15):
    """
    Classical DIP Line Completion for Normal (Full-Thickness) Line Art:
    1. Binarization
    2. Internal skeletonization only for reliable endpoint detection
    3. Distance transform on normal lines for accurate stroke thickness estimation
    4. Gap bridging with thickness-matched lines and smooth caps on normal image
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Running Normal-Image Baseline Line Completion on: {input_path.resolve()} ...\n")

    processed_count = 0
    error_count = 0

    # 3x3 kernel for counting 8-connected neighbors
    kernel = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ], dtype=np.uint8)

    for file_path in input_path.glob("*.png"):
        try:
            # 1. Read input image as grayscale
            img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # 2. Binary thresholding (Lines = black [0], background = white [255])
            _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            lines = (binary < 127).astype(np.uint8)

            # 3. Skeletonize *internally* just to find clean endpoint coordinates
            skeleton = skeletonize(lines > 0).astype(np.uint8)

            # 4. Automatic stroke thickness estimation using distance transform on normal lines
            distance_map = cv2.distanceTransform(lines, cv2.DIST_L2, 5)
            skeleton_distances = distance_map[skeleton > 0]
            
            if len(skeleton_distances) > 0:
                median_radius = np.median(skeleton_distances)
                estimated_thickness = max(1, int(round(2 * median_radius)))
            else:
                estimated_thickness = 1  # Default thickness for normal lines

            # 5. Endpoint Detection using 8-connected neighbors on skeleton
            neighbor_count = convolve2d(skeleton, kernel, mode="same", boundary="fill", fillvalue=0)
            endpoints_mask = (skeleton == 1) & (neighbor_count == 1)
            y_endpoints, x_endpoints = np.where(endpoints_mask)
            endpoints = list(zip(x_endpoints, y_endpoints))

            # 6. Gap Bridging on the NORMAL full-thickness image
            completed_img = binary.copy()
            connected_pairs = set()
            cap_radius = max(1, estimated_thickness // 2)

            for i, pt1 in enumerate(endpoints):
                for j, pt2 in enumerate(endpoints):
                    if i >= j:
                        continue
                    if (i, j) in connected_pairs or (j, i) in connected_pairs:
                        continue

                    # Calculate Euclidean distance
                    dist = np.linalg.norm(np.array(pt1) - np.array(pt2))

                    if 3 <= dist <= max_bridge_dist:
                        # Draw matching thick line on normal image
                        cv2.line(completed_img, pt1, pt2, color=0, thickness=estimated_thickness)
                        
                        # Add smooth round caps so the joint blends naturally
                        cv2.circle(completed_img, pt1, cap_radius, color=0, thickness=-1)
                        cv2.circle(completed_img, pt2, cap_radius, color=0, thickness=-1)

                        connected_pairs.add((i, j))
                        break

            # 7. Save completed image
            out_file_path = output_path / file_path.name
            cv2.imwrite(str(out_file_path), completed_img)
            processed_count += 1

        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            error_count += 1

    print("-" * 40)
    print("Normal-Image Baseline Processing Complete!")
    print(f"Completed images saved: {processed_count}")
    print(f"Errors encountered: {error_count}")
    print(f"Output Directory: {output_path.resolve()}")


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normal Image Baseline line completion")
    parser.add_argument("input_dir", help="Input directory containing normal gapped images")
    parser.add_argument("output_dir", help="Output directory for completed images")
    parser.add_argument("--max-bridge-dist", type=int, default=17, help="Maximum bridge distance")

    args = parser.parse_args()

    baseline_normal_line_completion(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_bridge_dist=args.max_bridge_dist
    )