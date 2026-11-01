from pathlib import Path
import cv2
import numpy as np
from scipy.signal import convolve2d
from skimage.morphology import skeletonize


def baseline_line_completion(input_dir, output_dir, max_bridge_dist=25):
    """
    Classical DIP Line Completion using:
    1. Binary thresholding
    2. Skeletonization
    3. Endpoint detection
    4. Automatic stroke thickness estimation
    5. Heuristic gap bridging

    Parameters:
        input_dir: Directory containing gapped line art PNGs.
        output_dir: Directory to save completed images.
        max_bridge_dist: Maximum Euclidean distance for bridging endpoints.
    """

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(
        f"Running Baseline DIP Line Completion on: "
        f"{input_path.resolve()} ...\n"
    )

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
            img = cv2.imread(
                str(file_path),
                cv2.IMREAD_GRAYSCALE
            )

            if img is None:
                continue

            # 2. Binary thresholding
            # Lines = black (0), background = white (255)
            _, binary = cv2.threshold(
                img, 127, 255, cv2.THRESH_BINARY
            )

            # Prepare image for skeletonization
            # Lines = 1, background = 0
            lines = (binary < 127).astype(np.uint8)

            # 3. Skeletonization
            skeleton = skeletonize(
                lines > 0
            ).astype(np.uint8)

            # 4. Automatic stroke thickness estimation
            distance_map = cv2.distanceTransform(
                lines,
                cv2.DIST_L2,
                5
            )

            # Sample distance values along the skeleton
            skeleton_distances = distance_map[
                skeleton > 0
            ]

            if len(skeleton_distances) > 0:

                median_radius = np.median(
                    skeleton_distances
                )

                estimated_thickness = max(
                    1,
                    int(round(2 * median_radius - 1))
                )

            else:
                estimated_thickness = 1

            print(
                f"{file_path.name}: "
                f"Estimated thickness = "
                f"{estimated_thickness} pixels"
            )

            # 5. Endpoint Detection using 8-connected neighbors
            neighbor_count = convolve2d(
                skeleton,
                kernel,
                mode="same",
                boundary="fill",
                fillvalue=0
            )

            # Endpoints have exactly one neighbor
            endpoints_mask = (
                (skeleton == 1) &
                (neighbor_count == 1)
            )

            y_endpoints, x_endpoints = np.where(
                endpoints_mask
            )

            endpoints = list(
                zip(x_endpoints, y_endpoints)
            )

            # 6. Gap Bridging
            completed_img = binary.copy()

            connected_pairs = set()

            for i, pt1 in enumerate(endpoints):

                for j, pt2 in enumerate(endpoints):

                    if i >= j:
                        continue

                    if (
                        (i, j) in connected_pairs or
                        (j, i) in connected_pairs
                    ):
                        continue

                    # Calculate Euclidean distance
                    dist = np.linalg.norm(
                        np.array(pt1) - np.array(pt2)
                    )

                    # Check whether endpoints are close enough
                    if 3 <= dist <= max_bridge_dist:

                        # Draw connecting line using estimated thickness
                        cv2.line(
                            completed_img,
                            pt1,
                            pt2,
                            color=0,
                            thickness=estimated_thickness
                        )

                        connected_pairs.add((i, j))

                        # Move to the next endpoint
                        break

            # 7. Save completed image
            out_file_path = (
                output_path / file_path.name
            )

            cv2.imwrite(
                str(out_file_path),
                completed_img
            )

            processed_count += 1

        except Exception as e:

            print(
                f"Error processing {file_path.name}: {e}"
            )

            error_count += 1

    # 8. Print processing summary
    print("-" * 40)
    print("Baseline Processing Complete!")
    print(f"Completed images saved: {processed_count}")
    print(f"Errors encountered: {error_count}")
    print(f"Output Directory: {output_path.resolve()}")


if __name__ == "__main__":

    # Configure directories
    GAPPED_INPUT_DIRECTORY = "./Inputs"
    BASELINE_OUTPUT_DIRECTORY = "./Results"

    baseline_line_completion(
        input_dir=GAPPED_INPUT_DIRECTORY,
        output_dir=BASELINE_OUTPUT_DIRECTORY,
        max_bridge_dist=11
    )