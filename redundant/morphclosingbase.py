from pathlib import Path
import cv2
import numpy as np
from scipy.signal import convolve2d
from skimage.morphology import closing, disk, skeletonize

def localized_morphological_completion(input_dir, output_dir, footprint_radius=2, max_bridge_dist=15):
    """
    Localized Morphological Closing: Only applies closing around detected endpoints
    to bridge gaps without thickening the rest of the clean line art.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Running Localized Morphological Closing on: {input_path.resolve()} ...\n")

    processed_count = 0
    error_count = 0

    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    footprint = disk(footprint_radius)

    for file_path in input_path.glob("*.png"):
        try:
            img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            lines = (binary < 127).astype(np.uint8)

            # Skeletonize to find endpoints
            skeleton = skeletonize(lines > 0).astype(np.uint8)
            neighbor_count = convolve2d(skeleton, kernel, mode="same", boundary="fill", fillvalue=0)
            endpoints_mask = (skeleton == 1) & (neighbor_count == 1)
            y_endpoints, x_endpoints = np.where(endpoints_mask)
            endpoints = list(zip(x_endpoints, y_endpoints))

            # Create a localized mask for where gaps are likely located
            gap_mask = np.zeros_like(binary, dtype=np.uint8)

            # Pair up close endpoints and draw local regions of interest (ROIs)
            for i, pt1 in enumerate(endpoints):
                for j, pt2 in enumerate(endpoints):
                    if i >= j:
                        continue
                    dist = np.linalg.norm(np.array(pt1) - np.array(pt2))
                    if 2 <= dist <= max_bridge_dist:
                        # Draw a small connecting line mask between candidate endpoints
                        cv2.line(gap_mask, pt1, pt2, color=255, thickness=footprint_radius * 2)

            # Apply morphological closing ONLY inside the candidate gap regions
            lines_bool = (binary < 127)
            closed_full = closing(lines_bool, footprint)
            
            # Blend: Keep original binary everywhere except where the localized gap mask triggers closing
            completed_img = binary.copy()
            local_patch = (closed_full & (gap_mask > 0))
            completed_img[local_patch] = 0  # Fill in as black line pixels

            out_file_path = output_path / file_path.name
            cv2.imwrite(str(out_file_path), completed_img)
            processed_count += 1

        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            error_count += 1

    print("-" * 40)
    print("Localized Morphological Processing Complete!")
    print(f"Output Directory: {output_path.resolve()}")

if __name__ == "__main__":
    GAPPED_INPUT_DIRECTORY = "./Inputs"
    CLOSING_OUTPUT_DIRECTORY = "./Results"

    localized_morphological_completion(
        input_dir=GAPPED_INPUT_DIRECTORY,
        output_dir=CLOSING_OUTPUT_DIRECTORY,
        footprint_radius=2,
        max_bridge_dist=12
    )