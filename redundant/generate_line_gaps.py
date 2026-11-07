import random
from pathlib import Path
import cv2
import numpy as np

def generate_clean_cuts(input_dir, output_dir, gap_probability=0.002, cut_length_range=(3, 8)):
    """
    Applies clean transverse line cuts (eraser strokes) to line art images
    so that lines are fully severed rather than just getting internal holes.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Generating clean-cut synthetic gaps from: {input_path.resolve()} ...\n")

    processed_count = 0
    error_count = 0

    for file_path in input_path.glob("*.png"):
        try:
            img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
            y_indices, x_indices = np.where(binary < 50)
            if len(x_indices) == 0:
                continue

            num_gaps = max(1, int(len(x_indices) * gap_probability))
            broken_img = binary.copy()

            for _ in range(num_gaps):
                idx = random.randint(0, len(x_indices) - 1)
                cx, cy = x_indices[idx], y_indices[idx]

                # Create a small line eraser segment to cleanly slice across the stroke
                length = random.randint(cut_length_range[0], cut_length_range[1])
                angle = random.uniform(0, np.pi)
                
                x2 = int(cx + length * np.cos(angle))
                y2 = int(cy + length * np.sin(angle))

                # Draw a white line (255) to cleanly sever the stroke
                cv2.line(broken_img, (cx, cy), (x2, y2), color=255, thickness=3)

            out_file_path = output_path / file_path.name
            cv2.imwrite(str(out_file_path), broken_img)
            processed_count += 1

        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            error_count += 1

    print("-" * 40)
    print(f"Clean-Cut Gap Generation Complete! Saved to: {output_path.resolve()}")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate clean-cut synthetic gaps")
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--gap-probability", type=float, default=0.002)
    args = parser.parse_args()

    generate_clean_cuts(args.input_dir, args.output_dir, args.gap_probability)

# Usage:
# python script.py ./Outputs_skeletonized ./Inputs_skeletonized
# python script.py ./Outputs_skeletonized ./Inputs_skeletonized --gap-probability 0.002
# python script.py ./Outputs_skeletonized ./Inputs_skeletonized --gap-probability 0.002 --eraser-radius-range 3 7