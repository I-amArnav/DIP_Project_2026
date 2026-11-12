import argparse
import random
from pathlib import Path
import cv2
import numpy as np


def generate_synthetic_gaps(
    input_dir,
    output_dir,
    gap_probability=0.002,
    thickness_multiplier=1.3,
    min_radius=7,
):
  """Applies synthetic contiguous gaps (eraser strokes) to line art images,

  using an eraser radius dynamically scaled to the local stroke thickness via
  distance transform.
  """
  input_path = Path(input_dir)
  output_path = Path(output_dir)
  output_path.mkdir(parents=True, exist_ok=True)

  print(f"Generating synthetic gaps from: {input_path.resolve()} ...\n")

  processed_count = 0
  error_count = 0

  for file_path in input_path.glob("*.png"):
    try:
      # Read image as grayscale
      img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
      if img is None:
        continue

      # Ensure binary line art: lines are black (low values), background is white (255)
      _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
      lines = (binary < 50).astype(np.uint8)

      # Compute distance transform to find local stroke thickness at every pixel
      distance_map = cv2.distanceTransform(lines, cv2.DIST_L2, 5)

      # Find coordinates of all black pixels (stroke pixels)
      y_indices, x_indices = np.where(binary < 50)
      if len(x_indices) == 0:
        continue

      # Determine number of gaps based on stroke pixel count
      num_gaps = int(len(x_indices) * gap_probability)
      num_gaps = max(1, num_gaps)

      # Copy image to apply modifications
      broken_img = binary.copy()

      # Randomly select anchor points along the strokes and "erase" them
      for _ in range(num_gaps):
        idx = random.randint(0, len(x_indices) - 1)
        cx, cy = x_indices[idx], y_indices[idx]

        # Dynamically scale eraser radius based on local distance transform thickness
        local_thickness = distance_map[cy, cx]
        radius = int(
            max(min_radius, local_thickness * thickness_multiplier + random.uniform(0, 2))
        )

        # Draw a white circle (255) to wipe out the line segment
        cv2.circle(broken_img, (cx, cy), radius, (255), thickness=-1)

      # Save the corrupted image
      out_file_path = output_path / file_path.name
      cv2.imwrite(str(out_file_path), broken_img)
      processed_count += 1

    except Exception as e:
      print(f"Error processing {file_path.name}: {e}")
      error_count += 1

  print("-" * 40)
  print(f"Gap Generation Complete!")
  print(f"Broken images saved: {processed_count}")
  print(f"Errors encountered: {error_count}")
  print(f"Output Directory: {output_path.resolve()}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Generate synthetic gaps with thickness-scaled radius"
  )

  parser.add_argument("input_dir")
  parser.add_argument("output_dir")
  parser.add_argument("--gap-probability", type=float, default=0.0035)
  parser.add_argument(
      "--thickness-multiplier",
      type=float,
      default=1.5,
      help="Multiplier for local thickness to ensure clean cuts",
  )
  parser.add_argument(
      "--min-radius",
      type=int,
      default=9,
      help="Minimum eraser radius in pixels",
  )

  args = parser.parse_args()

  generate_synthetic_gaps(
      input_dir=args.input_dir,
      output_dir=args.output_dir,
      gap_probability=args.gap_probability,
      thickness_multiplier=args.thickness_multiplier,
      min_radius=args.min_radius,
  )

# Usage:
# python script.py ./Outputs_skeletonized ./Inputs_skeletonized
# python script.py ./Outputs_skeletonized ./Inputs_skeletonized --gap-probability 0.002
# python script.py ./Outputs_skeletonized ./Inputs_skeletonized --gap-probability 0.002 --eraser-radius-range 3 7