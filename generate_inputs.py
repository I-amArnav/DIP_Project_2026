import random
from pathlib import Path
import cv2
import numpy as np


def generate_synthetic_gaps(
    input_dir, output_dir, gap_probability=0.002, eraser_radius_range=(3, 7)
):
  """Applies synthetic contiguous gaps (eraser strokes) to line art images.

  :param input_dir: Directory containing clean PNG line art.
  :param output_dir: Directory to save broken/gapped line art.
  :param gap_probability: Density of gaps (number of eraser strokes relative
  to image size).
  :param eraser_radius_range: Min and max radius of the eraser circle in pixels.
  """
  input_path = Path(input_dir)
  output_path = Path(output_dir)
  output_path.mkdir(parents=True, exist_ok=True)

  processed_count = 0
  error_count = 0

  print(f"Generating synthetic gaps from: {input_path.resolve()} ...\n")

  for file_path in input_path.glob("*.png"):
    try:
      # Read image as grayscale
      img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
      if img is None:
        continue

      # Ensure binary line art: lines are black (low values), background is white (255)
      # Assuming standard line art (dark lines on white background)
      # If your lines are white on black, invert this logic.
      _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

      # Find coordinates of all black pixels (stroke pixels)
      y_indices, x_indices = np.where(binary < 50)
      if len(x_indices) == 0:
        continue

      # Determine number of gaps based on stroke pixel count
      # (e.g., 1 gap per ~500 stroke pixels, adjustable via gap_probability)
      num_gaps = int(len(x_indices) * gap_probability)
      num_gaps = max(
          1, num_gaps
      )  # Ensure at least 1 gap if there are strokes

      # Copy image to apply modifications
      broken_img = binary.copy()

      # Randomly select anchor points along the strokes and "erase" them
      for _ in range(num_gaps):
        # Pick a random index from the black pixels
        idx = random.randint(0, len(x_indices) - 1)
        cx, cy = x_indices[idx], y_indices[idx]

        # Choose a random eraser radius
        radius = random.randint(
            eraser_radius_range[0], eraser_radius_range[1]
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
  # Update paths to match your workflow
  CLEAN_INPUT_DIRECTORY = "./Outputs"
  GAPPED_OUTPUT_DIRECTORY = "./Inputs"

  # Tune gap_probability and eraser_radius_range to make gaps harder or easier
  generate_synthetic_gaps(
      input_dir=CLEAN_INPUT_DIRECTORY,
      output_dir=GAPPED_OUTPUT_DIRECTORY,
      gap_probability=0.0015,  # Controls how many gaps per image
      eraser_radius_range=(2, 5),  # Controls gap width/size in pixels
  )