from pathlib import Path
import cv2
import numpy as np
from skimage.morphology import skeletonize


def create_skeletonized_dataset(input_dir, output_dir):
  """Reads clean line art, binarizes, skeletonizes to 1-pixel thickness,

  and saves to a new directory for baseline testing.
  """
  input_path = Path(input_dir)
  output_path = Path(output_dir)
  output_path.mkdir(parents=True, exist_ok=True)

  processed_count = 0
  error_count = 0

  print(f"Generating 1-pixel skeletonized dataset from: {input_path.resolve()} ...\n")

  for file_path in input_path.glob("*.png"):
    try:
      # 1. Read image as grayscale
      img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
      if img is None:
        continue

      # 2. Threshold to strict binary (lines = dark, background = light)
      _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

      # 3. Prepare for scikit-image skeletonize (lines = 1, background = 0)
      lines = (binary < 127).astype(np.uint8)

      # 4. Perform skeletonization to reduce everything to 1-pixel width
      skeleton = skeletonize(lines > 0)

      # 5. Convert back to standard image format (Black lines on White background)
      # skeleton is True/False; convert to 0 (black) and 255 (white)
      skel_img = np.ones_like(binary) * 255
      skel_img[skeleton] = 0

      # 6. Save result
      out_file_path = output_path / file_path.name
      cv2.imwrite(str(out_file_path), skel_img)
      processed_count += 1

    except Exception as e:
      print(f"Error processing {file_path.name}: {e}")
      error_count += 1

  print("-" * 40)
  print(f"✨ Skeletonization Complete!")
  print(f"📁 Skeletonized images saved: {processed_count}")
  print(f"⚠️ Errors encountered: {error_count}")
  print(f"📂 Output Directory: {output_path.resolve()}")


if __name__ == "__main__":
  # Point to your flattened clean dataset
  CLEAN_INPUT_DIRECTORY = "./flattened_png_dataset"
  SKELETON_OUTPUT_DIRECTORY = "./skeletonized_clean_dataset"

  create_skeletonized_dataset(
      input_dir=CLEAN_INPUT_DIRECTORY, output_dir=SKELETON_OUTPUT_DIRECTORY
  )