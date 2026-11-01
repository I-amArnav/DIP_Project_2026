import hashlib
from pathlib import Path
from PIL import Image


def process_dataset(input_dir, output_dir):
  input_path = Path(input_dir)
  output_path = Path(output_dir)
  output_path.mkdir(parents=True, exist_ok=True)

  # Supported image extensions (case-insensitive check)
  valid_extensions = {".jpg", ".jpeg", ".JPG", ".JPEG"}

  seen_hashes = set()
  converted_count = 0
  duplicate_count = 0
  error_count = 0

  print(f"Scanning directory: {input_path.resolve()} ...\n")

  # Recursively find all files
  for file_path in input_path.rglob("*"):
    if file_path.is_file():
      # Skip system files like Thumbs.db or hidden files
      if (
          file_path.name.startswith(".")
          or file_path.name.lower() == "thumbs.db"
      ):
        continue

      if file_path.suffix in valid_extensions:
        try:
          # 1. Compute MD5 hash of file content to detect true duplicates
          with open(file_path, "rb") as f:
            file_bytes = f.read()
            file_hash = hashlib.md5(file_bytes).hexdigest()

          if file_hash in seen_hashes:
            duplicate_count += 1
            continue
          seen_hashes.add(file_hash)

          # 2. Construct a collision-safe flat filename using relative path parts
          # e.g., animals/butterfly/butterfly1.JPG -> animals_butterfly_butterfly1.png
          rel_parts = file_path.relative_to(input_path).with_suffix(".png").parts
          flat_filename = "_".join(rel_parts)
          out_file_path = output_path / flat_filename

          # 3. Open, convert (handle CMYK/Palette modes), and save as PNG
          with Image.open(file_path) as img:
            if img.mode in ("CMYK", "P", "RGBA"):
              img = img.convert("RGB")
            img.save(out_file_path, "PNG")

          converted_count += 1

        except Exception as e:
          print(f"Error processing {file_path.name}: {e}")
          error_count += 1

  print("-" * 40)
  print(f"Processing Complete!")
  print(f"Total PNGs saved: {converted_count}")
  print(f"Duplicates removed: {duplicate_count}")
  print(f"Errors/Skipped: {error_count}")
  print(f"Output Directory: {output_path.resolve()}")


if __name__ == "__main__":
  # Update these paths to match your WSL/Linux environment paths
  INPUT_DIRECTORY = "./CBO_Dataset"
  OUTPUT_DIRECTORY = "./Outputs"

  process_dataset(INPUT_DIRECTORY, OUTPUT_DIRECTORY)