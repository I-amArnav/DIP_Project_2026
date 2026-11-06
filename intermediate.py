from pathlib import Path
import cv2
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# 1. Custom Lightweight Residual Network (Built from scratch)
class CustomLineArtNet(nn.Module):

  def __init__(self):
    super(CustomLineArtNet, self).__init__()
    # Initial feature extraction layer
    self.input_layer = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
    )
    # Intermediate context layers for gap continuity learning
    self.hidden_layers = nn.Sequential(
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
    )
    # Output reconstruction layer with residual addition
    self.output_layer = nn.Sequential(
        nn.Conv2d(32, 1, kernel_size=3, padding=1), nn.Sigmoid()
    )

  def forward(self, x):
    identity = x  # Save input for residual skip connection
    out = self.input_layer(x)
    out = self.hidden_layers(out)
    # Add identity skip connection so the network focuses purely on bridging gaps
    out = self.output_layer(out + identity)
    return out


# 2. Complete Batch Processing & I/O Pipeline
def run_line_art_completion(input_dir, output_dir, weights_path=None):
  input_path = Path(input_dir)
  output_path = Path(output_dir)
  output_path.mkdir(parents=True, exist_ok=True)

  # Check for GPU availability
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using compute device: {device}")

  # Initialize model and set to evaluation mode
  model = CustomLineArtNet().to(device)
  model.eval()

  # Load pre-trained weights if provided
  if weights_path and Path(weights_path).exists():
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print(f"Successfully loaded weights from: {weights_path}")
  else:
    print(
        "⚠️ Warning: No weights file found. Running inference with uninitialized"
        " weights."
    )

  print(f"\nScanning gapped input directory: {input_path.resolve()} ...")
  processed_count = 0
  error_count = 0

  # Iterate through all PNG files in the input folder
  for file_path in input_path.glob("*.png"):
    try:
      # --- INPUT I/O ---
      img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
      if img is None:
        print(f"Skipping unreadable file: {file_path.name}")
        continue

      # --- PREPROCESSING ---
      # Normalize pixel values from [0, 255] to [0.0, 1.0]
      img_float = img.astype(np.float32) / 255.0

      # Convert numpy array to PyTorch tensor format: [Batch=1, Channel=1, Height, Width]
      tensor_img = torch.tensor(img_float).unsqueeze(0).unsqueeze(0).to(device)

      # --- INFERENCE ---
      with torch.no_grad():
        output_tensor = model(tensor_img)

      # --- POST-PROCESSING ---
      # Convert output tensor back to numpy image array [0, 255]
      output_np = (output_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
      _, binary_pred = cv2.threshold(output_np, 127, 255, cv2.THRESH_BINARY)

      # Bitwise intersection: Retain original structure while applying predicted gap closures
      final_result = cv2.bitwise_and(binary_pred, img)

      # --- OUTPUT I/O ---
      out_file_path = output_path / file_path.name
      cv2.imwrite(str(out_file_path), final_result)
      processed_count += 1

    except Exception as e:
      print(f"Error processing {file_path.name}: {e}")
      error_count += 1

  print("-" * 40)
  print(f"Line Art Completion Batch Complete!")
  print(f"Successfully processed & saved: {processed_count}")
  print(f"Errors encountered: {error_count}")
  print(f"Output Directory: {output_path.resolve()}")


# 1. Custom Dataset Loader
class LineArtDataset(Dataset):

  def __init__(self, gapped_dir, clean_dir):
    self.gapped_paths = sorted(list(Path(gapped_dir).glob("*.png")))
    self.clean_paths = sorted(list(Path(clean_dir).glob("*.png")))

  def __len__(self):
    return len(self.gapped_paths)

  def __getitem__(self, idx):
    # Read gapped input and clean target
    gapped_img = (
        cv2.imread(str(self.gapped_paths[idx]), cv2.IMREAD_GRAYSCALE).astype(
            np.float32
        )
        / 255.0
    )
    clean_img = (
        cv2.imread(str(self.clean_paths[idx]), cv2.IMREAD_GRAYSCALE).astype(
            np.float32
        )
        / 255.0
    )

    # Convert to PyTorch tensors [Channel=1, Height, Width]
    return (
        torch.tensor(gapped_img).unsqueeze(0),
        torch.tensor(clean_img).unsqueeze(0),
    )


# 2. Import your Custom Network architecture from earlier


def train_model():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # Setup dataset and dataloader
  dataset = LineArtDataset(
      "./gapped_input_dataset", "./flattened_png_dataset"
  )
  dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

  model = CustomLineArtNet().to(device)
  criterion = nn.MSELoss()  # Mean Squared Error loss for pixel comparison
  optimizer = optim.Adam(model.parameters(), lr=0.001)

  epochs = 10
  print(f"Starting training for {epochs} epochs on {device}...")

  for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for gapped_batch, clean_batch in dataloader:
      gapped_batch = gapped_batch.to(device)
      clean_batch = clean_batch.to(device)

      optimizer.zero_grad()
      outputs = model(gapped_batch)
      loss = criterion(outputs, clean_batch)
      loss.backward()
      optimizer.step()

      running_loss += loss.item()

    print(
        f"Epoch [{epoch+1}/{epochs}], Loss:"
        f" {running_loss/len(dataloader):.4f}"
    )


  os.makedirs("./weight", exist_ok=True)
  torch.save(model.state_dict(), "./weight/custom_line_art_model.pth")
  print("Training complete! Weights saved to ./weight/custom_line_art_model.pth")


if __name__ == "__main__":
  train_model()