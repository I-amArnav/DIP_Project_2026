# Line Art Completion Project - Execution Guide

## Prerequisites
* Python 3.8+
* Required libraries: `opencv-python`, `numpy`, `scipy`, `scikit-image`, `Pillow`, `keras`, `tensorflow`

---

## Step-by-Step Execution Instructions

### 1. Dataset Extraction & Restructuring
```bash
unzip CBO_Dataset_20181113.zip
python3 restructure.py ./CBO_Dataset_20181113 ./Outputs
```

### 2. Generate Skeletonized Dataset
```bash
python3 skeletonize.py ./Outputs ./Outputs_sk
```

### 3. Generate Synthetic Gaps
For full-thickness images:
```bash
python3 generate_inputs.py ./Outputs ./Inputs
```

For skeletonized images:
```bash
python3 generate_inputs.py ./Outputs_sk ./Inputs_sk
```

### 4. Run Classical Completion Algorithms

**Algorithm 1 (Enhanced Geometric Baseline):**
```bash
python3 algo1.py ./Inputs ./Results1
```

**Algorithm 2 (Direction-Aware Bezier Heuristic):**
```bash
python3 algo2.py ./Inputs ./Results2
```

### 5. Evaluate Results

**Evaluate Algorithm 1:**
```bash
python3 evaluate.py ./Outputs ./Results ./Inputs
```

### 6. Generate Visual Overlays
```bash
python3 make_overlay.py ./Outputs ./Results ./Inputs ./Overlays
```