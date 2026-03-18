# Deepfake_detection_Using_CNN

A deepfake detection model built with EfficientNet-B4 trained on the FaceForensics++ (C23) dataset. The pipeline covers face extraction, augmentation, fine-tuning, and evaluation.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Roadmap](#roadmap)

---

## Overview

This project detects AI-generated (deepfake) faces in videos using a CNN-based binary classifier. It fine-tunes a pretrained **EfficientNet-B4** backbone on face crops extracted from the FaceForensics++ dataset, achieving strong generalization across multiple manipulation methods including Deepfakes, Face2Face, FaceSwap, NeuralTextures, FaceShifter, and DeepFakeDetection.

---

## Dataset

**FaceForensics++ C23** — 1000 real videos + 6 × 1000 manipulated videos compressed at quality level C23.

| Split | Source | Videos |
|-------|--------|--------|
| Real | `original/` | 1,000 |
| Fake | `Deepfakes/` | 1,000 |
| Fake | `Face2Face/` | 1,000 |
| Fake | `FaceSwap/` | 1,000 |
| Fake | `NeuralTextures/` | 1,000 |
| Fake | `FaceShifter/` | 1,000 |
| Fake | `DeepFakeDetection/` | 1,000 |

Download via Kaggle:

```python
import kagglehub
path = kagglehub.dataset_download("xdxd003/ff-c23")
```

---

## Pipeline

```
Raw Videos
    │
    ▼
Step 1 │ Data Collection       FaceForensics++ C23 via kagglehub
    │
    ▼
Step 2 │ Face Detection        MTCNN → align on landmarks → crop 224×224
    │
    ▼
Step 3 │ Augmentation          Flip, JPEG compression, brightness, noise
    │
    ▼
Step 4 │ Model                 EfficientNet-B4 + GAP + Dropout + Sigmoid
    │
    ▼
Step 5 │ Training              Freeze backbone → fine-tune top layers
    │
    ▼
Step 6 │ Evaluation            AUC-ROC, F1, EER, cross-dataset test
    │
    ▼
Step 7 │ Deploy                GradCAM explainability + ONNX export
```

---

## Project Structure

```
deepfake-detection/
├── data/
│   └── face_crops/
│       ├── train/
│       │   ├── real/
│       │   └── fake/
│       └── val/
│           ├── real/
│           └── fake/
├── notebooks/
│   └── deepfake_detection.ipynb   ← main Colab notebook
├── src/
│   ├── extract_faces.py           ← Step 2: face detection pipeline
│   ├── dataset.py                 ← PyTorch Dataset + augmentation
│   ├── model.py                   ← EfficientNet-B4 + SE block + head
│   ├── train.py                   ← training loop
│   └── evaluate.py                ← metrics + GradCAM
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/your-username/deepfake-detection.git
cd deepfake-detection
pip install -r requirements.txt
```

**requirements.txt**

```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
facenet-pytorch>=2.5.3
opencv-python-headless>=4.8.0
albumentations>=1.3.0
pytorch-grad-cam>=1.4.8
kagglehub
matplotlib
scikit-learn
```

---

## Usage

### Step 1 — Download dataset

```python
import kagglehub
path = kagglehub.dataset_download("xdxd003/ff-c23")
```

### Step 2 — Extract face crops

```python
from src.extract_faces import process_videos

BASE = "/kaggle/input/ff-c23/FaceForensics++_C23"

process_videos(f"{BASE}/original",   output_dir="data/face_crops/train/real", label="real")
process_videos(f"{BASE}/Deepfakes",  output_dir="data/face_crops/train/fake", label="fake")
```

### Step 3 — Train

```python
from src.train import train

train(
    data_dir   = "data/face_crops/",
    epochs     = 30,
    batch_size = 32,
    lr         = 1e-4
)
```

### Step 4 — Evaluate

```python
from src.evaluate import evaluate

evaluate(
    model_path = "checkpoints/best_model.pth",
    data_dir   = "data/face_crops/val/"
)
```

---

## Model Architecture

```
Input (224×224×3)
    │
    ▼
EfficientNet-B4 backbone    (pretrained on ImageNet, top layers unfrozen)
    │
    ▼
SE Block                    (channel recalibration — boosts relevant features)
    │
    ▼
Global Average Pooling      (1792-dim feature vector)
    │
    ▼
Dropout (p=0.5)
    │
    ▼
Linear → Sigmoid            (output: 0 = real, 1 = fake)
```

**Training strategy:**
- Phase 1 — freeze backbone, train head only (5 epochs, LR = 1e-3)
- Phase 2 — unfreeze top 3 blocks, fine-tune full network (25 epochs, LR = 1e-4)
- Loss: `BCEWithLogitsLoss`
- Optimizer: `AdamW`
- Scheduler: Cosine annealing

---

## Results

> Results will be updated as training completes.

| Metric | In-distribution | Cross-dataset |
|--------|----------------|---------------|
| AUC-ROC | — | — |
| F1 Score | — | — |
| EER | — | — |

---

## Roadmap

- [x] Dataset download and exploration
- [x] Face detection pipeline (MTCNN + alignment)
- [ ] Augmentation pipeline
- [ ] Model training
- [ ] Evaluation + GradCAM heatmaps
- [ ] ONNX export for deployment
- [ ] Temporal model (CNN + LSTM) for video-level prediction

---

## References

- [FaceForensics++](https://github.com/ondyari/FaceForensics) — Rössler et al., ICCV 2019
- [EfficientNet](https://arxiv.org/abs/1905.11946) — Tan & Le, ICML 2019
- [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507) — Hu et al., CVPR 2018
- [facenet-pytorch](https://github.com/timesler/facenet-pytorch) — MTCNN implementation
