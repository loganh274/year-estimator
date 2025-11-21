# Architecture: PhotoDate AI

## High-Level Overview
PhotoDate AI is a local desktop application that uses deep learning to estimate the year a photograph was taken. It consists of a Python backend for model inference and a Gradio-based web interface for user interaction.

## System Components

### 1. Data Layer
*   **Input:** Digital images (JPG, PNG, WEBP).
*   **Dataset:** Yearbook Dataset (Ginosar et al.) for training.
*   **Preprocessing:** Albumentations pipeline (Resize to 256x256, Normalize).

### 2. Model Layer
*   **Architecture:** EfficientNetV2 (Small/Medium).
*   **Task:** Regression (Output: Year as Float).
*   **Framework:** PyTorch.
*   **Loss Function:** MSELoss or L1Loss.

### 3. Application Layer
*   **Interface:** Gradio Web UI.
*   **Inference Engine:** Python script loading the trained `.pth` model.
*   **Hardware:** Local NVIDIA GPU (CUDA) for acceleration.

## Directory Structure (Proposed)
```
year-estimator/
├── data/               # Raw and processed datasets
├── models/             # Saved model artifacts (.pth)
├── src/                # Source code
│   ├── data/           # Dataset classes and loaders
│   ├── model/          # Model architecture definitions
│   ├── train.py        # Training loop
│   ├── inference.py    # Inference logic
│   └── app.py          # Gradio entry point
├── notebooks/          # Exploratory notebooks
├── environment.yml     # Conda environment definition
└── README.md
```
