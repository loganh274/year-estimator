# Tech Stack: AI Photo Year Estimator

This document outlines a completely free, open-source, and industry-standard tech stack for developing a deep learning model capable of estimating the year a photograph was taken. The stack is optimized for local development on a machine equipped with a dedicated NVIDIA GPU (CUDA).

## 1. Core Development Environment

*   **Language:** **Python 3.10+**
    *   *Why:* The absolute standard for AI/ML development with the largest ecosystem of libraries.
*   **Environment Manager:** **Miniconda (Conda)**
    *   *Why:* Essential for managing CUDA toolkit versions and Python dependencies without breaking your system configuration.

## 2. Deep Learning Framework

*   **Framework:** **PyTorch**
    *   *Why:* Currently dominates the research and industry landscape. It offers dynamic computation graphs (easier debugging) and excellent support for local CUDA acceleration.
*   **Model Library:** **timm (PyTorch Image Models)**
    *   *Why:* A massive library of state-of-the-art pre-trained models (ResNet, EfficientNet, Vision Transformers). Instead of writing a neural network from scratch, you can load a pre-trained architecture and "fine-tune" it for year estimation.

## 3. Model Architecture Strategy

Since "year estimation" relies on detecting film grain, color degradation, clothing styles, and background technology, you need a model that excels at texture and pattern recognition.

*   **Recommended Model:** **EfficientNetV2 (Small or Medium)**
    *   *Why:* It offers the best balance between accuracy and training speed. It is light enough to train on a single consumer GPU (like an RTX 3060 or 4070) but powerful enough to capture subtle features.
*   **Task Type:** **Regression**
    *   *Approach:* You will modify the final layer of the network to output a single float value (the year) and use **MSELoss** (Mean Squared Error) or **L1Loss** during training.

## 4. Data Processing & Augmentation

*   **Library:** **Albumentations**
    *   *Why:* The fastest image augmentation library.
    *   *Strategy:* To make your model robust, you must artificially degrade modern photos to look old (add noise, blur, desaturate) and clean up old photos during training. Albumentations handles this on the fly on the GPU/CPU.

## 5. The User Interface (The "Tool")

*   **Library:** **Gradio**
    *   *Why:* The fastest way to build a web interface for a model. With 5 lines of code, you get a drag-and-drop image uploader that runs in your browser and communicates with your local Python backend.
    *   *Cost:* Free and open-source.

## 6. The Dataset (Crucial Component)

Finding a dataset with verified "Year Taken" metadata is the hardest part of this project. Most internet images have EXIF data stripped or represent the year the image was *scanned*, not taken.

### Primary Recommendation: The "Yearbook" Dataset
*   **Description:** A collection of 37,921 frontal-facing American high school yearbook portraits from 1905 to 2013.
*   **Why it works:** The dates are verified. It is excellent for teaching the model to recognize hair, makeup, and clothing styles across decades.
*   **Source:** [The Yearbook Dataset (Ginosar et al.)](https://github.com/NEU-Gou/yearbook-dataset) (Check usage rights for academic/personal use).

### Secondary Recommendation: Flickr (Scraped)
For non-portrait photos (cars, streets, buildings), you need a custom dataset.
*   **Strategy:** Use the **Flickr API** (Free tier).
*   **Method:** Write a script to search for groups like "Vintage Photography" or tags like `taken:year=1985`.
*   **Filter:** Ensure you filter by "Date Taken," not "Date Uploaded."

## 7. Hardware Setup (Local Training)

To run this efficiently, you will need the following installed:

1.  **NVIDIA Drivers:** Latest proprietary drivers for your GPU.
2.  **CUDA Toolkit 11.8 or 12.x:** The computing platform.
3.  **CuDNN:** The deep neural network library for CUDA.

*Command to install the stack (once Conda is installed):*
```bash
conda create -n photodate python=3.10
conda activate photodate
# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# Install utilities
pip install timm albumentations gradio scikit-learn pandas matplotlib