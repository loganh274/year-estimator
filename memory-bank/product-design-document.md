Here is a simple Product Design Document (PDD) for the project.

***

# Product Design Document: PhotoDate AI

| Document Details | |
| :--- | :--- |
| **Project Name** | PhotoDate AI |
| **Version** | 0.1 (Draft) |
| **Status** | In Development |
| **Core Tech** | Python, PyTorch, CUDA |

---

## 1. Executive Summary
**PhotoDate AI** is a locally hosted machine learning tool designed to analyze digital photographs and estimate the year they were taken. By examining visual cues—such as film grain, color degradation, clothing fashion, and background technology—the tool provides a predicted year (e.g., "1983") to help users organize digitized archives and family photo albums.

## 2. Problem Statement
Millions of analog photos are currently being digitized (scanned) by families and archivists. However, scanners usually set the "Date Created" metadata to the *moment of scanning*, not the *moment the photo was taken*. Manually identifying the year for thousands of images is time-consuming and requires historical expertise.

## 3. Target Audience
*   **Digital Archivists:** Professionals managing historical databases.
*   **Genealogists/Families:** Individuals organizing digitized family albums.
*   **Data Hoarders:** Users with massive local libraries of unsorted images.

## 4. Functional Requirements

### 4.1 Core Features
*   **Image Ingestion:** Support for JPG, PNG, and WEBP formats.
*   **Year Estimation:** Output a specific year (e.g., 1990) and a confidence window (e.g., ± 3 years).
*   **Privacy-First:** All processing must happen locally on the user's machine; no images are uploaded to the cloud.
*   **Batch Processing:** Ability to point the tool at a folder and generate a CSV of estimated dates.

### 4.2 User Interface (MVP)
*   Simple Web Interface (served locally via browser).
*   Drag-and-drop upload zone.
*   Display area for the image alongside the predicted timeline.

## 5. Technical Architecture

### 5.1 The Stack
*   **Language:** Python 3.10+
*   **ML Framework:** PyTorch (chosen for dynamic graphing and ease of debugging).
*   **Compute:** NVIDIA CUDA (requires local GPU for training and fast inference).
*   **GUI Wrapper:** Gradio or Streamlit (for rapid frontend deployment).

### 5.2 Model Design
*   **Architecture:** **EfficientNetV2** or **ResNet50**.
    *   *Reasoning:* Good balance of accuracy vs. training speed on consumer GPUs.
*   **Task Type:** Regression (predicting a continuous value: `Year`).
*   **Loss Function:** MSELoss (Mean Squared Error) or L1Loss to minimize the difference between the predicted year and the actual year.

## 6. Data Strategy

To train the model without cost, we will utilize public datasets where the timestamp is verified.

### 6.1 Primary Dataset Source
*   **The Yearbook Dataset (Ginosar et al.):**
    *   *Content:* 37,000+ frontal-facing portraits from American high schools (1905–2013).
    *   *Advantage:* Extremely consistent lighting and pose; the model learns to date images based on hairstyles, glasses, and clothing collars.
    *   *License:* Publicly available for research.

### 6.2 Secondary Data Augmentation
To make the model robust against non-portrait photos, we will perform **Synthetic Degradation** using the **Albumentations** library:
*   Apply random sepia tones.
*   Add Gaussian noise (to simulate film grain).
*   Reduce resolution and add JPEG artifacts.

## 7. Development Roadmap

### Phase 1: Setup & Data (Weeks 1-2)
*   Set up local Conda environment with CUDA 11.8/12.x.
*   Download the Yearbook Dataset.
*   Write a preprocessing script to resize all images to 256x256 and normalize pixel values.

### Phase 2: Training (Weeks 3-4)
*   Implement the EfficientNet architecture with a modified output head (1 output node).
*   Train on local GPU (approx. 50-100 epochs).
*   Evaluate using Mean Absolute Error (MAE). Target: MAE < 5 years.

### Phase 3: Application Wrapper (Week 5)
*   Build the Gradio interface.
*   Connect the trained model `.pth` file to the interface.
*   Test with user-provided "wild" photos to verify generalization.

## 8. Success Metrics
1.  **Accuracy:** The model predicts the correct decade 90% of the time.
2.  **Precision:** The Mean Absolute Error (MAE) is within 4 years.
3.  **Performance:** Inference time per image is under 200ms on an RTX 3060 or better.