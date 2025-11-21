Here is the detailed Implementation Plan for **PhotoDate AI**. This plan focuses on the **Base Product (MVP)** using the architecture and data sources defined in your Product Design and Tech Stack documents.

---

# Implementation Plan: PhotoDate AI (MVP)

**Objective:** Build a functional local machine learning tool that accepts an image and outputs a predicted year using the Yearbook Dataset and EfficientNetV2.

## Phase 1: Environment & Hardware Setup
**Goal:** Establish a GPU-accelerated local development environment.

1.  **Verify Hardware Drivers**
    *   Update NVIDIA GPU drivers to the latest proprietary version to ensure compatibility with CUDA 11.8+.
    *   **Validation:** Run `nvidia-smi` in the terminal and confirm the driver version and GPU recognition.

2.  **Initialize Conda Environment**
    *   Create a new Miniconda environment named `photodate` running Python 3.10.
    *   Activate the environment.
    *   **Validation:** Run `python --version` and confirm it returns Python 3.10.x.

3.  **Install PyTorch & CUDA**
    *   Install PyTorch, Torchvision, and Torchaudio with the specific CUDA 11.8 (or 12.x) compute platform commands.
    *   **Validation:** Open a Python shell, import torch, and run `torch.cuda.is_available()`. It must return `True`.

4.  **Install Support Libraries**
    *   Install the secondary libraries defined in the Tech Stack: `timm`, `albumentations`, `gradio`, `pandas`, `scikit-learn`, and `matplotlib`.
    *   **Validation:** Run `pip list` and verify all packages are present.

---

## Phase 2: Data Acquisition & Preparation
**Goal:** Prepare the Yearbook Dataset for ingestion.

5.  **Acquire Data**
    *   Download the "Yearbook Dataset" (Ginosar et al.) from the official repository.
    *   Extract the dataset into a local directory structure (e.g., `./data/raw/`).
    *   **Validation:** Verify the total file count matches the expected dataset size (~37,921 images).

6.  **Create Metadata Manifest**
    *   Generate a master CSV file containing two columns: `filepath` (relative path to image) and `year` (integer).
    *   **Validation:** Open the CSV and verify that the first 5 rows correctly map a specific image file to its corresponding year.

7.  **Split Dataset**
    *   Using `scikit-learn`, split the master CSV into `train.csv` (80%) and `val.csv` (20%).
    *   Ensure the split is stratified by year to ensure all decades are represented in both sets.
    *   **Validation:** Calculate the distribution of years in both CSVs; the histograms should look statistically similar.

---

## Phase 3: The Data Pipeline
**Goal:** Create a robust loading mechanism that transforms images into tensors.

8.  **Define Image Transforms**
    *   Configure `Albumentations` to create a composition pipeline.
    *   **Step A:** Resize all images to 256x256 pixels (as per PDD).
    *   **Step B:** Normalize pixel values using ImageNet mean and standard deviation standards.
    *   **Step C:** Convert to PyTorch Tensor.
    *   **Validation:** Apply the transform to a dummy image; confirm output shape is `(3, 256, 256)`.

9.  **Implement Custom Dataset Class**
    *   Create a Python class inheriting from `torch.utils.data.Dataset`.
    *   Logic: Read image from path in CSV, apply Transforms, return a tuple of `(image_tensor, year_label)`.
    *   **Important:** Ensure `year_label` is converted to a Float type (required for Regression).
    *   **Validation:** Instantiate the class and pull index `0`. Confirm it returns a Tensor and a Float.

10. **Configure DataLoaders**
    *   Create PyTorch DataLoaders for both Train and Val sets.
    *   Set `batch_size` (start with 32) and `num_workers`.
    *   **Validation:** Iterate through one loop of the DataLoader. Confirm the batch shape is `[32, 3, 256, 256]` and label shape is `[32]`.

---

## Phase 4: Model Architecture
**Goal:** Adapt EfficientNetV2 for regression.

11. **Load Backbone**
    *   Use `timm` to load the `efficientnetv2_rw_m` (or `s`) architecture.
    *   Initialize with `pretrained=True` to utilize transfer learning.
    *   **Validation:** Print the model summary/architecture and locate the final classifier layer (often named `.classifier` or `.head`).

12. **Modify Output Head**
    *   Replace the final fully connected layer.
    *   Change the output features from 1000 (ImageNet classes) to **1** (The predicted year).
    *   **Validation:** Pass a dummy tensor of shape `[1, 3, 256, 256]` through the model. Confirm the output is a single scalar value.

---

## Phase 5: Training Logic
**Goal:** Teach the model to minimize the error between predicted year and actual year.

13. **Define Optimization components**
    *   Set Loss Function to `MSELoss` (Mean Squared Error) or `L1Loss` (Mean Absolute Error).
    *   Initialize Optimizer: `AdamW` with a learning rate of `1e-4`.
    *   **Validation:** Perform a manual calculation check: pass a prediction tensor and a target tensor to the loss function and ensure it returns a gradient-capable float.

14. **Implement Training Loop**
    *   Create a function that iterates over the Train DataLoader.
    *   Steps: Move data to GPU -> Forward Pass -> Calculate Loss -> Backward Pass -> Optimizer Step -> Zero Gradients.
    *   **Validation:** Run the loop for 10 batches. Confirm the Loss value is decreasing.

15. **Implement Validation Loop**
    *   Create a function that iterates over the Val DataLoader.
    *   Set model to evaluation mode (`model.eval()`) and disable gradient calculation.
    *   Metric Calculation: Compute the Mean Absolute Error (MAE) in years (e.g., Average difference of 4.5 years).
    *   **Validation:** Run the loop. Confirm it outputs a specific MAE number and memory usage remains stable.

---

## Phase 6: Execution
**Goal:** Train the final model artifact.

16. **Execute Full Training**
    *   Run the training loop for approx 50 epochs.
    *   Implement logic to save the model state dictionary (`state_dict`) whenever the Validation MAE improves (Check-pointing).
    *   **Validation:** Monitor console logs. Ensure training does not crash and MAE trends downward.

17. **Save Artifacts**
    *   Save the final best model as `photodate_v1.pth`.
    *   **Validation:** Start a fresh Python script, load the model architecture, load the `.pth` file, and ensure no "missing keys" errors occur.

---

## Phase 7: Application Wrapper (Gradio)
**Goal:** A user-friendly interface for inference.

18. **Build Inference Function**
    *   Write a Python function that:
        1.  Accepts a raw image path/object.
        2.  Applies the *exact same* validation transforms (Resize/Normalize) used in training.
        3.  Runs the image through the loaded model.
        4.  Returns the predicted year as a clean integer/string.
    *   **Validation:** Pass a random internet image into the function and print the result.

19. **Create Web Interface**
    *   Use `gradio` to create a `Interface` block.
    *   Input: Image (Drag and drop).
    *   Output: Text (The predicted year).
    *   Launch the server locally (`share=False`).
    *   **Validation:** Open `localhost:7860` in a browser. Upload a test photo and verify the model returns a date.