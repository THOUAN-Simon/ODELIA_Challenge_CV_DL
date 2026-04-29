# Phase 1: The Strict Baseline (Static CNN) for ODELIA Breast Lesion Classification

This document details the first phase of the ODELIA project, focusing on establishing a robust and reliable baseline for breast lesion classification using a 3D ResNet model. The primary goal of this phase was to develop a foundational pipeline that is rigorously free from data leakage, ensuring that any observed performance is a true reflection of the model's generalization capabilities.

## 1. Approach and Model Architecture

The core of Phase 1 utilizes a standard 3D Convolutional Neural Network (CNN) based on the ResNet architecture.

*   **Model Type:** 3D ResNet-18 (adapted from MONAI library).
*   **Input:** The model processes 5 specific MRI sequences concatenated as channels: T2, Pre-contrast, Post-contrast 1, Post-contrast 2, and Subtraction 1 (Sub_1).
*   **Input Size:** All input volumes are resized to a standardized dimension of (128, 128, 64).
*   **Architecture Details:**
    *   **Block Type:** Basic block
    *   **Layers:** [2, 2, 2, 2] (equivalent to ResNet-18)
    *   **In-planes:** [64, 128, 256, 512]
    *   **Input Channels:** 5
    *   **Output Classes:** 3 (Normal, Benign, Malignant)
    *   **Spatial Dimensions:** 3D

## 2. Training Procedure

The model was trained using a 5-Fold Cross-Validation strategy to ensure robustness and assess generalization across different data partitions.

### 2.1 Dataset and Data Leakage Prevention

A critical aspect of this phase was the meticulous handling of the dataset to prevent data leakage.
*   **Dataset Size:** The training utilized a clean dataset of 478 unique patients, derived from 4 institutions (CAM, MHA, RUMC, UKA).
*   **Split Logic:** The official ODELIA challenge split map (`split_unilateral.csv`) was rigorously applied. The `train_odelia.py` script was specifically designed to delegate fold partitioning, ensuring that patients appearing in a validation set for a given fold were strictly excluded from the training set for that same fold. This addressed initial challenges with subtle data leakage identified during early experiments.

### 2.2 Preprocessing and Augmentation (MONAI Transforms)

The following MONAI transforms were applied to the data:
*   **Loading:** `LoadImaged` for the essential MRI sequences.
*   **Channel Management:** `EnsureChannelFirstd` to correctly format image data.
*   **Resizing:** `Resized` to a uniform spatial size of (128, 128, 64).
*   **Normalization:** `ScaleIntensityd` for intensity scaling across sequences.
*   **Concatenation:** `ConcatItemsd` to combine the 5 sequences into a single multi-channel input tensor.
*   **Type Conversion:** `EnsureTyped` for consistent data types.

### 2.3 Hyperparameters

*   **Optimizer:** Adam
*   **Learning Rate:** 1e-4
*   **Loss Function:** CrossEntropyLoss
*   **Batch Size:** 4 (Training) / 2 (Validation)
*   **Epochs:** 50
*   **Validation Frequency:** Every 2 epochs

## 3. Key Learnings and Future Directions from Phase 1

This initial phase provided crucial insights that guided subsequent development:
*   **Data Leakage:** Early experiments highlighted the critical importance of a robust data splitting and loading mechanism to avoid subtle data leakage, which was successfully implemented and verified.
*   **Fixed Input Limitation:** The standardized 5-channel input, while simplifying the architecture, meant that additional available sequences (e.g., Post-3, Post-4 from certain centers) were not utilized. This suggested a need for more flexible architectures in future phases to leverage all available temporal information.
*   **Need for Explainability:** Recognizing the clinical context, the pipeline integrates the generation of Grad-CAM heatmaps during inference. This allows for visual interpretation of the model's decisions, a crucial step for clinical adoption.

## 4. Repository main Contents (Phase 1 Specific)

*   `Python/train_odelia.py`: The main script for training the 3D ResNet model with 5-Fold Cross-Validation.
*   `Python/generate_predictions.py`: Script for generating predictions and Grad-CAM heatmaps on the validation set.
*   `Python/plot_final_performance.py`: Utility for visualizing training curves and overall performance metrics (excluding specific results in this README).
*   `Log_and_helper_files/`: Has to contains metadata and split information used for data preparation.
*   `Training_results/`: Stores logs, saved model weights (`.pth`), and generated heatmaps.

## 5. Weights and rediction acess

The trained weights for the 5-fold cross-validation and the inference JSON files are available on [Hugging Face](https://huggingface.co/simontho/3DResNet-Odelia/)
