# Phase 2: Temporal Exploration (Swin-UNETR) for ODELIA Breast Lesion Classification

This document details the second phase of the ODELIA project. The primary goal of this phase was to move beyond the rigid 5-channel baseline and actively exploit the full temporal dynamics of the dynamic contrast-enhanced MRI (DCE-MRI) sequences. To achieve this, we transitioned to a Global Attention-based architecture utilizing a Vision Transformer (ViT).

## 1. Approach and Model Architecture

The core of Phase 2 replaces the CNN with a Multi-Stream Siamese network based on the `Swin-UNETR` (3D ViT) architecture.

*   **Model Type:** `MultiStreamSwin` (Custom implementation combining two `Swin-UNETR` backbones).
*   **Input:** The model processes the static T2 sequence and a dynamic sequence of subtracted post-contrast images independently.
*   **Input Size:** Spatial dimensions standardized to (128, 128, 64). The temporal dimension is fixed to T=4.
*   **Architecture Details:**
    *   **Structural Branch:** A Swin-UNETR encoder dedicated to extracting anatomical features from the static `T2` sequence.
    *   **Kinetic Branch (Siamese Network):** A second Swin-UNETR encoder with shared weights that independently processes each temporal phase of the contrast diffusion (`Post_i - Pre`).
    *   **Fusion & Header:** Features from both branches are extracted via Global Average Pooling, temporally flattened, concatenated into a single large vector (size 3840), and passed through a Classification MLP.
    *   **Transfer Learning:** To mitigate the data-hungry nature of ViTs, the backbones are initialized with pre-trained weights from MONAI [`swin_unetr_btcv.pt`](https://project-monai.github.io/model-zoo.html#/model/swin_unetr_btcv_segmentation).

## 2. Training Procedure

The model was trained using the same strict 5-Fold Cross-Validation strategy established in Phase 1 to guarantee the absence of data leakage.

### 2.1 Dataset and Temporal Processing (LOCF)

A major advancement in this phase is the dynamic handling of variable MRI sequences (e.g., handling centers with 2 vs. 4 post-contrast phases).
*   **Dynamic Subtraction:** The custom `PrepareTemporalKineticsd` transform calculates the subtractions (`Post_i - Pre`) on the fly, forcing the network to focus strictly on contrast absorption.
*   **Repeat Padding / LOCF:** To accommodate PyTorch batching constraints without destroying the temporal signal with zero-padding, we use the clinical method of *Last Observation Carried Forward (LOCF)*. If a patient has fewer than 4 post-contrast phases, the last available phase is repeated to simulate a contrast plateau up to T=4.
*   **Class Imbalance Handling:** To prevent the model from collapsing onto the majority class ("Normal"), dynamic class weights are calculated and injected directly into a `FocalLoss` function.

### 2.2 Preprocessing and Augmentation (MONAI Transforms)

The preprocessing pipeline was updated to handle temporal tensors:
*   **Loading:** `LoadImaged` for T2, Pre, and all available Post sequences.
*   **Resizing:** `Resized` to a uniform spatial size of (128, 128, 64).
*   **Temporal Formatting:** `PrepareTemporalKineticsd` to create the standardized (T=4, C=1, H, W, D) kinetic tensor.
*   **Type Conversion:** `EnsureTyped` for the specific outputs required by the two-stream model.

### 2.3 Hyperparameters

*   **Optimizer:** Adam
*   **Learning Rate:** 1e-4
*   **Loss Function:** Weighted FocalLoss (gamma=2.0)
*   **Batch Size:** Simulated Batch Size of 4 (True Batch Size of 1 + 4 Gradient Accumulation steps to fit VRAM).
*   **Mixed Precision:** Automatic Mixed Precision (AMP) using `autocast` and `GradScaler`.
*   **Epochs:** 50

## 3. Key Learnings and Future Directions from Phase 2

This temporal ViT exploration provided vital insights for the final architecture:
*   **Data Hunger and Stability:** Observing the learning curves confirmed that Vision Transformers require massive amounts of data. Despite Transfer Learning, stabilizing the Swin-UNETR across the smaller 5-Fold data partitions proved challenging and computationally expensive.
*   **Lack of Local Inductive Bias:** While excellent at detecting the aggressive global temporal patterns of malignant tumors, the model struggled significantly to geometrically delineate benign lesions from normal tissue. The patching mechanism of ViTs lacks the strong local spatial bias inherent to CNNs.
*   **Evolution towards Phase 3:** The inability to accurately identify benign lesions natively justified pivoting toward a hybrid architecture for Phase 3. The final pipeline will leverage CNNs (ResNet) for local precision and RNNs (LSTM) for native, padding-free temporal flexibility.

## 4. Repository main Contents (Phase 2 Specific)

*   `Python/train_odelia.py`: Main training script updated for dynamic data loading, Multi-Stream ViT processing, AMP, and gradient accumulation.
*   `Python/generate_predictions.py`: Script for generating probabilistic predictions and extracting Grad-CAM heatmaps from targeted transformer layers using custom forward wrappers.
*   `Python/utils.py`: Centralized module containing the `MultiStreamSwin` network class and the `PrepareTemporalKineticsd` transformation.
*   `Python/plot_final_performance.py`: Utility for visualizing training curves and evaluating cross-validation stability.
*   `Log_and_helper_files/`: Has to contains metadata, the pre-trained `.pt` weights, and split information.
*   `Training_results/`: Stores logs, saved model weights (`.pth`), and exported NIfTI heatmaps.

## 5. Weights and Prediction Access

The trained weights for the 5-fold cross-validation and the inference JSON files are available on [Hugging Face](https://huggingface.co/simontho/Swin-UNETR-Odelia)
