# Phase 3: Hybrid and Explainable Architecture (MultiStream 3D-ResNet)

This document details the third and final phase of the ODELIA project. To address the limitations of the Vision Transformer from Phase 2 (data hunger and poor performance on benign lesions due to a lack of local inductive bias), this phase returns to a Convolutional Neural Network (CNN) backbone. However, it preserves the advanced temporal and multi-stream processing capabilities introduced in Phase 2, resulting in a highly flexible, dynamic, and explainable architecture.

## 1. Approach and Model Architecture

The core of Phase 3 utilizes a Multi-Stream Siamese network based on the 3D `ResNet` architecture, designed to handle variable-length temporal sequences without destructive padding.

*   **Model Type:** `MultiStreamResNet` (Custom implementation combining two ResNet backbones).
*   **Input:** The model processes the static T2 sequence and a dynamic sequence of subtracted post-contrast images independently.
*   **Input Size:** Spatial dimensions standardized to (128, 128, 64). The temporal dimension `T` is strictly dynamic (e.g., 2, 4, or 6 depending on the patient's actual scan).
*   **Architecture Details:**
    *   **Structural Branch:** A ResNet encoder dedicated to extracting anatomical features from the static `T2` sequence.
    *   **Kinetic Branch (Siamese Network):** A second ResNet encoder with shared weights that independently processes each available temporal phase of the contrast diffusion (`Post_i - Pre`).
    *   **Instance Normalization:** Because the dynamic temporal sequence length forces the `DataLoader` to use a `batch_size` of 1, standard Batch Normalization fails. The ResNet blocks were heavily modified to use `InstanceNorm` to maintain stability.
    *   **Temporal Mean Pooling:** Instead of a complex LSTM or the rigid flattening used in Phase 2, the variable sequence of spatial vectors extracted by the kinetic branch is fused using Mean Pooling across the temporal axis. This efficiently summarizes the global kinetics without adding learnable parameters.
    *   **Header:** Features from both branches are concatenated and passed through a Classification MLP with Dropout.

## 2. Training Procedure

The model was trained using the established 5-Fold Cross-Validation strategy to guarantee the absence of data leakage.

### 2.1 Dynamic Time and Regularization

This phase introduces major advancements in robustness and data handling:
*   **Complete Abandonment of Padding:** By processing each kinetic phase individually before temporal pooling, the network reads the exact sequence of each patient. No `Zero-Padding` or `Repeat-Padding` (LOCF) is required, eliminating mathematical artifacts.
*   **Modality Dropout:** To prevent co-adaptation (where the network relies solely on the easiest phase, like T2), a custom `ModalityDropoutd` transform was created. During training, there is a 20% probability that the T2 volume or a specific kinetic phase is completely zeroed out. This forces the model to generalize across all available modalities.
*   **Class Imbalance Handling:** Dynamic class weights are calculated and injected into a `FocalLoss` function (with `gamma=2.0`) to severely penalize the model for ignoring difficult minority classes (like Benign lesions).

### 2.2 Preprocessing and Spatial Augmentation (MONAI Transforms)

The preprocessing pipeline was heavily upgraded to combat overfitting:
*   **Loading & Subtraction:** `LoadImaged` and `PrepareTemporalKineticsd` compute the subtractions (`Post_i - Pre`) on the fly.
*   **Region of Interest (ROI):** `CropForegroundd` was initially explored to automatically crop the image around the breast tissue and ignore empty air/thorax, though it was conditionally disabled during XAI alignment testing.
*   **Spatial Augmentations:** `RandFlipd` and `RandAffined` (rotations and scaling) were introduced, which proved crucial for learning the geometrical nuances of Benign lesions.
*   **Noise Injection:** `RandGaussianNoised` was added to improve the overall robustness of the feature extractors.

### 2.3 Hyperparameters

*   **Optimizer:** AdamW (with Weight Decay for regularization)
*   **Learning Rate Scheduler:** `CosineAnnealingLR` (T_max=50, eta_min=1e-6)
*   **Loss Function:** Weighted FocalLoss
*   **Batch Size:** True Batch Size of 1 (required for dynamic temporal dimensions) + 4 Gradient Accumulation steps.
*   **Mixed Precision:** Automatic Mixed Precision (AMP) using PyTorch's native `torch.amp.autocast`.
*   **Epochs:** 50

## 3. Explainable AI (XAI) for Clinical Validation

A major focus of Phase 3 is making the Deep Learning decisions interpretable for medical professionals. 

*   **Grad-CAM Integration:** The `generate_predictions.py` script features an advanced implementation of MONAI's `Grad-CAM` algorithm, specifically optimized for multi-stream architectures.
*   **Custom Wrappers:** To extract gradients from a multi-stream architecture, a custom forward wrapper was implemented to trick the Grad-CAM module into targeting the final convolutional layer (`layer4`) of the T2 structural encoder.
*   **Clinical Output:** During inference, the script saves both the processed `T2` volume and the generated `Heatmap` as NIfTI files (`.nii.gz`) using a standardized affine matrix (`np.eye(4)`). This allows the exact region the neural network focused on to be perfectly overlaid onto the original MRI in 3D Slicer.

## 4. Repository Contents (Phase 3 Specific)

*   `Python/train_odelia.py`: Main training script featuring AMP, gradient accumulation, spatial augmentations, Cosine Annealing, and Focal Loss.
*   `Python/generate_predictions.py`: Inference script generating probabilistic predictions alongside 3D Grad-CAM heatmaps and aligned NIfTI volumes.
*   `Python/utils.py`: Centralized module containing the `MultiStreamResNet` architecture (with InstanceNorm and Temporal Pooling), `PrepareTemporalKineticsd`, and `ModalityDropoutd`.
*   `Python/ensemble_predictions.py`: Script to aggregate predictions across folds and compute clinical metrics (Sensitivity, Specificity).
*   `Python/plot_final_performance.py`: Utility for visualizing training curves and CV stability.
*   `Training_results/`: Stores logs, saved model weights (`.pth`), and exported XAI heatmaps.

## 5. Weights and Prediction Access

The trained weights for the 5-fold cross-validation and the inference JSON files are available on [Hugging Face](https://huggingface.co/simontho/MultiStream-3DResNet-Odelia).
