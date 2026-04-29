# ODELIA Project: Breast Lesion Classification via Deep Learning (DCE-MRI)

Welcome to the main repository for our mini-project on the ODELIA dataset. This project aims to develop a robust Deep Learning pipeline for classifying breast lesions (Normal, Benign, Malignant) using dynamic contrast-enhanced MRI (DCE-MRI) sequences.

The complexity of this challenge lies in the heterogeneity of the clinical data (variable number of contrast phases depending on the hospital) and the structural difficulty in differentiating certain benign lesions from malignant tumors.

To address this, our approach was built iteratively across three major phases, with each bringing new ideas comparing to the others to try to perform better.

---

## Architecture Evolution

### 🔹 Phase 1: The Strict Baseline (Static CNN)
*See the `3DResNet/` folder for technical details.*

The first step was to establish a solid baseline, guaranteed to be free of "Data Leakage".
* **Approach:** Use of a standard 3D ResNet.
* **Method:** Standardized input on 5 strict channels (T2, Pre, Post_1, Post_2, Sub_1) with strict 5-Fold Cross-Validation.
* **Outcome:** Stable architecture serving as a reliable benchmark. Detailed cross-validation and clinical ensembling metrics are available in: `ODELIA_Challenge_CV_DL/3DResNet/Results/ensemble_metrics_results.md`
* **Explainable AI (XAI):** Integration of the `Grad-CAM` algorithm generating 3D heatmaps (.nii.gz) that can be overlaid in 3D Slicer to validate the clinical relevance of the predictions.

### 🔹 Phase 2: Temporal Exploration (using ViT, ie Vision Transformer)
*See the `Swin-UNETR/` folder for technical details.*

To break the rigidity of the baseline and try to use ViT for this problem, we transitioned to a Global Attention-based architecture to capture the kinetics of the contrast agent.
* **Approach:** Multi-Stream model based on `Swin-UNETR` (3D ViT).
* **Method:** Processing contrast phases via a Siamese network. Managed the variable number of MRI sequences using *Repeat Padding* (clinical LOCF method) to force a temporal dimension of T=4.
* **Outcome:** Demonstrated strong detection of malignant patterns but showed limitations in delineating benign lesions. Full metrics available in: `ODELIA_Challenge_CV_DL/Swin-UNETR/Results/ensemble_metrics_results.md`
* **Explainable AI (XAI):** Integration of the `Grad-CAM` algorithm generating 3D heatmaps (.nii.gz) that can be overlaid in 3D Slicer to validate the clinical relevance of the predictions.

### 🔹 Phase 3: Hybrid and Explainable Architecture (CNN + Temporal Pooling)
*See the `MultiStream-3DResNet/` folder for technical details.*

The culmination of the project. This phase merges the local excellence of convolutions with the sequential power of recurrent networks, while making the decisions interpretable for medical professionals.
* **Approach:** Dynamic Spatio-Temporal Modeling (Siamese `ResNet18` + Dynamic Spatio-Temporal Modeling via `Temporal Pooling`).
* **Key Mechanisms:**
  1. **Automatic ROI:** Intelligent cropping (`CropForegroundd`) to focus computational power directly on the useful breast tissue.
  2. **Dynamic Time:** Complete abandonment of padding. The network reads the exact sequence of each patient, whether they have 2 or 6 post-contrast images.
  3. **Regularization:** Addition of `Modality Dropout` to force the model to generalize across all available modalities.
  4. **Explainable AI (XAI):** Integration of the `Grad-CAM` algorithm generating 3D heatmaps (.nii.gz) that can be overlaid in 3D Slicer to validate the clinical relevance of the predictions.
* **Outcome:** Our most advanced model, achieving the best balance between local precision and temporal flexibility. Comprehensive results available in: `ODELIA_Challenge_CV_DL/MultiStream-3DResNet/Results/ensemble_metrics_results.md`

---

## Repository Structure

* `3DResNet/`: Code and logs of the initial ResNet implementation.
* `Swin-UNETR/`: Implementation of the Swin-UNETR pipeline and LOCF management.
* `MultiStream-3DResNet/`: Final source code (ResNet+Temporal Pooling), training scripts, Data Augmentation utilities, and Grad-CAM heatmap generation.
* In each folder : 
    * `Job_script_sh`: sh scripts used to launch the jobs in one pipeline with slurm
    * `Jupyter` : just to store the only notebook used at the beginning of the project to gather predictions
    * `Python` : all the source code used
    * `Training_results`: used for reporting all the results. Sometimes one folder with a date is used for multiple attemps, especially when a lot of them went wrong. Via the git, there is only the READM files, but you can ask for us to add you the permission to see the whole results directly from idun.

*(Note: The `.pth` model weights are not versioned due to their size), but everything is stored on idun and on [huggingFace](https://huggingface.co/collections/simontho/odelia-deep-learning-breast-cancer-classification)"*
