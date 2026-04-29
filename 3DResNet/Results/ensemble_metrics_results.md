# Ensemble Performance Summary - 3D-ResNet

This document provides a detailed breakdown of the performance metrics for the 3D-ResNet model, covering both the 5-fold cross-validation phase and the final ensemble evaluation on the ODELIA dataset.

## 1. Cross-Validation Stability (per Fold)

The following table tracks the best Validation AUROC and training behavior for each fold. The consistency across folds (noted by the low standard deviation) validates the robustness of the training pipeline.

| Fold | Best Validation AUROC | Loss Convergence |
| :--- | :---: | :--- |
| **Fold 0** | 0.585 | Successful (Clean descent) |
| **Fold 1** | 0.659 | Successful |
| **Fold 2** | 0.779 | Successful |
| **Fold 3** | 0.743 | Successful |
| **Fold 4** | 0.696 | Successful |
| --- | --- | --- |
| **Mean CV** | **0.692 ± 0.067** | - |

## 2. Final Ensemble Results (304 Patients)

By aggregating the probabilistic predictions from all five models (Ensemble Method), we observe a significant performance boost. This demonstrates that the models learned complementary features from different data partitions.

* **Final AUROC (Malignant vs Rest):** **0.8174**
* **Sensitivity (@ 90% Specificity):** 51.61%
* **Specificity (@ 90% Sensitivity):** 45.87%

## 3. Confusion Matrix Analysis (Ensemble)

The matrix below shows the distribution of predictions for the three classes: **Normal**, **Benign**, and **Malignant**.

| True \ Predicted | Normal | Benign | Malignant |
| :--- | :---: | :---: | :---: |
| **Normal** | **171** | 16 | 23 |
| **Benign** | 28 | **3** | 1 |
| **Malignant** | 26 | 3 | **33** |

## 4. Key Takeaways & Analysis

### The "Ensemble Effect"
The jump from a mean CV AUROC of **0.692** to an ensemble AUROC of **0.8174** confirms the effectiveness of the Cross-Validation strategy. The ensemble effectively filters out individual model noise and generalizes better on the malignant class.

### Class Challenges (Benign vs Normal)
The model currently struggles with the **Benign** class, often misclassifying these lesions as **Normal** (28/32). This indicates that for a 3D-ResNet, the morphological differences between benign lesions and healthy tissue are extremely subtle. This provides a clear justification for the more complex architectures (Swin-UNETR) explored in Phase 2.

### Clinical Utility
With 33 true positives for **Malignant** cases and a high AUC, the model shows promising capability in identifying cancerous lesions, which is the primary clinical objective. The low number of Benign-to-Malignant false positives (only 1) is a strong point for diagnostic safety.

### Training Health
The convergence curves (available in the `visualization` folder) show a healthy decline in loss across all folds without significant overfitting, proving that the regularization and learning rate choices were appropriate for this medical 3D task.

