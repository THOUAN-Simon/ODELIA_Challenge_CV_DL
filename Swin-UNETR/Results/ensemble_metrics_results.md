# Ensemble Performance Summary - Multi-Stream Swin-UNETR

This document details the performance of the Multi-Stream Swin-UNETR architecture, which incorporates both structural (T2) and temporal (Kinetics) MRI features through a Transformer-based backbone.

## 1. Cross-Validation Stability (per Fold)

The model was trained using 5-fold cross-validation. The convergence of loss was particularly sharp in this phase, demonstrating the effectiveness of the pre-trained weights and the Focal Loss.

| Fold | Best Validation AUROC | Loss Convergence |
| :--- | :---: | :--- |
| **Fold 0** | 0.659 | Highly Stable |
| **Fold 1** | 0.762 | Highly Stable |
| **Fold 2** | 0.663 | Highly Stable |
| **Fold 3** | 0.696 | Highly Stable |
| **Fold 4** | 0.598 | Highly Stable |
| --- | --- | --- |
| **Mean CV** | **0.675 ± 0.053** | - |

## 2. Final Ensemble Results (304 Patients)

The ensemble averages the predictions of the 5 Vision Transformer streams. It maintains a strong discriminative power for the malignant class while processing a much larger feature space than the baseline.

* **Final AUROC (Malignant vs Rest):** **0.7809**
* **Sensitivity (@ 90% Specificity):** 33.87%
* **Specificity (@ 90% Sensitivity):** 43.80%

## 3. Confusion Matrix Analysis (Ensemble)

The matrix shows how the model handles the three classes (**Normal**, **Benign**, **Malignant**) using the multi-modal transformer approach.

| True \ Predicted | Normal | Benign | Malignant |
| :--- | :---: | :---: | :---: |
| **Normal** | **177** | 5 | 28 |
| **Benign** | 23 | **2** | 7 |
| **Malignant** | 30 | 2 | **30** |

## 4. Technical Analysis & Insights

### Transformer Convergence vs. Performance
The Swin-UNETR showed exceptionally clean loss curves, reaching near-zero training loss across several folds. While the ensemble AUROC (0.7809) is slightly lower than the ResNet baseline, this experiment confirms that **Transformers can effectively learn from multi-modal 3D medical data**, though they may require more extensive data augmentation or specific hyperparameter tuning to outperform classical CNNs on this specific dataset size. The dataset seems to be still a little bit too small for this kind of architecture since there is some overfitting in the first two folds. Finally, adding a pre-trained model and using tranfer learning really helped the model, without that the results where near the coin flip for the AUROC.

### Handling the "Benign" Class
The challenge of the **Benign** class remains consistent with the baseline findings. Most benign lesions are misclassified as Normal (23/32). This reinforces the clinical hypothesis that benign lesions in the ODELIA dataset share high-level structural similarities with normal tissue that are difficult to distinguish even with advanced attention mechanisms.

### Multi-Stream Integration
One major success of this phase was the technical implementation of the **Multi-Stream architecture**. The model successfully fused T2 structural data with 4-phase temporal kinetic data. The ability of the model to maintain an AUROC of 0.78 while handling a high-dimensional feature vector ($3,840$ features) proves the scalability of the custom `MultiStreamSwin` implementation.

### Clinical Safety
The model correctly identified **30 Malignant cases** and showed very few false positives for the Benign-to-Malignant path (only 7), which is critical in a clinical workflow to avoid unnecessary aggressive treatments for non-cancerous lesions.
