# Evaluation Report: Advanced Multi-Stream 3D-ResNet (Phase 3)

This document provides an objective analysis of the Phase 3 model performance. Despite several architectural innovations aimed at stabilizing the training process, this iteration shows a **significant regression** in generalization capabilities compared to previous phases.

## 1. Cross-Validation Stability (per Fold)

The learning curves reveal chronic instability. The mean validation AUROC across the 5 folds is near statistical randomness (0.5), highlighting the model's inability to extract robust descriptors from the multi-modal input.

| Fold | Best Validation AUROC | Observation |
| :--- | :---: | :--- |
| **Fold 0** | 0.526 | Performance near random chance |
| **Fold 1** | 0.515 | Major instability (early peak followed by drop) |
| **Fold 2** | 0.513 | Training failure / No convergence |
| **Fold 3** | 0.545 | Marginal signal detected |
| **Fold 4** | 0.682 | Only fold showing a positive learning trend |
| --- | --- | --- |
| **Mean CV** | **0.556 ± 0.064** | **Global Convergence Failure** |

## 2. Final Ensemble Results (304 Patients)

While the ensemble effect slightly improves the global AUROC compared to the individual fold average, the metrics remain insufficient for any clinical consideration.

* **Final AUROC (Malignant vs Rest):** **0.6673** (Compared to 0.8174 in Phase 1)
* **Sensitivity (@ 90% Specificity):** 27.42%
* **Specificity (@ 90% Sensitivity):** 30.17%

## 3. Confusion Matrix Analysis (Ensemble)

The confusion matrix reveals a collapse in the model's discriminative power, characterized by an extreme bias toward the majority class.

| True \ Predicted | Normal | Benign | Malignant |
| :--- | :---: | :---: | :---: |
| **Normal** | **208** | 0 | 2 |
| **Benign** | 32 | **0** | 0 |
| **Malignant** | 58 | 0 | **4** |

### Key Observations:
* **Majority Class Bias:** The model has almost entirely ceased predicting the "Benign" and "Malignant" classes. Out of 304 patients, only 6 lesions were predicted.
* **Benign Class Failure:** 100% of benign cases were misclassified as normal.
* **Critical Sensitivity Gap:** 58 out of 62 malignant cases were ignored (False Negatives), representing an unacceptable clinical risk.

## 4. Technical Discussion & Root Cause Analysis

Post-training analysis suggests several hypotheses for this performance regression:

1.  **The Instance Normalization Trade-off:** While InstanceNorm stabilized the gradient flow for a batch size of 1, it likely stripped away global distribution information ("style" features) that was essential for distinguishing pathological tissue from healthy parenchyma.
2.  **Temporal Signal Dilution:** The transition to a multi-stream architecture with Adaptive Mean Pooling appears to have diluted precise structural features from the T2 stream in favor of noisy or poorly learned kinetic features.
3.  **Complexity vs. Data Volume:** The increased parameter count and stream separation made the search space too vast for the dataset size (~400 patients), leading to "noise memorization" rather than semantic learning.

## 5. Conclusion: Occam’s Razor

Phase 3 serves as a textbook example of **Occam’s Razor**.

By attempting to solve mathematical bottlenecks through architectural sophistication (multi-stream, InstanceNorm, dynamic fusion), we introduced complexity that hindered the capture of the biological signal. The simpler models (Phase 1: Baseline ResNet-18 with 5-channel concatenation) remain the most performant and robust. 

Technical sophistication cannot compensate for the lack of a clear signal-to-noise ratio in complex 3D volumes. Moving forward, a return to the foundations of Phase 1, enriched with improved data augmentation, appears to be the most rational path.
