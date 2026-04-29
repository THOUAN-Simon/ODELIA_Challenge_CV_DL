"""
Script to visualize the training convergence and validation performance 
across all 5 cross-validation folds.
It parses the generated `training.log` file to extract loss and AUROC metrics,
computes the final statistics (mean and standard deviation), and generates a 
side-by-side plot comparing learning curves and validation AUROC.
"""

import matplotlib.pyplot as plt
import pandas as pd
import re
import os
import numpy as np

log_path = "Log_and_helper_files/training.log"
output_image = "Training_results/final_training_curves.png"

# Parses the training log file to extract loss and AUROC metrics per fold
def parse_logs(path):
    folds_data = {}
    current_fold = -1
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # Detect fold number
            fold_match = re.search(r"FOLD\s+(\d+)\s+:", line, re.IGNORECASE)
            if fold_match:
                current_fold = int(fold_match.group(1))
                if current_fold not in folds_data:
                    folds_data[current_fold] = {'epochs': [], 'loss': [], 'auc_epochs': [], 'auc': []}
            
            # Extract epoch and average loss (Regex matches the log output format from train_odelia.py)
            loss_match = re.search(r"Epoch\s+(\d+).*Average\s+loss:\s+([\d\.]+)", line, re.IGNORECASE)
            if loss_match and current_fold != -1:
                folds_data[current_fold]['epochs'].append(int(loss_match.group(1)))
                folds_data[current_fold]['loss'].append(float(loss_match.group(2)))

            # Extract Validation AUROC
            auc_match = re.search(r"AUROC\s+VAL:\s+([\d\.]+)", line, re.IGNORECASE)
            if auc_match and current_fold != -1:
                if folds_data[current_fold]['epochs']:
                    folds_data[current_fold]['auc_epochs'].append(folds_data[current_fold]['epochs'][-1])
                    folds_data[current_fold]['auc'].append(float(auc_match.group(1)))
    return folds_data

data = parse_logs(log_path)

# Extract the final AUROC value for each fold to compute overall performance
all_final_auc = [m['auc'][-1] for m in data.values() if m['auc']]

# Calculate statistics (Mean and Standard Deviation)
mean_auc = np.mean(all_final_auc)
std_auc = np.std(all_final_auc)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Plot learning curves and AUROC validation curves for each fold
for fold, m in data.items():
    c = colors[fold % len(colors)]
    ax1.plot(m['epochs'], m['loss'], label=f'Fold {fold}', color=c, alpha=0.7)
    ax2.plot(m['auc_epochs'], m['auc'], label=f'Fold {fold} (Final: {m["auc"][-1]:.3f})', color=c, marker='o', markersize=4)

ax1.set_title('Convergence of Loss', fontweight='bold')
ax2.set_title(f'Validation AUROC \nFinal Mean : {mean_auc:.3f} ± {std_auc:.3f}', fontweight='bold', color='darkblue')

# Aesthetics / Formatting
ax2.axhline(mean_auc, color='gray', linestyle='--', alpha=0.5)
ax2.fill_between(range(50), mean_auc - std_auc, mean_auc + std_auc, color='gray', alpha=0.1, label='Standard Deviation CV')
ax2.set_ylim([0.45, 0.85]); ax2.legend(loc='lower right', fontsize='small')
plt.tight_layout()
plt.savefig(output_image)
