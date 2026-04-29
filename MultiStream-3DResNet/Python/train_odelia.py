"""
Script to train the MultiStreamResNet model for breast lesion classification (Phase 3).
Returns to a ResNet-based backbone but maintains the multi-stream temporal processing 
introduced in Phase 2. For detailed differences with the baseline, see the README.
"""

import os
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
import logging
import glob
import argparse
from utils import PrepareTemporalKineticsd, ModalityDropoutd, MultiStreamResNet
from tqdm import tqdm
from monai.transforms import (
    Compose, LoadImaged, ScaleIntensityd, EnsureTyped, 
    EnsureChannelFirstd, Resized, RandFlipd, RandAffined, RandGaussianNoised
)
from monai.networks.nets import ResNet
from monai.losses import FocalLoss
from monai.data import Dataset, DataLoader
from monai.metrics import ROCAUCMetric
from monai.networks.utils import one_hot

torch.cuda.empty_cache()

BASE_ROOT = "/cluster/projects/vc/courses/TDT17/mic/ODELIA2025"
essential_keys = ["Pre", "Post_1", "Post_2", "Sub_1", "T2"]
device = torch.device("cuda")

#------------------------------------
# --- OUTPUT CONFIGURATION ---
#------------------------------------
# Define the output directory
output_dir = "Training_results"
os.makedirs(output_dir, exist_ok=True)

def main():
    # --- DATA PREPARATION ---
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--out_dir', type=str, default="Training_results")
    args = parser.parse_args()

    output_dir = args.out_dir
    os.makedirs(output_dir, exist_ok=True)

    FOLD_TO_USE = args.fold
    # Define the model save path here
    model_save_path = os.path.join(output_dir, f"best_resnet_odelia_fold{args.fold}.pth")

    # --- LOG CONFIGURATION ---

    log_path = os.path.join(output_dir, "training.log")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', 
                        handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
    logger = logging.getLogger()

    def get_dynamic_data_dict(subset):
        data = []
        for _, row in subset.iterrows():
            p_dir = os.path.join(BASE_ROOT, "data", row['Institution'], "data_unilateral", row['UID'])
            
            # 1. Search for all available Post-contrast phases (e.g., Post_1, Post_2, Post_3, Post_4...)
            post_paths = sorted(glob.glob(os.path.join(p_dir, "Post_*.nii.gz")))
            t2_path = os.path.join(p_dir, "T2.nii.gz")
            pre_path = os.path.join(p_dir, "Pre.nii.gz")

            # Minimum condition for the folder to be valid
            if os.path.exists(t2_path) and os.path.exists(pre_path) and len(post_paths) > 0:
                item = {
                    "UID": row['UID'],
                    "T2": t2_path,
                    "Pre": pre_path,
                    "Post": post_paths, # List of paths (variable length, as the number of post-contrast phases varies)
                    "label": int(row['Lesion'])
                }
                data.append(item)
        return data

    #------------------------------------
    # --- SPLIT LOGIC (AUDIT VALIDATED) ---
    #------------------------------------
    # Load metadata
    df_meta = pd.read_csv("Log_and_helper_files/odelia_merged_metadata.csv")

    # Load the official split map
    split_path = "/cluster/projects/vc/courses/TDT17/mic/ODELIA2025/split_unilateral.csv"
    df_split = pd.read_csv(split_path)

    # Filter and Merge
    current_split_map = df_split[df_split['Fold'] == FOLD_TO_USE]
    df_current_fold = pd.merge(df_meta, current_split_map[['UID', 'Split']], on='UID', how='inner')

    # Create Train and Val datasets
    train_df = df_current_fold[df_current_fold['Split'] == 'train']
    val_df = df_current_fold[df_current_fold['Split'] == 'val']

    train_files = get_dynamic_data_dict(train_df)
    val_files = get_dynamic_data_dict(val_df)

    logger.info(f"FOLD {FOLD_TO_USE} : {len(train_files)} train / {len(val_files)} val (Audit Passed)")

    # --- DATA LOADING ---

    keys_to_load = ["T2", "Pre", "Post"]

    # Separate Train and Val to avoid adding noise during validation
    val_transforms = Compose([
        LoadImaged(keys=keys_to_load, image_only=True),
        EnsureChannelFirstd(keys=keys_to_load),
        # CropForegroundd disabled for V3.2 testing
        Resized(keys=keys_to_load, spatial_size=(128, 128, 64)),
        ScaleIntensityd(keys=keys_to_load),
        PrepareTemporalKineticsd(keys=["Pre", "Post"]),
        EnsureTyped(keys=["T2", "Kinetics", "label"])
    ])

    train_transforms = Compose([
        LoadImaged(keys=keys_to_load, image_only=True),
        EnsureChannelFirstd(keys=keys_to_load),
        # 1. Spatial Augmentation (Crucial for Benign class)
        RandFlipd(keys=["T2", "Pre", "Post"], spatial_axis=[0], prob=0.5),
        RandAffined(
            keys=["T2", "Pre", "Post"],
            prob=0.5,
            rotate_range=(0.1, 0.1, 0.1),
            scale_range=(0.1, 0.1, 0.1),
            mode="bilinear"
        ),
        Resized(keys=keys_to_load, spatial_size=(128, 128, 64)),
        ScaleIntensityd(keys=keys_to_load),
        # 2. Noise for robustness
        RandGaussianNoised(keys=["T2", "Post"], prob=0.1, mean=0.0, std=0.05),
        PrepareTemporalKineticsd(keys=["Pre", "Post"]),
        ModalityDropoutd(keys=["T2", "Kinetics"], prob=0.20),
        EnsureTyped(keys=["T2", "Kinetics", "label"])
    ])

    # Use 4 workers to speed up loading
    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)
    train_loader = DataLoader(
        train_ds, 
        batch_size=1, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        prefetch_factor=2
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=1, # Batch size 1 is required because tensors can now have different temporal lengths.
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )


    # --- MODEL ---
    model = MultiStreamResNet(num_classes=3, hidden_dim=256).to(device)

    # --- OPTIMIZER AND METRICS ---
    # 1. Dynamic weight calculation to compensate for class imbalance
    num_classes = 3

    # Safely count samples per class (0: Normal, 1: Benign, 2: Malignant)
    class_counts = [len(train_df[train_df['Lesion'] == c]) for c in range(num_classes)]
    total_samples = len(train_df)

    # Apply the formula W = N_total / (C * N_c)
    # Adding 1e-8 is a common safety measure to prevent division by zero
    weights = [total_samples / (num_classes * count + 1e-8) for count in class_counts]
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    # --- OPTIMIZER AND FOCAL LOSS ---
    # Focal Loss to force the model to stop ignoring difficult cases (Benign)
    loss_func = FocalLoss(weight=class_weights, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    logger.info(f"Class distribution (Train) : {class_counts}")
    logger.info(f"Weights applied to Loss : {class_weights.cpu().numpy()}")

    auc_metric = ROCAUCMetric()

    # --- TRAINING LOOP ---

    scaler = torch.amp.GradScaler('cuda') # Initialize the scaler for Automatic Mixed Precision (AMP)
    accumulation_steps = 4 # To simulate a batch size of 4 (1 real * 4 steps)

    logger.info("Starting training...")
    best_auc = 0
    for epoch in range(50):
        model.train()
        total_loss = 0
        optimizer.zero_grad() # Reset gradients BEFORE the batch loop
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for i, batch in enumerate(progress_bar):

            t2_inputs = batch["T2"].to(device)
            kin_inputs = batch["Kinetics"].to(device)
            labels = batch["label"].to(device)        

            # 5. AMP Zone: calculations in float16
            with torch.amp.autocast('cuda'):
                outputs = model(t2_inputs, kin_inputs) 
                # Transform the label index (e.g., 1) into a one-hot vector (e.g., [0, 1, 0])
                labels_one_hot = one_hot(labels, num_classes=3)
                loss = loss_func(outputs, labels_one_hot) / accumulation_steps

            # 6. Backward with the scaler
            scaler.scale(loss).backward()
            
            # 7. Update weights ONLY every 'accumulation_steps'
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # Multiply by accumulation_steps for correct log display
            total_loss += loss.item() * accumulation_steps
        
        scheduler.step()
        perte_moyenne = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} finished. Average loss: {perte_moyenne:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        if (epoch + 1) % 2 == 0:
            model.eval()
            with torch.no_grad():
                y_pred, y_true = [], []
                for v_batch in val_loader:
                    # 1. Separate streams for validation as well
                    v_t2 = v_batch["T2"].to(device)
                    v_kin = v_batch["Kinetics"].to(device)
                    v_lab = v_batch["label"].to(device)
                    
                    # 2. Pass both streams to the model
                    outputs = model(v_t2, v_kin)
                    y_pred.append(outputs.softmax(dim=1))
                    
                    y_true.append(one_hot(v_lab, num_classes=3))
                    
                y_pred = torch.cat(y_pred, dim=0) 
                y_true = torch.cat(y_true, dim=0)
                
                auc_metric(y_pred, y_true)
                val_auc = auc_metric.aggregate().item()
                auc_metric.reset()
                
                logger.info(f"--- Validation AUROC: {val_auc:.4f} ---")
                if val_auc > best_auc:
                    best_auc = val_auc
                    torch.save(model.state_dict(), model_save_path)
                    logger.info(f"Best model saved to : {model_save_path}")

if __name__ == "__main__":
    main()
