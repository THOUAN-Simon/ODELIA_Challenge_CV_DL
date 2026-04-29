"""
Script to train a MultiStreamSwin model for breast lesion classification using 5-Fold Cross-Validation.
This version extends the baseline with the ResNet by dynamically loading all available post-contrast images.
It utilizes a Vision Transformer (ViT) based architecture with a multi-stream approach and weight sharing
(Siamese Networks) to process different post-contrast phases. The multi-stream design includes separate branches
for T2, Pre-contrast, and dynamic Post-contrast sequences.
The model leverages a [pre-trained Swin-UNETR backbone from MONAI](https://project-monai.github.io/model-zoo.html#/model/swin_unetr_btcv_segmentation) for transfer learning.
"""

import os
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
import logging
import glob
import argparse
from utils import MultiStreamSwin, PrepareTemporalKineticsd
from tqdm import tqdm
from monai.transforms import Compose, LoadImaged, ScaleIntensityd, EnsureTyped, EnsureChannelFirstd, Resized
from monai.data import Dataset, DataLoader
from monai.metrics import ROCAUCMetric
from monai.networks.utils import one_hot
from monai.losses import FocalLoss

BASE_ROOT = "/cluster/projects/vc/courses/TDT17/mic/ODELIA2025"
essential_keys = ["Pre", "Post_1", "Post_2", "Sub_1", "T2"]
device = torch.device("cuda")

#------------------------------------
# --- OUTPUT CONFIGURATION ---
#------------------------------------
# Define the output directory
output_dir = "Training_results"
os.makedirs(output_dir, exist_ok=True)

#-------------------
# --- DATA PREP ---
#-------------------
parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--out_dir', type=str, default="Training_results")
args = parser.parse_args()

FOLD_TO_USE = args.fold
# Save path defined here
model_save_path = os.path.join(output_dir, f"best_swin_odelia_fold{args.fold}.pth")

# --- LOG CONFIG ---

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
                "Post": post_paths, # List of paths (variable length, as number of post-contrast phases varies)
                "label": int(row['Lesion'])
            }
            data.append(item)
    return data

#------------------------------------
# --- SPLIT LOGIC ---
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

# --- DataLoad ---
logger.info(f"FOLD {FOLD_TO_USE} : {len(train_files)} train / {len(val_files)} val (Audit Passed)")

keys_to_load = ["T2", "Pre", "Post"]

transforms = Compose([
    LoadImaged(keys=keys_to_load, image_only=True),
    EnsureChannelFirstd(keys=keys_to_load),
    Resized(keys=keys_to_load, spatial_size=(128, 128, 64)),
    ScaleIntensityd(keys=keys_to_load),
    PrepareTemporalKineticsd(keys=["Pre", "Post"], max_phases=4),
    EnsureTyped(keys=["T2", "Kinetics", "label"])
])

train_ds = Dataset(data=train_files, transform=transforms)
val_ds = Dataset(data=val_files, transform=transforms)
# Use 4 workers for data loading
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
    batch_size=2, 
    shuffle=False, 
    num_workers=2,
    pin_memory=True
)

#-----------------
# --- MODEL ---
#-----------------
path_weights= "Log_and_helper_files/swin_unetr_btcv.pt"
model = MultiStreamSwin(num_classes=3, pretrained_path=path_weights).to(device)

#---------------------------------
# --- OPTIMIZER AND METRICS ---
#---------------------------------
# 1. Dynamic weight calculation to compensate for class imbalance
num_classes = 3

# Safely count samples per class (0: Normal, 1: Benign, 2: Malignant)
class_counts = [len(train_df[train_df['Lesion'] == c]) for c in range(num_classes)]
total_samples = len(train_df)

# Apply the formula W = N_total / (C * N_c)
# Adding 1e-8 done to prevent division by zero
weights = [total_samples / (num_classes * count + 1e-8) for count in class_counts]
class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

# 2. Inject weights into FocalLoss
logger.info(f"Class distribution (Train) : {class_counts}")
logger.info(f"Weights applied to Loss : {class_weights.cpu().numpy()}")
loss_func = FocalLoss(weight=class_weights, gamma=2.0, to_onehot_y=True)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
auc_metric = ROCAUCMetric()

# --- LOOP ---

scaler = GradScaler() # Initialize scaler for Automatic Mixed Precision (AMP)
accumulation_steps = 4 # To simulate a batch size of 4 (1 real * 4 steps)

logger.info("Starting training...")
best_auc = 0
for epoch in range(50):
    model.train()
    total_loss = 0
    optimizer.zero_grad() # Reset gradients before the batch loop
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for i, batch in enumerate(progress_bar):

        t2_inputs = batch["T2"].to(device)
        kin_inputs = batch["Kinetics"].to(device)
        labels = batch["label"].to(device).long()  

        # 5. AMP Zone: calculations in float16
        with autocast():
            outputs = model(t2_inputs, kin_inputs) # Keep raw outputs (logits)
            loss = loss_func(outputs, labels) / accumulation_steps

        # 6. Backward with the scaler
        scaler.scale(loss).backward()
        
        # 7. Update weights only every 'accumulation_steps'
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # times 'accumulation_steps' for correct log printing
        total_loss += loss.item() * accumulation_steps
    
    logger.info(f"Epoch {epoch+1} finished. Average loss: {total_loss/len(train_loader):.4f}")

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
