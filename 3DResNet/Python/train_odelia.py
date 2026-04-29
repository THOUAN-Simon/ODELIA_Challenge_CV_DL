"""
Script to train a 3D-ResNet model for breast lesion classification using 5-Fold Cross-Validation.
It partitions the dataset into 5 folds (A, B, C, D, E).
For example:
Fold 0: Validation = A, Train = B+C+D+E.
Fold 1: Validation = B, Train = A+C+D+E.
"""

import os
import torch
import pandas as pd
import logging
from tqdm import tqdm
from monai.transforms import Compose, LoadImaged, ConcatItemsd, ScaleIntensityd, EnsureTyped, EnsureChannelFirstd, Resized
from monai.data import CacheDataset, DataLoader
from monai.networks.nets import ResNet
from monai.metrics import ROCAUCMetric
from monai.networks.utils import one_hot
import argparse

essential_keys = ["Pre", "Post_1", "Post_2", "Sub_1", "T2"]
device = torch.device("cuda")

#------------------------------------
# --- OUTPUT CONFIGURATION ---
#------------------------------------

# --- DATA PREP ---
parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--out_dir', type=str, default="Training_results")
parser.add_argument('--data_dir', type=str, default="/cluster/projects/vc/courses/TDT17/mic/ODELIA2025", help="Path to the dataset root")
parser.add_argument('--split_csv', type=str, default="Log_and_helper_files/split_unilateral.csv", help="Path to the split CSV file")
parser.add_argument('--metadata_csv', type=str, default="Log_and_helper_files/odelia_merged_metadata.csv")
args = parser.parse_args()

output_dir = args.out_dir
os.makedirs(output_dir, exist_ok=True)

BASE_ROOT = args.data_dir
FOLD_TO_USE = args.fold
# Define the model save path
model_save_path = os.path.join(output_dir, f"best_resnet_odelia_fold{args.fold}.pth")

# --- LOG CONFIGURATION ---
log_path = os.path.join(output_dir, "training.log")
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', 
                    handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
logger = logging.getLogger()

def get_data_dict(subset):
    data = []
    for _, row in subset.iterrows():
        p_dir = os.path.join(BASE_ROOT, "data", row['Institution'], "data_unilateral", row['UID'])
        files = {k: os.path.join(p_dir, f"{k}.nii.gz") for k in essential_keys}
        if all(os.path.exists(f) for f in files.values()):
            files["label"] = int(row['Lesion'])
            data.append(files)
    return data

# --- SPLIT LOGIC ---

# 1. Load metadata (Master Dataset of 478 unique patients without Fold/Split)
df_meta = pd.read_csv(args.metadata_csv)

# 2. Load the official split map
split_path = args.split_csv
df_split = pd.read_csv(split_path)

# 3. Filter and Merge 
current_split_map = df_split[df_split['Fold'] == FOLD_TO_USE]
df_current_fold = pd.merge(df_meta, current_split_map[['UID', 'Split']], on='UID', how='inner')

# 4. Create Train and Val datasets
train_df = df_current_fold[df_current_fold['Split'] == 'train']
val_df = df_current_fold[df_current_fold['Split'] == 'val']

train_files = get_data_dict(train_df)
val_files = get_data_dict(val_df)

logger.info(f"FOLD {FOLD_TO_USE} : {len(train_files)} train / {len(val_files)} val (Audit Passed)")

transforms = Compose([
    LoadImaged(keys=essential_keys, image_only=True),
    EnsureChannelFirstd(keys=essential_keys),
    Resized(keys=essential_keys, spatial_size=(128, 128, 64)),
    ScaleIntensityd(keys=essential_keys),
    ConcatItemsd(keys=essential_keys, name="image"),
    EnsureTyped(keys=["image", "label"])
])

# Use 8 workers to speed up data loading
train_ds = CacheDataset(data=train_files, transform=transforms, cache_rate=0.1)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=8) # Batch size set to 4 utilizing 32GB RAM
val_loader = DataLoader(CacheDataset(data=val_files, transform=transforms, cache_rate=0.1), batch_size=2)

# --- MODEL ---
model = ResNet(
    block="basic", 
    layers=[2, 2, 2, 2], 
    block_inplanes=[64, 128, 256, 512],
    n_input_channels=5, 
    num_classes=3, 
    spatial_dims=3
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_func = torch.nn.CrossEntropyLoss()
auc_metric = ROCAUCMetric()

# --- LOOP ---
logger.info("Starting training...")
best_auc = 0
for epoch in range(50):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch in progress_bar:
        inputs, labels = batch["image"].to(device), batch["label"].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})

    logger.info(f"Epoch {epoch+1} finished. Average loss: {total_loss/len(train_loader):.4f}")

    if (epoch + 1) % 2 == 0:
        model.eval()
        with torch.no_grad():
            y_pred, y_true = [], []
            for v_batch in val_loader:
                v_in, v_lab = v_batch["image"].to(device), v_batch["label"].to(device)
                y_pred.append(model(v_in).softmax(dim=1))
                y_true.append(one_hot(v_lab, num_classes=3))
            y_pred = torch.cat(y_pred, dim=0); y_true = torch.cat(y_true, dim=0)
            auc_metric(y_pred, y_true); val_auc = auc_metric.aggregate().item(); auc_metric.reset()
            logger.info(f"--- AUROC VAL: {val_auc:.4f} ---")
            if val_auc > best_auc:
                best_auc = val_auc
                # Uses the full path defined earlier
                torch.save(model.state_dict(), model_save_path)
                logger.info(f"Best model saved to : {model_save_path}")
