"""
Script to generate predictions and Grad-CAM heatmaps for a specific fold using the MultiStreamResNet model (Phase 3).
It loads the trained model, runs inference on the validation set, and extracts Grad-CAM heatmaps 
for the T2 modality to explain predictions.
Outputs include JSON files for classification probabilities and NIfTI files for the 
T2 volume and corresponding Grad-CAM heatmaps for visualization in 3D Slicer.
"""

import os
import json
import torch
import glob
import pandas as pd
import numpy as np
import nibabel as nib # Required to save the 3D map as .nii.gz
import argparse
import torch.nn as nn

from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Resized, ScaleIntensityd, EnsureTyped, CropForegroundd
from monai.data import DataLoader, Dataset
from monai.visualize import GradCAM # L'outil magique d'Explainable AI
from utils import MultiStreamResNet, PrepareTemporalKineticsd 

#---------------
# --- CONFIG ---
#---------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_ROOT = "/cluster/projects/vc/courses/TDT17/mic/ODELIA2025"

#------------------------------------
# --- OUTPUT CONFIGURATION ---
#------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--out_dir', type=str, default="Training_results")
args = parser.parse_args()

output_dir = args.out_dir
os.makedirs(output_dir, exist_ok=True)

# Create fold-specific subdirectory for heatmaps
heatmaps_dir = os.path.join(output_dir, "heatmaps", f"heatmap_fold{args.fold}")
os.makedirs(heatmaps_dir, exist_ok=True)

model_path = os.path.join(output_dir, f"best_resnet_odelia_fold{args.fold}.pth")
output_file = os.path.join(output_dir, f"predictions_fold{args.fold}.json")

#---------------------------
# --- MODEL LOADING ---
#---------------------------
# V3: Load the new ResNet + LSTM architecture
model = MultiStreamResNet(num_classes=3, hidden_dim=256).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

#-------------------------------------------------------------
# --- DATA PREPARATION (Validation and Test sets) ---
#-------------------------------------------------------------
df_meta = pd.read_csv("Log_and_helper_files/odelia_merged_metadata.csv")
df_split = pd.read_csv("/cluster/projects/vc/courses/TDT17/mic/ODELIA2025/split_unilateral.csv")

map_val = df_split[(df_split['Fold'] == args.fold) & (df_split['Split'] == 'val')]
val_df = pd.merge(df_meta, map_val[['UID']], on='UID', how='inner')

print(f"Inférence Fold {args.fold} : {len(val_df)} patients à prédire.")
print(f"Inference Fold {args.fold} : {len(val_df)} patients to predict.")

data_list = []
for _, row in val_df.iterrows():
    p_dir = os.path.join(BASE_ROOT, "data", row['Institution'], "data_unilateral", row['UID'])
    
    post_paths = sorted(glob.glob(os.path.join(p_dir, "Post_*.nii.gz"))) # Sorted to ensure temporal order
    t2_path = os.path.join(p_dir, "T2.nii.gz") # T2 path
    pre_path = os.path.join(p_dir, "Pre.nii.gz") # Pre-contrast path

    if os.path.exists(t2_path) and os.path.exists(pre_path) and len(post_paths) > 0:
        data_list.append({
            "UID": row['UID'],
            "T2": t2_path,
            "Pre": pre_path,
            "Post": post_paths
        })

keys_to_load = ["T2", "Pre", "Post"]

#----------------------------
# --- TRANSFORMS PIPELINE ---
#----------------------------
# Identical to the new training (Crop + no max_phases)
transforms = Compose([
    LoadImaged(keys=keys_to_load, image_only=True),
    EnsureChannelFirstd(keys=keys_to_load),
    # CropForegroundd(keys=keys_to_load, source_key="Pre", margin=5),
    Resized(keys=keys_to_load, spatial_size=(128, 128, 64)),
    ScaleIntensityd(keys=keys_to_load),
    PrepareTemporalKineticsd(keys=["Pre", "Post"]),
    EnsureTyped(keys=["T2", "Kinetics"])
])

loader = DataLoader(Dataset(data=data_list, transform=transforms), batch_size=1)

#------------------
# --- INFERENCE ---
#------------------
predictions = {}
print("Generating predictions and Grad-CAM maps...")

# Freeze the entire model to prevent modification
for param in model.parameters():
    param.requires_grad = False

for batch in loader:
    uid = batch["UID"][0]
    
    t2_inputs = batch["T2"].to(device)
    kin_inputs = batch["Kinetics"].to(device)
    
    # --------------------------------------------------
    # STEP 1: Standard Prediction (without gradients)
    # --------------------------------------------------
    with torch.no_grad():
        outputs = model(t2_inputs, kin_inputs)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_class = int(torch.argmax(outputs, dim=1).item())
        
        predictions[uid] = {
            "prediction": pred_class,
            "probabilities": probs.tolist() 
        }

    # -----------------------------------------------------
    # STEP 2: EXPLAINABLE AI (Grad-CAM)
    # -----------------------------------------------------
    # Temporarily reactivate gradients just for the image
    with torch.set_grad_enabled(True):
        # --- Save aligned T2 (For 3D Slicer) ---
        # Retrieve T2 as it comes out of the DataLoader (resized to 128x128x64)
        t2_vol_processed = t2_inputs.squeeze().cpu().numpy()
        # Use affine=np.eye(4) so it's on the same reference frame as the heatmap
        t2_nii_processed = nib.Nifti1Image(t2_vol_processed, affine=np.eye(4))
        t2_aligned_filename = os.path.join(heatmaps_dir, f"t2_aligned_{uid}.nii.gz")
        nib.save(t2_nii_processed, t2_aligned_filename)
        
        t2_cam = t2_inputs.clone().requires_grad_(True)
        
        class GradCamWrapper(nn.Module):
            def __init__(self, base_model, kin_fixed):
                super().__init__()
                self.base_model = base_model
                self.kin_fixed = kin_fixed
            def forward(self, x_t2):
                return self.base_model(x_t2, self.kin_fixed)

        wrapper = GradCamWrapper(model, kin_inputs) # Wrap the model to make it compatible with GradCAM
        
        # Target the very last convolutional layer of the T2 encoder (layer4)
        cam = GradCAM(nn_module=wrapper, target_layers="base_model.encoder_t2.layer4")
        
        # Generate the 3D Heatmap for the predicted class
        heatmap = cam(x=t2_cam, class_idx=pred_class)
        
        # Clean up the tensor: (1, 1, 128, 128, 64) -> (128, 128, 64)
        heatmap_vol = heatmap.squeeze().detach().cpu().numpy()
        
        # Save in NIfTI format
        nii_img = nib.Nifti1Image(heatmap_vol, affine=np.eye(4))
        cam_filename = os.path.join(heatmaps_dir, f"heatmap_fold{args.fold}_{uid}.nii.gz")
        nib.save(nii_img, cam_filename)

#----------------------
# --- SAVE RESULTS ---
#----------------------
with open(output_file, "w") as f:
    json.dump(predictions, f, indent=4)

print(f"Done! {len(predictions)} predictions and Grad-CAM maps saved in : {output_dir}")
