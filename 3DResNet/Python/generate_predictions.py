"""
Script to generate predictions and Grad-CAM heatmaps for a specific fold.
It loads the trained 3D-ResNet model, runs inference on the validation set,
saves the classification probabilities to a JSON file, and exports the 
corresponding Grad-CAM heatmaps and aligned T2 volumes as NIfTI files for 
visualization and analysis in 3D Slicer.
"""

import os
import json
import torch
import pandas as pd
import numpy as np
import nibabel as nib
import argparse
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Resized, ScaleIntensityd, ConcatItemsd, EnsureTyped
from monai.data import DataLoader, Dataset
from monai.networks.nets import ResNet
from monai.visualize import GradCAM

#---------------
# --- CONFIG ---
#---------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
essential_keys = ["Pre", "Post_1", "Post_2", "Sub_1", "T2"]

#------------------------------------
# --- OUTPUT CONFIGURATION ---
#------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--out_dir', type=str, default="Training_results")
parser.add_argument('--data_dir', type=str, default="/cluster/projects/vc/courses/TDT17/mic/ODELIA2025", help="Path to the dataset root")
parser.add_argument('--split_csv', type=str, default="Log_and_helper_files/split_unilateral.csv", help="Path to the split CSV file")
parser.add_argument('--metadata_csv', type=str, default="Log_and_helper_files/odelia_merged_metadata.csv")
args = parser.parse_args()

BASE_ROOT = args.data_dir
output_dir = args.out_dir
os.makedirs(output_dir, exist_ok=True)

# Create fold-specific subdirectory for Grad-CAM heatmaps
heatmaps_dir = os.path.join(output_dir, "heatmaps", f"heatmap_fold{args.fold}")
os.makedirs(heatmaps_dir, exist_ok=True)

model_path = os.path.join(output_dir, f"best_resnet_odelia_fold{args.fold}.pth")
output_file = os.path.join(output_dir, f"predictions_fold{args.fold}.json")

#---------------------------
# --- MODEL LOADING ---
#---------------------------
model = ResNet(
    block="basic", 
    layers=[2, 2, 2, 2], 
    block_inplanes=[64, 128, 256, 512],
    n_input_channels=5, 
    num_classes=3, 
    spatial_dims=3
).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

cam = GradCAM(nn_module=model, target_layers="layer4")

#-------------------------------------------------------------
# --- DATA PREPARATION (Validation and Test sets) ---
#-------------------------------------------------------------

df_meta = pd.read_csv(args.metadata_csv)
df_split = pd.read_csv(args.split_csv)

# Filter validation instructions specifically for THIS fold
map_val = df_split[(df_split['Fold'] == args.fold) & (df_split['Split'] == 'val')]
val_df = pd.merge(df_meta, map_val[['UID']], on='UID', how='inner')

print(f" Inference Fold {args.fold} : {len(val_df)} patients to predict.")

data_list = []
for _, row in val_df.iterrows():
    p_dir = os.path.join(BASE_ROOT, "data", row['Institution'], "data_unilateral", row['UID'])
    files = {k: os.path.join(p_dir, f"{k}.nii.gz") for k in essential_keys}
    if all(os.path.exists(f) for f in files.values()):
        data_list.append({"UID": row['UID'], **files})

transforms = Compose([
    LoadImaged(keys=essential_keys, image_only=True),
    EnsureChannelFirstd(keys=essential_keys),
    Resized(keys=essential_keys, spatial_size=(128, 128, 64)),
    ScaleIntensityd(keys=essential_keys),
    ConcatItemsd(keys=essential_keys, name="image"),
    EnsureTyped(keys=["image"])
])

loader = DataLoader(Dataset(data=data_list, transform=transforms), batch_size=1)

#------------------
# --- INFERENCE ---
#------------------
predictions = {}
print("Generating predictions...")

with torch.no_grad():
    for batch in loader:
        uid = batch["UID"][0]
        inputs = batch["image"].to(device) # Tensor shape: (1, 5, 128, 128, 64)

        # STEP 1: Standard Prediction
        with torch.no_grad():
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()

        # STEP 2: Explainable AI (Grad-CAM)
        # Enable gradients on the input tensor to allow CAM computation
        inputs_grad = inputs.clone().requires_grad_(True)
        
        with torch.set_grad_enabled(True):
            # Generate the Heatmap for the predicted class
            heatmap = cam(x=inputs_grad, class_idx=pred_class)
            
            # Clean up and convert to numpy array for saving (128, 128, 64)
            heatmap_vol = heatmap.squeeze().detach().cpu().numpy()
            
            # Extract only the T2 channel (Index 4) for alignment in 3D Slicer
            # In the ConcatItemsd transform, T2 corresponds to index 4
            t2_vol_processed = inputs[0, 4, :, :, :].cpu().numpy()

        # STEP 3: NIfTI Saves (Using identity affine eye(4) for 3D Slicer compatibility)
        # 1. Save the aligned T2 volume
        t2_nii = nib.Nifti1Image(t2_vol_processed, affine=np.eye(4))
        t2_filename = os.path.join(heatmaps_dir, f"t2_aligned_{uid}.nii.gz")
        nib.save(t2_nii, t2_filename)

        # 2. Save the Grad-CAM Heatmap
        cam_nii = nib.Nifti1Image(heatmap_vol, affine=np.eye(4))
        cam_filename = os.path.join(heatmaps_dir, f"heatmap_fold{args.fold}_{uid}.nii.gz")
        nib.save(cam_nii, cam_filename)
        
        # Expected JSON format
        predictions[uid] = {
            "prediction": int(torch.argmax(outputs, dim=1).item()),
            "probabilities": probs[0].tolist() # [Normal, Benign, Malignant]
        }

#----------------------
# --- SAVE RESULTS ---
#----------------------
with open(output_file, "w") as f:
    json.dump(predictions, f, indent=4)

print(f"✅ Done! {len(predictions)} predictions saved in: {output_file}")
