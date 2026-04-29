"""
Script to generate predictions and Grad-CAM heatmaps for a specific fold using the Swin-UNETR model.
It loads the trained MultiStreamSwin model, runs inference on the validation set, 
and extracts Grad-CAM heatmaps for both the static T2 modality and the dynamic 
Kinetics (Post-contrast) modality using dedicated wrappers.
Outputs are saved as JSON for probabilities and NIfTI files for the heatmaps.
"""

import os
import json
import torch
import glob
import pandas as pd
import nibabel as nib
import numpy as np
import argparse
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Resized, ScaleIntensityd, EnsureTyped
from monai.data import DataLoader, Dataset
from monai.visualize import GradCAM
from utils import MultiStreamSwin, PrepareTemporalKineticsd 

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

# Create fold-specific subdirectory for Grad-CAM heatmaps
heatmaps_dir = os.path.join(output_dir, "heatmaps", f"heatmap_fold{args.fold}")
os.makedirs(heatmaps_dir, exist_ok=True)

model_path = os.path.join(output_dir, f"best_swin_odelia_fold{args.fold}.pth")
output_file = os.path.join(output_dir, f"predictions_fold{args.fold}.json")

#---------------------------
# --- MODEL LOADING ---
#---------------------------
# MultiStreamSwin
model = MultiStreamSwin(num_classes=3).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# --- WRAPPERS FOR GRAD-CAM ---
class T2Wrapper(torch.nn.Module):
    def __init__(self, full_model, kinetics_fixed):
        super().__init__()
        self.full_model = full_model
        self.kin_fixed = kinetics_fixed
    def forward(self, t2):
        return self.full_model(t2, self.kin_fixed)
    
class KinPhaseWrapper(torch.nn.Module):
    def __init__(self, full_model, t2_fixed, kinetics_base, phase_idx=1):
        super().__init__()
        self.full_model = full_model
        self.t2_fixed = t2_fixed
        self.kin_base = kinetics_base.clone()
        self.phase_idx = phase_idx
    def forward(self, phase_img):
        # We replace only the target phase to allow the gradient to pass through
        # phase_img: (B, 1, H, W, D) -> Inject it into the tensor (B, 4, 1, H, W, D)
        current_kin = self.kin_base.clone()
        current_kin[:, self.phase_idx, ...] = phase_img
        return self.full_model(self.t2_fixed, current_kin)

#-------------------------------------------------------------
# --- DATA PREPARATION (Validation and test sets) ---
#-------------------------------------------------------------
df_meta = pd.read_csv("Log_and_helper_files/odelia_merged_metadata.csv")
df_split = pd.read_csv("/cluster/projects/vc/courses/TDT17/mic/ODELIA2025/split_unilateral.csv")

map_val = df_split[(df_split['Fold'] == args.fold) & (df_split['Split'] == 'val')]
val_df = pd.merge(df_meta, map_val[['UID']], on='UID', how='inner')

print(f"Inference Fold {args.fold} : {len(val_df)} patients to predict.")

# Dynamic loading as during training
data_list = []
for _, row in val_df.iterrows():
    p_dir = os.path.join(BASE_ROOT, "data", row['Institution'], "data_unilateral", row['UID'])
    
    post_paths = sorted(glob.glob(os.path.join(p_dir, "Post_*.nii.gz")))
    t2_path = os.path.join(p_dir, "T2.nii.gz")
    pre_path = os.path.join(p_dir, "Pre.nii.gz")

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
transforms = Compose([
    LoadImaged(keys=["T2", "Pre", "Post"], image_only=True),
    EnsureChannelFirstd(keys=["T2", "Pre", "Post"]),
    Resized(keys=["T2", "Pre", "Post"], spatial_size=(128, 128, 64)),
    ScaleIntensityd(keys=["T2", "Pre", "Post"]),
    PrepareTemporalKineticsd(keys=["Pre", "Post"], max_phases=4),
    EnsureTyped(keys=["T2", "Kinetics"])
])

loader = DataLoader(Dataset(data=data_list, transform=transforms), batch_size=1)

#------------------
# --- INFERENCE ---
#------------------
predictions = {}
print("Generating predictions and Heatmaps...")

dummy_t2 = torch.zeros((1, 1, 128, 128, 64)).to(device)
dummy_kin = torch.zeros((1, 4, 1, 128, 128, 64)).to(device)

wrapper_t2 = T2Wrapper(model, dummy_kin)
wrapper_kin = KinPhaseWrapper(model, dummy_t2, dummy_kin, phase_idx=1)

cam_t2_exec = GradCAM(nn_module=wrapper_t2, target_layers=["full_model.encoder_t2.swinViT.layers4.0"])
cam_kin_exec = GradCAM(nn_module=wrapper_kin, target_layers=["full_model.encoder_kinetics.swinViT.layers4.0"])

for batch in loader:
    uid = batch["UID"][0]
    t2 = batch["T2"].to(device)
    kinetics = batch["Kinetics"].to(device)

    # A. PREDICTION (Required to get pred_class before the heatmap)
    with torch.no_grad():
        outputs = model(t2, kinetics)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_class = int(np.argmax(probs)) #

    # B. UPDATE WRAPPERS
    # Inject current patient data into the already created wrappers
    wrapper_t2.kin_fixed = kinetics
    wrapper_kin.t2_fixed = t2
    wrapper_kin.kin_base = kinetics.clone()

    # C. HEATMAP GENERATION
    # Enable gradients only here
    with torch.set_grad_enabled(True):
        # T2
        t2_grad = t2.clone().requires_grad_(True)
        heatmap_t2 = cam_t2_exec(x=t2_grad, class_idx=pred_class)
        
        # Kinetics (Phase 1)
        kin_phase = kinetics[:, 1, ...].clone().requires_grad_(True)
        heatmap_kin = cam_kin_exec(x=kin_phase, class_idx=pred_class)

    # D. SAVE
    t2_vol = t2[0, 0, ...].cpu().numpy()
    nib.save(nib.Nifti1Image(t2_vol, np.eye(4)), os.path.join(heatmaps_dir, f"t2_aligned_{uid}.nii.gz"))
    
    h_t2 = heatmap_t2.squeeze().detach().cpu().numpy()
    nib.save(nib.Nifti1Image(h_t2, np.eye(4)), os.path.join(heatmaps_dir, f"heatmap_T2_{uid}.nii.gz"))
    
    h_kin = heatmap_kin.squeeze().detach().cpu().numpy()
    nib.save(nib.Nifti1Image(h_kin, np.eye(4)), os.path.join(heatmaps_dir, f"heatmap_kin_{uid}.nii.gz"))

    predictions[uid] = {"prediction": pred_class, "probabilities": probs.tolist()}

#----------------------
# --- SAVE RESULTS ---
#----------------------
with open(output_file, "w") as f:
    json.dump(predictions, f, indent=4)

print(f"✅ Done ! {len(predictions)} predictions saved in : {output_file}")
