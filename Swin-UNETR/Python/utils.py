"""
Utility classes and transformations used by the Swin-UNETR training and inference scripts.
Contains the `MultiStreamSwin` architecture for multi-modal processing (T2 + Temporal Kinetics) 
and the `PrepareTemporalKineticsd` MONAI transform for handling variable-length dynamic contrast-enhanced sequences.
"""

import torch
import os
import torch.nn as nn
from monai.networks.nets import SwinUNETR
from monai.transforms import MapTransform

class MultiStreamSwin(nn.Module):
    def __init__(self, num_classes=3, pretrained_path=None): 
        super().__init__()
        
        # --- 1. BACKBONES (Spatial Extractors) ---
        self.encoder_t2 = SwinUNETR(spatial_dims=3, in_channels=1, out_channels=1, feature_size=48)
        self.encoder_kinetics = SwinUNETR(spatial_dims=3, in_channels=1, out_channels=1, feature_size=48)
        
        # --- LOAD PRE-TRAINED WEIGHTS ---
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"Loading pre-trained weights from: {pretrained_path}")
            # map_location="cpu" avoids VRAM overload during initial loading
            weights = torch.load(pretrained_path, map_location="cpu") 
        
            # Retrieve the architecture of our current encoder
            model_dict = self.encoder_t2.state_dict()
            
            # Keep only weights that have exactly the same shape
            # as in our model.
            pretrained_dict = {
                k: v for k, v in weights.items() 
                if k in model_dict and v.shape == model_dict[k].shape
            }
            
            # Load this cleaned dictionary
            self.encoder_t2.load_state_dict(pretrained_dict, strict=False)
            self.encoder_kinetics.load_state_dict(pretrained_dict, strict=False)
            print("✅ Pre-trained weights loaded into both streams!")

        # The final feature vector size will be 48 * 2^4 = 768.
        hidden_dim = 768
        
        # --- 2. HEADER (Classification MLP) ---
        # T2 provides a vector of 768. 
        # The 4 kinetic phases provide 4 * 768 = 3072.
        # Total input size: 768 + 3072 = 3840.
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + (hidden_dim * 4), 128),
            nn.ReLU(),
            nn.Dropout(0.3), 
            nn.Linear(128, num_classes)
        )
        
    def extract_features(self, x, encoder):
        all_features = encoder.swinViT(x) 
        
        enc_out = all_features[-1]
        
        pooled = enc_out.mean(dim=[2, 3, 4]) 
        
        return pooled

    def forward(self, t2, kinetics):
        f_t2 = self.extract_features(t2, self.encoder_t2) 
        
        B, T, C, H, W, D = kinetics.shape
        
        kin_features = []
        for t in range(T):
            phase_t = kinetics[:, t, ...]
            f_phase = self.extract_features(phase_t, self.encoder_kinetics)
            kin_features.append(f_phase)

        kin_stacked = torch.stack(kin_features, dim=1)
        f_kin_flat = kin_stacked.flatten(start_dim=1) 
        
        merged = torch.cat([f_t2, f_kin_flat], dim=1) 
        out = self.classifier(merged)
        
        return out
    
class PrepareTemporalKineticsd(MapTransform):
    """
    Transform to create a temporal tensor T=4 by computing (Post_i - Pre).
    Handles variable sizes by truncating the excess or duplicating the 
    last available phase (Repeat Padding/LOCF).
    Also cleans up raw keys to prevent DataLoader crashes.
    """
    def __init__(self, keys, max_phases=4, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.max_phases = max_phases

    def __call__(self, data):
        d = dict(data)
        
        pre_img = d["Pre"]
        post_imgs = d["Post"]
        
        # Correct dimension if EnsureChannelFirstd added an artificial channel
        if post_imgs.ndim == 5 and post_imgs.shape[0] == 1:
            post_imgs = post_imgs[0] # Revert to shape (N, 128, 128, 64)
            
        temporal_sequence = []
        
        # 2. Compute dynamic subtractions (iterate over N)
        for i in range(post_imgs.shape[0]):
            # Explicitly keep the channel dimension for compatibility with pre_img
            post_phase = post_imgs[i:i+1] # Shape: (1, 128, 128, 64)
            sub = post_phase - pre_img
            temporal_sequence.append(sub)
            
        # 3. TRUNCATION
        if len(temporal_sequence) > self.max_phases:
            temporal_sequence = temporal_sequence[:self.max_phases]
            
        # 4. REPEAT PADDING
        while len(temporal_sequence) < self.max_phases:
            temporal_sequence.append(temporal_sequence[-1])
            
        # 5. Create Kinetics tensor: Guaranteed final shape (4, 1, 128, 128, 64)
        d["Kinetics"] = torch.stack(temporal_sequence, dim=0) 
        
        # Remove variable-sized keys so that collate_fn doesn't crash
        del d["Pre"]
        del d["Post"]
        
        return d
