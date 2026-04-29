"""
Utility classes and transformations used by the MultiStream-3DResNet training and inference scripts.
Contains the `MultiStreamResNet` architecture for multi-modal processing (T2 + Temporal Kinetics),
and custom MONAI transforms for handling variable-length dynamic contrast sequences and modality dropout.
"""

import torch
import torch.nn as nn
import random
from monai.transforms import MapTransform
from monai.networks.nets import ResNet

class PrepareTemporalKineticsd(MapTransform):
    """
    Transform used to create a dynamic temporal tensor T by computing (Post_i - Pre).
    Maintains the true number of phases for the patient, allowing the network to process the exact temporal sequence.
    """
    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        
        pre_img = d["Pre"] 
        post_imgs = d["Post"] 
        
        if post_imgs.ndim == 5 and post_imgs.shape[0] == 1:
            post_imgs = post_imgs[0] 
            
        temporal_sequence = []
        
        # Compute dynamic subtractions
        for i in range(post_imgs.shape[0]):
            post_phase = post_imgs[i:i+1] 
            sub = post_phase - pre_img
            temporal_sequence.append(sub)
            
        # Final shape: (T, 1, 128, 128, 64) where T is variable (depends on the number of post images)
        d["Kinetics"] = torch.stack(temporal_sequence, dim=0) 
        
        del d["Pre"]
        del d["Post"]
        
        return d
    
class ModalityDropoutd(MapTransform):
    """
    Randomly zeroes out the T2 modality or a specific phase 
    of the kinetics with probability 'prob', forcing the model to generalize.
    """
    def __init__(self, keys, prob=0.2, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.prob = prob

    def __call__(self, data):
        d = dict(data)
        
        if "T2" in self.keys and random.random() < self.prob:
            d["T2"] = torch.zeros_like(d["T2"])
            
        if "Kinetics" in self.keys and random.random() < self.prob:
            num_phases = d["Kinetics"].shape[0]
            t_drop = random.randint(0, num_phases - 1)
            d["Kinetics"][t_drop] = 0.0
            
        return d


class MultiStreamResNet(nn.Module):
    """
    Corrected Architecture: ResNet Extractors (InstanceNorm) + Temporal Mean Pooling
    """
    def __init__(self, num_classes=3, hidden_dim=256):
        super().__init__()
        
        # --- 1. SPATIAL EXTRACTORS (Custom CNNs) ---
        # Reconstruct an equivalent of ResNet18, but with InstanceNorm
        self.encoder_t2 = ResNet(
            block='basic',
            layers=[2, 2, 2, 2],
            block_inplanes=[64, 128, 256, 512],
            spatial_dims=3,
            n_input_channels=1,
            num_classes=hidden_dim,
            norm=('instance', {'affine': True})
        )
        
        self.encoder_kinetics = ResNet(
            block='basic',
            layers=[2, 2, 2, 2],
            block_inplanes=[64, 128, 256, 512],
            spatial_dims=3,
            n_input_channels=1,
            num_classes=hidden_dim,
            norm=('instance', {'affine': True})
        )

        # --- 2. FINAL CLASSIFIER ---
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, t2, kinetics):
        # --- T2 Stream ---
        f_t2 = self.encoder_t2(t2) 
        
        # --- Kinetic Stream ---
        B, T, C, H, W, D = kinetics.shape
        kin_flat = kinetics.view(B * T, C, H, W, D) 
        f_kin = self.encoder_kinetics(kin_flat)
        
        # Recover the temporal sequence: (Batch, Time, hidden_dim)
        f_kin_seq = f_kin.view(B, T, -1) 
        
        # --- TEMPORAL FUSION (Mean Pooling) ---
        # Instead of an LSTM, average the vectors over the Time axis (dim=1)
        f_kin_summary = f_kin_seq.mean(dim=1) 

        # --- Final Fusion ---
        merged = torch.cat([f_t2, f_kin_summary], dim=1) 
        out = self.classifier(merged)
        
        return out
