"""Dry-load sanity check for EffiDec3D before fine-tuning.

Builds SwinUNETR_EffiDec3D (SwinV2 encoder + EffiDec3D decoder, feature_size 48,
n_decoder_channels 48, resolution_factor 2), loads the VoxelFox SSL encoder weights
through the existing encoder loader, runs a forward pass, and confirms the output is
restored to full input resolution.

Run (from repo root):  python models/check_effidec3d_load.py
Expect:
  * encoder unexpected == 0  (SSL weights map cleanly into model.swinViT)
  * decoder is reported as "randomly initialised" (SSL never trains a decoder)
  * forward output shape == input spatial shape (head_upsample restores full res)
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.swin_voco_utils import build_effidec3d, load_swin_encoder_pretrained

CKPT = "pretrained_models/voxelfox_swinvit.pt"

args = types.SimpleNamespace(
    roi_x=96, roi_y=96, roi_z=64,
    in_channels=1, out_channels=2,
    swin_depths=(2, 2, 2, 2), swin_num_heads=(3, 6, 12, 24),
    feature_size=48, norm_name="instance",
    n_decoder_channels=48, resolution_factor=2, head_upsample="trilinear",
    dropout_rate=0.0, dropout_path_rate=0.0,
    use_checkpoint=False, spatial_dims=3, swin_use_v2=True,
)

model = build_effidec3d(args)
n_total = sum(p.numel() for p in model.parameters())
n_enc = sum(p.numel() for p in model.swinViT.parameters())
print(f"params: total={n_total:,} | encoder(swinViT)={n_enc:,} | decoder+head={n_total - n_enc:,}")

# --- load SSL encoder weights ---
msg = load_swin_encoder_pretrained(model, CKPT, strict=False)
print(f"encoder load: missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
print("encoder OK" if len(msg.unexpected_keys) == 0 else
      "FIX: add EffiDec3D/VoxelFox key prefix to extract_swin_encoder_state()")

# --- forward pass: confirm full-resolution output ---
model.eval()
x = torch.randn(1, 1, 96, 96, 64)
with torch.no_grad():
    y = model(x)
print(f"forward: in={tuple(x.shape)} -> out={tuple(y.shape)}")
assert y.shape[2:] == x.shape[2:], "output not restored to full input resolution!"
print("forward OK — output at full input resolution")
