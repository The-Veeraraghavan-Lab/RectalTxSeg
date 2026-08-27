"""Dry-load sanity check for VoxelFox SSL weights before fine-tuning.

Builds the same swinv2 SwinUNETR (feature_size 48) used for VoCo-ACT and
loads pretrained_models/voxelfox_swinvit.pt through the existing encoder loader.

Run (from repo root):  python models/check_voxelfox_load.py
Expect: unexpected ~= 0. If unexpected is high, VoxelFox uses a key prefix not
covered in extract_swin_encoder_state() -> add it there (one line).
"""
import os
import sys
import types

# allow running from a subfolder: put repo root on the path so `models.` resolves
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.swin_voco_utils import build_monai_swin_unetr, load_swin_encoder_pretrained

CKPT = "pretrained_models/voxelfox_swinvit.pt"

args = types.SimpleNamespace(
    roi_x=96, roi_y=96, roi_z=64,
    in_channels=1, out_channels=2,
    swin_depths=(2, 2, 2, 2), swin_num_heads=(3, 6, 12, 24),
    feature_size=48, norm_name="instance",
    dropout_rate=0.0, dropout_path_rate=0.0,
    use_checkpoint=False, spatial_dims=3, swin_use_v2=True,
)

model = build_monai_swin_unetr(args)
msg = load_swin_encoder_pretrained(model, CKPT, strict=False)
n_enc = sum(1 for _ in model.swinViT.state_dict())
print(f"\nSummary: encoder keys in model = {n_enc} | "
      f"missing = {len(msg.missing_keys)} | unexpected = {len(msg.unexpected_keys)}")
print("OK to train" if len(msg.unexpected_keys) == 0 else
      "FIX: add VoxelFox's key prefix to extract_swin_encoder_state()")
