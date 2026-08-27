#!/usr/bin/env python
"""
Report total vs trainable parameters for every VoxelFox variant, for parameter-efficiency and
params-vs-performance comparisons. Builds each architecture's STRUCTURE only — no SSL weights, no GPU,
no checkpoint needed — so it runs in seconds on CPU.

Run from repo root (either works):
    python -m models.lora_ensemble.report_params
    python models/lora_ensemble/report_params.py

Writes analysis/uncertainty/lora_param_counts.csv and prints a table.
"""
from __future__ import annotations
import os, os.path as osp, sys, csv, types
sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))  # repo root

from models.swin_voco_utils import build_effidec3d, build_monai_swin_unetr
from models.lora_ensemble import wrap_window_attention, freeze_for_lora, count_params, EnsembleSeg
from models.lora_ensemble.decoder_ensemble import build_decoder_ensemble, build_lora_decoder_ensemble, count_params_dedup


def eff_args():
    return types.SimpleNamespace(
        roi_x=96, roi_y=96, roi_z=64, in_channels=1, out_channels=2, feature_size=48, norm_name="instance",
        swin_depths=(2, 2, 2, 2), swin_num_heads=(3, 6, 12, 24), swin_use_v2=1,
        n_decoder_channels=48, resolution_factor=2, head_upsample="trilinear",
        dropout_rate=0.0, dropout_path_rate=0.0, use_checkpoint=False, spatial_dims=3)


def swinv2_args():  # VoxelFox = MONAI SwinUNETR-V2 (full decoder), no effidec-only args
    return types.SimpleNamespace(
        roi_x=96, roi_y=96, roi_z=64, in_channels=1, out_channels=2, feature_size=48, norm_name="instance",
        swin_depths=(2, 2, 2, 2), swin_num_heads=(3, 6, 12, 24), swin_use_v2=1,
        dropout_rate=0.0, dropout_path_rate=0.0, use_checkpoint=False, spatial_dims=3)


def main():
    rows = []

    def rec(name, m):
        cp = count_params_dedup(m)   # dedups shared params (matters for the decoder-ensemble)
        rows.append([name, cp["total"], cp["trainable"], round(cp["pct"], 3)])

    # 1) VoxelFox = full SwinUNETR (full fine-tune)
    rec("VoxelFox (SwinUNETR, full-FT)", build_monai_swin_unetr(swinv2_args()))
    # 2) VoxelFox + EffiDec decoder (full fine-tune)
    rec("VoxelFox-EffiDec (full-FT)", build_effidec3d(eff_args()))
    # 3) EffiDec + plain LoRA (frozen encoder), rank 4 and 16
    for r in (4, 16):
        m = build_effidec3d(eff_args())
        wrap_window_attention(m.swinViT, rank=r, n_members=1, single=True)
        freeze_for_lora(m, encoder_attr="swinViT")
        rec(f"EffiDec + LoRA (single, r{r})", m)
    # 4) EffiDec + implicit LoRA-Ensemble (shared decoder), N=4 r4
    m = build_effidec3d(eff_args())
    wrap_window_attention(m.swinViT, rank=4, n_members=4, single=False)
    freeze_for_lora(m, encoder_attr="swinViT")
    rec("EffiDec + LoRA-Ensemble (N=4, shared dec)", EnsembleSeg(m, 4))
    # 5) EffiDec + LoRA decoder-ensemble upper-bound (shared encoder/adapters + N decoders), N=4 r4
    base = build_effidec3d(eff_args())
    wrap_window_attention(base.swinViT, rank=4, n_members=1, single=True)
    freeze_for_lora(base, encoder_attr="swinViT")
    de = build_decoder_ensemble(base, lambda: build_effidec3d(eff_args()), 4, encoder_attr="swinViT")
    rec("EffiDec + LoRA + Decoder-Ensemble upper-bound (N=4)", de)
    # 6) Paper-closest segmentation port: per-member LoRA adapters + per-member decoders, N=4 r4
    base = build_effidec3d(eff_args())
    lde = build_lora_decoder_ensemble(base, lambda: build_effidec3d(eff_args()), 4, rank=4, encoder_attr="swinViT")
    rec("EffiDec + LoRA-Ensemble + Decoder-Ensemble (N=4)", lde)

    w = max(len(r[0]) for r in rows)
    print(f"\n{'config':{w}s} {'total':>13s} {'trainable':>13s} {'train%':>8s}")
    for n, t, tr, p in rows:
        print(f"{n:{w}s} {t:13,d} {tr:13,d} {p:7.2f}%")
    out = "analysis/uncertainty/lora_param_counts.csv"
    os.makedirs(osp.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        cw = csv.writer(f); cw.writerow(["config", "total", "trainable", "train_pct"]); cw.writerows(rows)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
