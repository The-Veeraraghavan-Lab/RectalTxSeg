#!/usr/bin/env python
"""Compute foreground-ROI ECE from saved probability maps and LoRA reliability CSVs.

This is intentionally paper-facing and boring:
  * Probability-map rows are computed from *_prob.nii.gz files and manual labels.
  * LoRA extractor rows can be read from lora_reliability_<tag>.csv if maps are
    not yet exported or if we want the exact extractor-side ECE.

Outputs:
  analysis/uncertainty/ece_all_models_<split>.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import os.path as osp
import re
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import generate_binary_structure, label as label_connected_components


PROB_MODELS = [
    ("VoxelFox full", "results/rectal_voxelfox_swinunetr_96x96x64_pretrained/voxelfox_swinunetr"),
    ("VoxelFox-EffiDec", "results/rectal_effidec3d_96x96x64_pretrained/effidec3d"),
    ("LoRA r4", "results/rectal_effidec3d_loraS_r4_bs2/effidec3d"),
    ("LoRA-LDE4", "results/rectal_effidec3d_loraLDE4/effidec3d"),
]

RELIABILITY_TAGS = {
    "testing": [
        ("LoRA r4", "single_r4_bs2_pp10_test"),
        ("LoRA-LDE4", "lde_bs2_pp10_test"),
    ],
    "validation": [
        ("LoRA r4", "single_r4_bs2_pp10_val"),
        ("LoRA-LDE4", "lde_bs2_pp10_val"),
    ],
}


def case_id_from_prob(path: Path) -> str:
    stem = path.name
    stem = re.sub(r"\.nii(\.gz)?$", "", stem)
    stem = re.sub(r"_prob$", "", stem)
    return re.sub(r"_z800$", "", stem)


def label_path(case_id: str, split: str) -> Path:
    label_dir = "labelsTs" if split == "testing" else "labelsTr"
    return Path("data_rectal") / label_dir / f"{case_id}.nii.gz"


def remove_small_components(mask: np.ndarray, min_component_voxels: int) -> np.ndarray:
    mask = np.asarray(mask).astype(bool)
    if min_component_voxels <= 0 or not mask.any():
        return mask
    cc, _ = label_connected_components(mask, structure=generate_binary_structure(3, 3))
    counts = np.bincount(cc.ravel())
    keep = np.where(counts > min_component_voxels)[0]
    keep = keep[keep != 0]
    if keep.size == 0:
        return np.zeros_like(mask, dtype=bool)
    return np.isin(cc, keep)


def postprocess_prob_for_eval(prob: np.ndarray, gt: np.ndarray, min_component_voxels: int) -> tuple[np.ndarray, np.ndarray]:
    gt = remove_small_components(gt, min_component_voxels)
    pred = remove_small_components(prob >= 0.5, min_component_voxels)
    eval_prob = prob.copy()
    if min_component_voxels > 0:
        eval_prob[~pred] = np.where(prob[~pred] >= 0.5, 0.0, prob[~pred])
    return eval_prob, gt


def pooled_ece_from_prob_dir(prob_dir: Path, split: str, n_bins: int, min_component_voxels: int) -> dict | None:
    prob_paths = sorted(prob_dir.glob("*_prob.nii.gz"))
    if not prob_paths:
        return None
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    conf_sum = np.zeros(n_bins)
    acc_sum = np.zeros(n_bins)
    cnt = np.zeros(n_bins)
    case_ece = []
    missing_labels = 0
    for prob_path in prob_paths:
        case_id = case_id_from_prob(prob_path)
        lbl_path = label_path(case_id, split)
        if not lbl_path.exists():
            missing_labels += 1
            continue
        prob = nib.load(str(prob_path)).get_fdata().astype(np.float32)
        gt = nib.load(str(lbl_path)).get_fdata() > 0.5
        if prob.shape != gt.shape:
            raise ValueError(f"shape mismatch for {prob_path}: prob={prob.shape}, label={gt.shape}")
        prob, gt = postprocess_prob_for_eval(prob, gt, min_component_voxels)
        roi = (prob >= 0.05) | gt
        p = prob[roi].astype(float)
        g = gt[roi].astype(float)
        if p.size == 0:
            continue
        idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
        this_ece = 0.0
        for b in range(n_bins):
            sel = idx == b
            if not sel.any():
                continue
            conf = float(p[sel].mean())
            acc = float(g[sel].mean())
            this_ece += float(sel.mean()) * abs(conf - acc)
            cnt[b] += int(sel.sum())
            conf_sum[b] += float(p[sel].sum())
            acc_sum[b] += float(g[sel].sum())
        case_ece.append(this_ece)
    if cnt.sum() == 0:
        return None
    conf = np.divide(conf_sum, cnt, out=np.zeros_like(cnt), where=cnt > 0)
    acc = np.divide(acc_sum, cnt, out=np.zeros_like(cnt), where=cnt > 0)
    pooled = float(np.sum((cnt / cnt.sum()) * np.abs(conf - acc)))
    return {
        "n_prob_maps": len(prob_paths),
        "n_cases_used": len(case_ece),
        "missing_labels": missing_labels,
        "pooled_ece": pooled,
        "mean_case_ece": float(np.mean(case_ece)) if case_ece else float("nan"),
        "median_case_ece": float(np.median(case_ece)) if case_ece else float("nan"),
    }


def ece_from_reliability_csv(path: Path) -> dict | None:
    if not path.exists():
        return None
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return None
    conf = np.array([float(r["conf"]) for r in rows])
    acc = np.array([float(r["acc"]) for r in rows])
    cnt = np.array([float(r["count"]) for r in rows])
    pooled = float(np.sum((cnt / cnt.sum()) * np.abs(conf - acc)))
    return {
        "n_prob_maps": "",
        "n_cases_used": "",
        "missing_labels": "",
        "pooled_ece": pooled,
        "mean_case_ece": "",
        "median_case_ece": "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="testing", choices=["testing", "validation"])
    ap.add_argument("--n_bins", type=int, default=15)
    ap.add_argument("--min_component_voxels", type=int, default=10)
    ap.add_argument("--outdir", default="analysis/uncertainty")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    out_csv = Path(args.outdir) / f"ece_all_models_{args.split}.csv"
    rows = []

    for label, root in PROB_MODELS:
        metrics = pooled_ece_from_prob_dir(Path(root) / args.split, args.split, args.n_bins, args.min_component_voxels)
        if metrics is None:
            rows.append({"model": label, "split": args.split, "source": "prob_maps_missing"})
            continue
        rows.append({"model": label, "split": args.split, "source": "prob_maps", **metrics})

    for label, tag in RELIABILITY_TAGS.get(args.split, []):
        metrics = ece_from_reliability_csv(Path("analysis/uncertainty") / f"lora_reliability_{tag}.csv")
        if metrics is None:
            continue
        rows.append({"model": label, "split": args.split, "source": f"reliability_csv:{tag}", **metrics})

    fields = [
        "model", "split", "source", "n_prob_maps", "n_cases_used", "missing_labels",
        "pooled_ece", "mean_case_ece", "median_case_ece",
    ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})

    print(f"wrote {out_csv}")
    for row in rows:
        if "pooled_ece" in row:
            print(f"{row['model']:20s} {row['source']:28s} ECE={float(row['pooled_ece']):.4f}")
        else:
            print(f"{row['model']:20s} {row['source']}")


if __name__ == "__main__":
    main()
