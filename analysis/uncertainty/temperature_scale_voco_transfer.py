#!/usr/bin/env python
"""Validation-fitted calibration summary for the VoCo transfer experiment."""
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import generate_binary_structure, label as label_connected_components


MODELS = [
    ("VoCo", "results/rectal_voco_swinunetr_96x96x64_pretrained/voco_swinunetr"),
    ("VoCo-EffiDec", "results/rectal_effidec3d_voco_96x96x64_pretrained/effidec3d_voco"),
    ("VoCo-LoRA", "results/rectal_effidec3d_voco_loraS_r4_bs2/effidec3d"),
    ("VoCo-LDE4", "results/rectal_effidec3d_voco_loraLDE4_bs2/effidec3d"),
]
EPS = 1e-6


def case_id_from_prob(path: Path) -> str:
    stem = re.sub(r"\.nii(\.gz)?$", "", path.name)
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


def postprocess_prob(prob: np.ndarray, gt: np.ndarray, min_component_voxels: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gt = remove_small_components(gt, min_component_voxels)
    pred = remove_small_components(prob >= 0.5, min_component_voxels)
    eval_prob = prob.copy()
    if min_component_voxels > 0:
        eval_prob[~pred] = np.where(prob[~pred] >= 0.5, 0.0, prob[~pred])
    return eval_prob, gt, pred


def temp_scale(prob: np.ndarray, temperature: float) -> np.ndarray:
    prob = np.clip(prob.astype(np.float64), EPS, 1.0 - EPS)
    logits = np.log(prob / (1.0 - prob))
    return 1.0 / (1.0 + np.exp(-logits / temperature))


def collect_validation(root: Path, min_component_voxels: int) -> tuple[np.ndarray, np.ndarray]:
    probs, labels = [], []
    for prob_path in sorted((root / "validation").glob("*_prob.nii.gz")):
        case_id = case_id_from_prob(prob_path)
        lbl_path = label_path(case_id, "validation")
        if not lbl_path.exists():
            continue
        prob = nib.load(str(prob_path)).get_fdata().astype(np.float32)
        gt = nib.load(str(lbl_path)).get_fdata() > 0.5
        if prob.shape != gt.shape:
            raise ValueError(f"shape mismatch: {prob_path} vs {lbl_path}")
        prob, gt, _ = postprocess_prob(prob, gt, min_component_voxels)
        roi = (prob >= 0.05) | gt
        probs.append(prob[roi].astype(np.float32))
        labels.append(gt[roi].astype(np.float32))
    if not probs:
        raise FileNotFoundError(f"No validation probability maps under {root}")
    return np.concatenate(probs), np.concatenate(labels)


def nll(prob: np.ndarray, gt: np.ndarray, temperature: float) -> float:
    p = np.clip(temp_scale(prob, temperature), EPS, 1.0 - EPS)
    return float((-(gt * np.log(p) + (1.0 - gt) * np.log(1.0 - p))).mean())


def fit_temperature(prob: np.ndarray, gt: np.ndarray) -> float:
    grid = np.exp(np.linspace(math.log(0.10), math.log(20.0), 360))
    best = float(grid[int(np.argmin([nll(prob, gt, t) for t in grid]))])
    for width in (0.35, 0.08):
        lo = max(0.03, best * (1.0 - width))
        hi = best * (1.0 + width)
        grid = np.linspace(lo, hi, 180)
        best = float(grid[int(np.argmin([nll(prob, gt, t) for t in grid]))])
    return best


def empty_bins(n_bins: int) -> dict[str, np.ndarray]:
    return {"cnt": np.zeros(n_bins), "conf_sum": np.zeros(n_bins), "acc_sum": np.zeros(n_bins)}


def update_bins(acc: dict[str, np.ndarray], prob: np.ndarray, gt: np.ndarray, n_bins: int) -> None:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(prob, edges) - 1, 0, n_bins - 1)
    for b in range(n_bins):
        sel = idx == b
        if sel.any():
            acc["cnt"][b] += int(sel.sum())
            acc["conf_sum"][b] += float(prob[sel].sum())
            acc["acc_sum"][b] += float(gt[sel].sum())


def summarize_bins(acc: dict[str, np.ndarray]) -> tuple[float, float, float]:
    cnt = acc["cnt"]
    conf = np.divide(acc["conf_sum"], cnt, out=np.zeros_like(cnt), where=cnt > 0)
    actual = np.divide(acc["acc_sum"], cnt, out=np.zeros_like(cnt), where=cnt > 0)
    frac = np.divide(cnt, cnt.sum(), out=np.zeros_like(cnt), where=cnt.sum() > 0)
    ece = float(np.sum(frac * np.abs(conf - actual)))
    avg_conf = float(acc["conf_sum"].sum() / cnt.sum()) if cnt.sum() else float("nan")
    avg_acc = float(acc["acc_sum"].sum() / cnt.sum()) if cnt.sum() else float("nan")
    return ece, avg_conf, avg_acc


def evaluate(root: Path, split: str, temperature: float, n_bins: int, min_component_voxels: int) -> dict:
    raw_bins = empty_bins(n_bins)
    cal_bins = empty_bins(n_bins)
    raw_brier_sum = cal_brier_sum = raw_nll_sum = cal_nll_sum = 0.0
    voxels = cases = missing_labels = 0
    prob_paths = sorted((root / split).glob("*_prob.nii.gz"))
    for prob_path in prob_paths:
        case_id = case_id_from_prob(prob_path)
        lbl_path = label_path(case_id, split)
        if not lbl_path.exists():
            missing_labels += 1
            continue
        prob = nib.load(str(prob_path)).get_fdata().astype(np.float32)
        gt = nib.load(str(lbl_path)).get_fdata() > 0.5
        if prob.shape != gt.shape:
            raise ValueError(f"shape mismatch: {prob_path} vs {lbl_path}")
        prob, gt, _ = postprocess_prob(prob, gt, min_component_voxels)
        roi = (prob >= 0.05) | gt
        p_raw = prob[roi].astype(np.float64)
        g = gt[roi].astype(np.float64)
        if p_raw.size == 0:
            continue
        p_cal = temp_scale(p_raw, temperature)
        p_raw_clip = np.clip(p_raw, EPS, 1 - EPS)
        p_cal_clip = np.clip(p_cal, EPS, 1 - EPS)
        raw_brier_sum += float(((p_raw - g) ** 2).sum())
        cal_brier_sum += float(((p_cal - g) ** 2).sum())
        raw_nll_sum += float((-(g * np.log(p_raw_clip) + (1 - g) * np.log(1 - p_raw_clip))).sum())
        cal_nll_sum += float((-(g * np.log(p_cal_clip) + (1 - g) * np.log(1 - p_cal_clip))).sum())
        update_bins(raw_bins, p_raw, g, n_bins)
        update_bins(cal_bins, p_cal, g, n_bins)
        voxels += int(p_raw.size)
        cases += 1
    raw_ece, raw_conf, raw_acc = summarize_bins(raw_bins)
    cal_ece, cal_conf, _ = summarize_bins(cal_bins)
    return {
        "split": split,
        "n_prob_maps": len(prob_paths),
        "n_cases_used": cases,
        "missing_labels": missing_labels,
        "roi_voxels": voxels,
        "temperature": temperature,
        "ece_raw": raw_ece,
        "ece_calibrated": cal_ece,
        "brier_raw": raw_brier_sum / voxels if voxels else float("nan"),
        "brier_calibrated": cal_brier_sum / voxels if voxels else float("nan"),
        "nll_raw": raw_nll_sum / voxels if voxels else float("nan"),
        "nll_calibrated": cal_nll_sum / voxels if voxels else float("nan"),
        "avg_confidence_raw": raw_conf,
        "avg_confidence_calibrated": cal_conf,
        "avg_accuracy": raw_acc,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="analysis/uncertainty")
    ap.add_argument("--n_bins", type=int, default=15)
    ap.add_argument("--min_component_voxels", type=int, default=10)
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for model, root_str in MODELS:
        root = Path(root_str)
        print(f"[fit] {model}")
        val_prob, val_gt = collect_validation(root, args.min_component_voxels)
        temperature = fit_temperature(val_prob, val_gt)
        for split in ("validation", "testing"):
            row = evaluate(root, split, temperature, args.n_bins, args.min_component_voxels)
            rows.append({"model": model, **row})
            print(f"  {split:10s} ECE {row['ece_raw']:.4f}->{row['ece_calibrated']:.4f}")

    fields = [
        "model", "split", "n_prob_maps", "n_cases_used", "missing_labels", "roi_voxels",
        "temperature", "ece_raw", "ece_calibrated", "brier_raw", "brier_calibrated",
        "nll_raw", "nll_calibrated", "avg_confidence_raw", "avg_confidence_calibrated",
        "avg_accuracy",
    ]
    out_csv = outdir / "temperature_scaling_voco_transfer.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: row.get(k, "") for k in fields} for row in rows)
    print(f"[write] {out_csv}")


if __name__ == "__main__":
    main()
