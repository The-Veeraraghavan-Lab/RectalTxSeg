#!/usr/bin/env python
"""Validation-fitted temperature scaling for all paper-facing probability maps.

Fits one scalar temperature per contour generator on validation foreground-ROI
voxels, then applies the fixed temperature to validation and held-out test
probability maps. The transform is monotonic in foreground probability, so the
0.5-threshold contour is unchanged; only probability calibration changes.

Outputs:
  analysis/uncertainty/temperature_scaling_all_models.csv
  analysis/uncertainty/temperature_scaling_bins_validation.csv
  analysis/uncertainty/temperature_scaling_bins_testing.csv
"""
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
    ("VoxelFox", "results/rectal_voxelfox_swinunetr_96x96x64_pretrained/voxelfox_swinunetr"),
    ("VoxelFox-EffiDec", "results/rectal_effidec3d_96x96x64_pretrained/effidec3d"),
    ("LoRA", "results/rectal_effidec3d_loraS_r4_bs2/effidec3d"),
    ("LoRA-Ens", "results/rectal_effidec3d_loraLDE4/effidec3d"),
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


def postprocess_prob_for_eval(prob: np.ndarray, gt: np.ndarray, min_component_voxels: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def collect_validation_arrays(root: Path, n_bins: int, min_component_voxels: int) -> tuple[np.ndarray, np.ndarray]:
    probs = []
    labels = []
    for prob_path in sorted((root / "validation").glob("*_prob.nii.gz")):
        case_id = case_id_from_prob(prob_path)
        lbl_path = label_path(case_id, "validation")
        if not lbl_path.exists():
            continue
        prob = nib.load(str(prob_path)).get_fdata().astype(np.float32)
        gt = nib.load(str(lbl_path)).get_fdata() > 0.5
        if prob.shape != gt.shape:
            raise ValueError(f"shape mismatch: {prob_path} vs {lbl_path}")
        prob, gt, _ = postprocess_prob_for_eval(prob, gt, min_component_voxels)
        roi = (prob >= 0.05) | gt
        probs.append(prob[roi].astype(np.float32))
        labels.append(gt[roi].astype(np.float32))
    if not probs:
        raise FileNotFoundError(f"No validation probability maps with labels found under {root}")
    return np.concatenate(probs), np.concatenate(labels)


def nll_for_temperature(prob: np.ndarray, gt: np.ndarray, temperature: float) -> float:
    p = temp_scale(prob, temperature)
    p = np.clip(p, EPS, 1.0 - EPS)
    return float((-(gt * np.log(p) + (1.0 - gt) * np.log(1.0 - p))).mean())


def binary_entropy(prob: np.ndarray) -> np.ndarray:
    prob = np.clip(prob.astype(np.float64), EPS, 1.0 - EPS)
    return -(prob * np.log(prob) + (1.0 - prob) * np.log(1.0 - prob))


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    denom = pred.sum() + gt.sum()
    return float((2.0 * np.logical_and(pred, gt).sum()) / denom) if denom else 1.0


def rankdata_average(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    rx = rankdata_average(np.asarray(x, float))
    ry = rankdata_average(np.asarray(y, float))
    rx -= rx.mean()
    ry -= ry.mean()
    den = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    return float((rx * ry).sum() / den) if den else float("nan")


def auroc(score: np.ndarray, positive: np.ndarray) -> float:
    score = np.asarray(score, float)
    positive = np.asarray(positive, bool)
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata_average(score)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def fit_temperature(prob: np.ndarray, gt: np.ndarray) -> float:
    # Coarse log-grid, then two local refinements around the best value.
    grid = np.exp(np.linspace(math.log(0.10), math.log(20.0), 360))
    best_t = float(grid[int(np.argmin([nll_for_temperature(prob, gt, t) for t in grid]))])
    for width in (0.35, 0.08):
        lo = max(0.03, best_t * (1.0 - width))
        hi = best_t * (1.0 + width)
        grid = np.linspace(lo, hi, 180)
        best_t = float(grid[int(np.argmin([nll_for_temperature(prob, gt, t) for t in grid]))])
    return best_t


def _empty_bin_accumulators(n_bins: int) -> dict[str, np.ndarray]:
    return {
        "cnt": np.zeros(n_bins, float),
        "conf_sum": np.zeros(n_bins, float),
        "acc_sum": np.zeros(n_bins, float),
    }


def _update_bins(acc: dict[str, np.ndarray], prob: np.ndarray, gt: np.ndarray, n_bins: int) -> None:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(prob, edges) - 1, 0, n_bins - 1)
    for b in range(n_bins):
        sel = idx == b
        if not sel.any():
            continue
        acc["cnt"][b] += int(sel.sum())
        acc["conf_sum"][b] += float(prob[sel].sum())
        acc["acc_sum"][b] += float(gt[sel].sum())


def _summarize_bins(acc: dict[str, np.ndarray]) -> tuple[float, float, float, list[dict]]:
    cnt = acc["cnt"]
    conf = np.divide(acc["conf_sum"], cnt, out=np.zeros_like(cnt), where=cnt > 0)
    acc_rate = np.divide(acc["acc_sum"], cnt, out=np.zeros_like(cnt), where=cnt > 0)
    frac = np.divide(cnt, cnt.sum(), out=np.zeros_like(cnt), where=cnt.sum() > 0)
    ece = float(np.sum(frac * np.abs(conf - acc_rate)))
    avg_conf = float(acc["conf_sum"].sum() / cnt.sum()) if cnt.sum() else float("nan")
    avg_acc = float(acc["acc_sum"].sum() / cnt.sum()) if cnt.sum() else float("nan")
    rows = []
    edges = np.linspace(0.0, 1.0, len(cnt) + 1)
    for b in range(len(cnt)):
        rows.append({
            "bin": b,
            "bin_lo": edges[b],
            "bin_hi": edges[b + 1],
            "conf": conf[b],
            "acc": acc_rate[b],
            "count": cnt[b],
            "fraction": frac[b],
        })
    return ece, avg_conf, avg_acc, rows


def evaluate_split(root: Path, split: str, temperature: float, n_bins: int, min_component_voxels: int) -> tuple[dict, list[dict]]:
    raw_bins = _empty_bin_accumulators(n_bins)
    cal_bins = _empty_bin_accumulators(n_bins)
    raw_brier_sum = raw_nll_sum = 0.0
    cal_brier_sum = cal_nll_sum = 0.0
    voxels = cases = missing_labels = 0
    case_dice = []
    case_entropy_unc = []
    for prob_path in sorted((root / split).glob("*_prob.nii.gz")):
        case_id = case_id_from_prob(prob_path)
        lbl_path = label_path(case_id, split)
        if not lbl_path.exists():
            missing_labels += 1
            continue
        prob = nib.load(str(prob_path)).get_fdata().astype(np.float32)
        gt = nib.load(str(lbl_path)).get_fdata() > 0.5
        if prob.shape != gt.shape:
            raise ValueError(f"shape mismatch: {prob_path} vs {lbl_path}")
        prob, gt, pred = postprocess_prob_for_eval(prob, gt, min_component_voxels)
        roi = (prob >= 0.05) | gt
        p_raw = prob[roi].astype(np.float64)
        g = gt[roi].astype(np.float64)
        if p_raw.size == 0:
            continue
        ent = binary_entropy(prob)
        pred_region = pred
        case_dice.append(dice(pred, gt))
        case_entropy_unc.append(float(ent[pred_region].mean()) if pred_region.any() else float(ent.mean()))
        p_cal = temp_scale(p_raw, temperature)
        p_raw_clip = np.clip(p_raw, EPS, 1.0 - EPS)
        p_cal_clip = np.clip(p_cal, EPS, 1.0 - EPS)
        raw_brier_sum += float(((p_raw - g) ** 2).sum())
        cal_brier_sum += float(((p_cal - g) ** 2).sum())
        raw_nll_sum += float((-(g * np.log(p_raw_clip) + (1.0 - g) * np.log(1.0 - p_raw_clip))).sum())
        cal_nll_sum += float((-(g * np.log(p_cal_clip) + (1.0 - g) * np.log(1.0 - p_cal_clip))).sum())
        _update_bins(raw_bins, p_raw, g, n_bins)
        _update_bins(cal_bins, p_cal, g, n_bins)
        voxels += int(p_raw.size)
        cases += 1

    raw_ece, raw_conf, raw_acc, raw_bin_rows = _summarize_bins(raw_bins)
    cal_ece, cal_conf, cal_acc, cal_bin_rows = _summarize_bins(cal_bins)
    case_dice_arr = np.asarray(case_dice, float)
    case_unc_arr = np.asarray(case_entropy_unc, float)
    metrics = {
        "split": split,
        "n_prob_maps": len(list((root / split).glob("*_prob.nii.gz"))),
        "n_cases_used": cases,
        "missing_labels": missing_labels,
        "roi_voxels": voxels,
        "temperature": temperature,
        "ece_raw": raw_ece,
        "ece_calibrated": cal_ece,
        "ece_delta": cal_ece - raw_ece,
        "brier_raw": raw_brier_sum / voxels if voxels else float("nan"),
        "brier_calibrated": cal_brier_sum / voxels if voxels else float("nan"),
        "nll_raw": raw_nll_sum / voxels if voxels else float("nan"),
        "nll_calibrated": cal_nll_sum / voxels if voxels else float("nan"),
        "avg_confidence_raw": raw_conf,
        "avg_confidence_calibrated": cal_conf,
        "avg_accuracy": raw_acc,
        "failure_auroc_entropy": auroc(case_unc_arr, case_dice_arr <= 0.20),
        "spearman_entropy_dice": spearman(case_unc_arr, case_dice_arr),
        "n_failures_dice_le_020": int((case_dice_arr <= 0.20).sum()),
    }
    bin_rows = []
    for phase, rows in (("raw", raw_bin_rows), ("calibrated", cal_bin_rows)):
        for row in rows:
            bin_rows.append({"split": split, "phase": phase, **row})
    return metrics, bin_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="analysis/uncertainty")
    ap.add_argument("--n_bins", type=int, default=15)
    ap.add_argument("--min_component_voxels", type=int, default=10)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    bin_rows_by_split: dict[str, list[dict]] = {"validation": [], "testing": []}

    for model, root_str in MODELS:
        root = Path(root_str)
        print(f"[fit] {model}")
        val_prob, val_gt = collect_validation_arrays(root, args.n_bins, args.min_component_voxels)
        temperature = fit_temperature(val_prob, val_gt)
        print(f"      T={temperature:.4f} from {val_prob.size:,} validation ROI voxels")
        for split in ("validation", "testing"):
            metrics, bin_rows = evaluate_split(root, split, temperature, args.n_bins, args.min_component_voxels)
            summary_rows.append({"model": model, **metrics})
            for row in bin_rows:
                bin_rows_by_split[split].append({"model": model, **row})
            print(
                f"      {split:10s} ECE {metrics['ece_raw']:.4f}->{metrics['ece_calibrated']:.4f} "
                f"NLL {metrics['nll_raw']:.4f}->{metrics['nll_calibrated']:.4f}"
            )

    summary_fields = [
        "model", "split", "n_prob_maps", "n_cases_used", "missing_labels", "roi_voxels",
        "temperature", "ece_raw", "ece_calibrated", "ece_delta",
        "brier_raw", "brier_calibrated", "nll_raw", "nll_calibrated",
        "avg_confidence_raw", "avg_confidence_calibrated", "avg_accuracy",
        "failure_auroc_entropy", "spearman_entropy_dice", "n_failures_dice_le_020",
    ]
    out_csv = outdir / "temperature_scaling_all_models.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows({k: row.get(k, "") for k in summary_fields} for row in summary_rows)
    print(f"[write] {out_csv}")

    bin_fields = ["model", "split", "phase", "bin", "bin_lo", "bin_hi", "conf", "acc", "count", "fraction"]
    for split, rows in bin_rows_by_split.items():
        path = outdir / f"temperature_scaling_bins_{split}.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=bin_fields)
            writer.writeheader()
            writer.writerows({k: row.get(k, "") for k in bin_fields} for row in rows)
        print(f"[write] {path}")


if __name__ == "__main__":
    main()
