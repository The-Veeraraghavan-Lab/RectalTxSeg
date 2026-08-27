#!/usr/bin/env python
"""
Segmentation-uncertainty evaluation for LoRA-Ensemble models.

Given, per case: an ensemble mean-probability map, a per-voxel foreground uncertainty (variance)
map, and the GT mask, compute the metrics that decide whether the section is real:
  - lesion Dice (of the ensemble mean, argmax)
  - a scalar case-level uncertainty (mean foreground variance over predicted region)
  - across the cohort: Spearman(uncertainty, Dice)  -> should be strongly NEGATIVE
  - failure-flagging AUROC: does high uncertainty detect low-Dice ("failed") cases?

No torch required — operates on numpy arrays, so it can run anywhere and be unit-tested.
"""
from __future__ import annotations
import numpy as np


def dice(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    pred = pred.astype(bool); gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    return float((2 * inter + eps) / (pred.sum() + gt.sum() + eps))


def case_uncertainty(mean_prob_fg: np.ndarray, var_fg: np.ndarray, thr: float = 0.5) -> float:
    """Mean foreground variance inside the predicted region (falls back to whole volume if empty)."""
    region = mean_prob_fg >= thr
    if region.sum() == 0:
        return float(var_fg.mean())
    return float(var_fg[region].mean())


def binary_entropy(p: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-voxel predictive entropy of the foreground probability (single-model uncertainty)."""
    p = np.clip(p, eps, 1 - eps)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def ece_reliability(prob_fg: np.ndarray, gt_fg: np.ndarray, mask: np.ndarray | None = None,
                    n_bins: int = 15):
    """
    Foreground calibration: bin voxels by predicted fg-probability, compare mean confidence vs empirical
    accuracy (fraction actually GT-foreground). Returns (ECE, reliability_bins). `mask` restricts to the
    region of interest (e.g. pred>=0.05 | GT) so background doesn't trivially dominate.
    Returns ECE=nan if no voxels in mask.
    """
    p = prob_fg[mask] if mask is not None else prob_fg.ravel()
    g = (gt_fg[mask] if mask is not None else gt_fg.ravel()).astype(float)
    if p.size == 0:
        return float("nan"), []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, N, bins = 0.0, p.size, []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        sel = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        k = int(sel.sum())
        if k == 0:
            continue
        conf, acc = float(p[sel].mean()), float(g[sel].mean())
        ece += (k / N) * abs(conf - acc)
        bins.append({"conf": conf, "acc": acc, "count": k})
    return float(ece), bins


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """One-based average ranks, with ties assigned their mean rank."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sorted_x = x[order]
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and sorted_x[end] == sorted_x[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3:
        return float("nan")
    rx = _average_ranks(x); ry = _average_ranks(y)
    rx = rx - rx.mean(); ry = ry - ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def flagging_auroc(uncertainty: np.ndarray, dice_scores: np.ndarray, dice_fail: float = 0.20) -> float:
    """
    AUROC for using case uncertainty to flag failures (Dice <= dice_fail as the positive class).
    Rank-based (Mann-Whitney) AUROC; returns nan if only one class present.
    """
    unc = np.asarray(uncertainty, float)
    fail = (np.asarray(dice_scores, float) <= dice_fail).astype(int)
    n_pos, n_neg = fail.sum(), (1 - fail).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(unc)
    auc = (ranks[fail == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def summarize(records: list[dict], dice_fail: float = 0.20) -> dict:
    """
    records: list of {'id', 'dice', 'uncertainty', 'size_cc'(optional)} per case.
    Returns a cohort-level summary.
    """
    d = np.array([r["dice"] for r in records], float)
    u = np.array([r["uncertainty"] for r in records], float)
    out = {
        "n": len(records),
        "median_dice": float(np.median(d)),
        "spearman_unc_dice": spearman(u, d),
        "flagging_auroc": flagging_auroc(u, d, dice_fail),
        "n_failures": int((d <= dice_fail).sum()),
    }
    if records and "size_cc" in records[0]:
        s = np.array([r["size_cc"] for r in records], float)
        out["spearman_unc_size"] = spearman(u, s)
    if records and "foreground_roi_ece" in records[0]:
        out["mean_foreground_roi_ece"] = float(np.nanmean([r["foreground_roi_ece"] for r in records]))
    if records and "ece" in records[0]:
        out["mean_ece"] = float(np.nanmean([r["ece"] for r in records]))
    if records and "disagreement" in records[0]:
        out["mean_disagreement"] = float(np.nanmean([r["disagreement"] for r in records]))
    return out


if __name__ == "__main__":  # tiny self-test with synthetic data (exercises the flagging path)
    rng = np.random.default_rng(0)
    recs = []
    for i in range(40):
        # ~15% deliberate failures (small/atypical) so the flagging AUROC path is exercised
        true_d = float(np.clip(rng.normal(0.15, 0.05) if i % 7 == 0 else rng.normal(0.6, 0.15), 0.02, 0.95))
        # uncertainty anti-correlated with dice + noise (the pattern we hope to see)
        unc = float(np.clip(0.25 * (1 - true_d) + rng.normal(0, 0.02), 0, 1))
        recs.append({"id": f"c{i}", "dice": true_d, "uncertainty": unc, "size_cc": float(rng.uniform(2, 40)),
                     "foreground_roi_ece": float(np.clip(rng.normal(0.08, 0.02), 0, 1)),
                     "disagreement": float(np.clip(0.1 * (1 - true_d), 0, 1))})
    print(summarize(recs))
    tied_u = np.zeros(10)
    tied_d = np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
    print(f"tied uncertainty self-test: spearman={spearman(tied_u, tied_d)}, "
          f"AUROC={flagging_auroc(tied_u, tied_d):.3f} (want nan, 0.500)")
    # ece_reliability sanity: perfectly-calibrated synthetic -> ECE ~ 0
    p = rng.uniform(0, 1, 20000); g = (rng.uniform(0, 1, 20000) < p).astype(float)
    e, bins = ece_reliability(p, g, n_bins=15)
    print(f"ece_reliability self-test: ECE={e:.4f} (want small), n_bins_filled={len(bins)}")
