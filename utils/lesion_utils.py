"""Lesion-level geometry/binning utilities shared across analysis and eval scripts.

_key_name_part MIRRORS main_inference.get_key_name_part - KEEP IN SYNC.
"""
import os.path as osp

import numpy as np
from scipy.ndimage import label as cc_label, generate_binary_structure

STRUCT = generate_binary_structure(3, 3)                      # 26-connectivity for 3D CC labelling
BINS = [(0, 5), (5, 10), (10, 20), (20, 40), (40, 1e9)]
BIN_ORDER = [f"{lo}-{hi if hi < 1e9 else '+'}cc" for lo, hi in BINS]


def bin_label(cc):
    for lo, hi in BINS:
        if lo <= cc < hi:
            return f"{lo}-{hi if hi < 1e9 else '+'}cc"


def _gt_lesions(gt, spacing, min_vox=10):
    """Connected-component GT lesions (>min_vox voxels), relabelled 1..K, with per-lesion cc volumes."""
    lab, n = cc_label(gt == 1, structure=STRUCT)
    if n == 0:
        return lab, []
    counts = np.bincount(lab.ravel())
    keep = [i for i in range(1, len(counts)) if counts[i] > min_vox]
    clean = np.where(np.isin(lab, keep), lab, 0)
    lab2, n2 = cc_label(clean > 0, structure=STRUCT)
    voxvol = float(np.prod(spacing)) / 1000.0
    return lab2, [(lab2 == c).sum() * voxvol for c in range(1, n2 + 1)]


def _key_name_part(path, min_len=4):
    """MIRROR of main_inference.get_key_name_part — KEEP IN SYNC. Determines the file key that
    --save_probs uses to name <key>_prob.nii.gz. Falls back to the parent dir name for
    short/generic stems, so external cohorts don't silently miss their prob files."""
    fname = osp.basename(path)
    stem = fname.split(".")[0]
    parent = osp.basename(osp.dirname(path))
    throwaway = {"ctimg", "mrimg", "scan", "img", "image"}
    if len(stem) < min_len or stem.lower() in throwaway:
        return parent
    return stem
