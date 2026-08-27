"""How much accuracy do we forfeit by producing the mask at reduced resolution?

EffiDec3D outputs the segmentation at (input / resolution_factor) and then restores
full resolution with a 1x1x1 SegHead + 2x trilinear upsampling (paper Eq. 4). This
script measures the UPPER BOUND that restoration imposes -- independent of any network
weights -- by taking the ground-truth mask itself, pushing it down to the coarse grid,
restoring it, and scoring the reconstruction against the original GT.

Interpretation: the numbers here are the BEST a perfectly-trained EffiDec3D decoder
could achieve at that resolution_factor. If the ceiling is already low, no amount of
training recovers it. The paper's own Table 4 shows only a 0.19% avg-Dice drop at
(D/2,H/2,W/2) -- but that is on BTCV (large abdominal organs). Rectal tumours are small,
so we re-measure the ceiling on OUR labels and stratify by tumour volume.

Restore variants scored (the "1x1 vs 3x3 / interpolation" question):
  * nearest    : subsample down, nearest-neighbour up   (what a hard/argmax pipeline gets)
  * trilinear  : area-average down, trilinear up + 0.5 threshold  (the paper's SegHead path)

Run from repo root (uses the local/server `python`, which has torch/scipy/nibabel):
    python analysis/metrics/resolution_ceiling.py --split validation
    python analysis/metrics/resolution_ceiling.py --split fibrosis --factors 2 4 8
"""
import os
import os.path as osp
import json
import argparse

import numpy as np
import nibabel as nib
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_erosion, distance_transform_edt


# ----------------------------- resolution round-trip -----------------------------
def roundtrip(mask, factor, mode):
    """Downsample a binary mask by `factor`, restore to original size, re-binarise.

    mode='nearest'   : strided subsample (float interp nearest) then nearest upsample.
    mode='trilinear' : area-average pool (fraction of tumour per coarse voxel) then
                       trilinear upsample, threshold at 0.5  ==  the paper's soft path.
    Returns the reconstructed binary mask at the ORIGINAL resolution.
    """
    D, H, W = mask.shape
    x = torch.from_numpy(mask.astype(np.float32))[None, None]  # (1,1,D,H,W)
    cd, ch, cw = max(1, D // factor), max(1, H // factor), max(1, W // factor)
    if mode == "nearest":
        down = F.interpolate(x, size=(cd, ch, cw), mode="nearest")
        up = F.interpolate(down, size=(D, H, W), mode="nearest")
        rec = up[0, 0].numpy() >= 0.5
    elif mode == "trilinear":
        # adaptive_avg_pool -> each coarse voxel holds the tumour volume fraction
        down = F.adaptive_avg_pool3d(x, output_size=(cd, ch, cw))
        up = F.interpolate(down, size=(D, H, W), mode="trilinear", align_corners=False)
        rec = up[0, 0].numpy() >= 0.5
    else:
        raise ValueError(mode)
    return rec


# ----------------------------- metrics -----------------------------
def dice(a, b):
    a, b = a.astype(bool), b.astype(bool)
    s = a.sum() + b.sum()
    if s == 0:
        return 1.0
    return 2.0 * np.logical_and(a, b).sum() / s


def _surfaces(a, b, spacing):
    a_surf = a ^ binary_erosion(a)
    b_surf = b ^ binary_erosion(b)
    if a_surf.sum() == 0 or b_surf.sum() == 0:
        return None
    dt_to_b = distance_transform_edt(~b_surf, sampling=spacing)
    dt_to_a = distance_transform_edt(~a_surf, sampling=spacing)
    return dt_to_b[a_surf], dt_to_a[b_surf]  # d(gt_surf->pred), d(pred_surf->gt)


def hd95(a, b, spacing):
    sd = _surfaces(a, b, spacing)
    if sd is None:
        return np.nan
    d_ab, d_ba = sd
    return float(np.percentile(np.concatenate([d_ab, d_ba]), 95))


def surface_dice(a, b, spacing, tau=2.0):
    sd = _surfaces(a, b, spacing)
    if sd is None:
        return np.nan
    d_ab, d_ba = sd
    return float((np.sum(d_ab <= tau) + np.sum(d_ba <= tau)) / (len(d_ab) + len(d_ba)))


def vol_bin(cc):
    if cc < 5:
        return "small (<5cc)"
    if cc < 15:
        return "mid (5-15cc)"
    return "large (>=15cc)"


# ----------------------------- driver -----------------------------
def gt_path(data_dir, item):
    if item.get("label"):
        return osp.join(data_dir, item["label"])
    return osp.join(data_dir, item["image"]).replace("imagesTs/", "labelsTs/").replace("_z800", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data_rectal")
    ap.add_argument("--json_file", default="Trainval_set1.json")
    ap.add_argument("--split", default="validation", help="JSON key: validation | testing | fibrosis")
    ap.add_argument("--factors", type=int, nargs="+", default=[2, 4, 8],
                    help="resolution_factor values to test")
    ap.add_argument("--modes", nargs="+", default=["trilinear", "nearest"])
    ap.add_argument("--tau", type=float, default=2.0, help="surface-DSC tolerance (mm)")
    ap.add_argument("--out", default="analysis/csvs/resolution_ceiling")
    args = ap.parse_args()

    with open(osp.join(args.data_dir, args.json_file)) as f:
        items = json.load(f)[args.split]
    print(f"{args.split}: {len(items)} cases | factors={args.factors} | modes={args.modes}")

    rows = []
    for i, item in enumerate(items):
        p = gt_path(args.data_dir, item)
        if not osp.exists(p):
            print("  missing:", p); continue
        nii = nib.load(p)
        spacing = tuple(float(s) for s in nii.header.get_zooms()[:3])
        gt = (nii.get_fdata() == 1)
        if gt.sum() == 0:
            continue
        vcc = gt.sum() * float(np.prod(spacing)) / 1000.0
        vb = vol_bin(vcc)
        for f_ in args.factors:
            for m in args.modes:
                rec = roundtrip(gt, f_, m)
                rec_cc = rec.sum() * float(np.prod(spacing)) / 1000.0
                rows.append(dict(
                    name=osp.basename(p), gt_cc=round(vcc, 2), vol_bin=vb,
                    factor=f_, mode=m,
                    dice=round(dice(gt, rec), 4),
                    surface_dsc=round(surface_dice(gt, rec, spacing, args.tau), 4),
                    hd95=round(hd95(gt, rec, spacing), 3),
                    vol_ratio=round(rec_cc / vcc, 4) if vcc > 0 else np.nan,
                ))
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(items)}")

    df = pd.DataFrame(rows)
    os.makedirs(osp.dirname(args.out), exist_ok=True)
    csv = f"{args.out}_{args.split}.csv"
    df.to_csv(csv, index=False)

    # ---- console summary: mean Dice / sDSC by resolution_factor x mode x volume bin ----
    order = ["small (<5cc)", "mid (5-15cc)", "large (>=15cc)"]
    print("\n" + "=" * 78)
    print("UPPER-BOUND (ceiling) accuracy of restoring GT from reduced resolution")
    print("=" * 78)
    for metric in ("dice", "surface_dsc"):
        print(f"\n### mean {metric} ###")
        piv = (df.groupby(["factor", "mode", "vol_bin"])[metric].mean().reset_index()
                 .pivot_table(index=["factor", "mode"], columns="vol_bin", values=metric))
        piv = piv.reindex(columns=[c for c in order if c in piv.columns])
        print(piv.round(3).to_string())
    print(f"\nn per volume bin: {df.drop_duplicates('name').groupby('vol_bin').size().to_dict()}")
    print(f"\nsaved -> {csv}")


if __name__ == "__main__":
    main()
