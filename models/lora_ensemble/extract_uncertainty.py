#!/usr/bin/env python
"""
LoRA uncertainty evaluator for 3D segmentation.

Runs on EITHER an ensemble checkpoint (--lora_members > 1) or a single-model baseline
(--lora_members 1), and reports, per case and pooled over the cohort:
  - lesion Dice (softmax-mean argmax vs GT)
  - case uncertainty:  ensemble = mean foreground member-variance;  single = mean foreground entropy
  - CALIBRATION: segmentation-adapted foreground ROI ECE + pooled reliability bins
  - DIVERSITY (ensemble only): mean member disagreement (std across members) - confirms members didn't
    collapse through the shared decoder
  - failure-flagging AUROC + Spearman(uncertainty, Dice)  (the clinical add-on)

Run the ensemble AND the single-LoRA baseline, then compare their ECE / flagging to make the
paper's "ensemble is better-calibrated than a single model at low cost" claim. Reuses
main_inference.load_model (same build + strict load as the efficiency runs) and the training val loader.

  python models/lora_ensemble/extract_uncertainty.py --ckpt runs/rectal_effidec3d_loraLDE4/model_final.pt \
      --lora_members 4 --lora_rank 4 --lora_decoder_ensemble --tag lde4
  python models/lora_ensemble/extract_uncertainty.py --ckpt runs/rectal_effidec3d_loraS_r4_bs2/model_final.pt \
      --lora_members 1 --lora_rank 4 --tag single
"""
from __future__ import annotations
import argparse, os, os.path as osp, types, csv, json
import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import generate_binary_structure, label as label_connected_components

import sys
sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))  # repo root

from monai.inferers import sliding_window_inference
from main_inference import load_model
from utils.data_utils import get_loader_v2_mri
from models.lora_ensemble.eval_uncertainty import (dice as _dice, case_uncertainty, summarize,
                                                   binary_entropy, ece_reliability)


def build_args(cli):
    return types.SimpleNamespace(
        model_name=cli.model_name, use_upernet=cli.use_upernet,
        roi_x=cli.roi_x, roi_y=cli.roi_y, roi_z=cli.roi_z,
        in_channels=1, out_channels=2, feature_size=48, norm_name="instance",
        swin_depths=tuple(cli.swin_depths), swin_num_heads=tuple(cli.swin_num_heads),
        swin_use_v2=cli.swin_use_v2,
        n_decoder_channels=cli.n_decoder_channels, resolution_factor=cli.resolution_factor,
        head_upsample=cli.head_upsample,
        dropout_rate=0.0, dropout_path_rate=0.0, use_checkpoint=False, spatial_dims=3,
        use_lora=True, lora_rank=cli.lora_rank, lora_members=cli.lora_members,
        decoder_ensemble=cli.decoder_ensemble, lora_decoder_ensemble=cli.lora_decoder_ensemble,
        pretrained_model_path=cli.ckpt,
        data_dir=cli.data_dir, json_list=cli.json_list, train_key=cli.train_key, val_key=cli.val_key,
        label_key=cli.label_key, fallback_label_key=cli.fallback_label_key,
        prefer_label_plus=cli.prefer_label_plus,
        a_min=cli.a_min, a_max=cli.a_max, b_min=0.0, b_max=1.0,
        space_x=cli.space_x, space_y=cli.space_y, space_z=cli.space_z,
        workers=2, batch_size=1, distributed=False, use_normal_dataset=True,
    )


def _ensemble_predictor(model, n):
    def _pred(x):
        lg = model(x)                                            # [N,B,C,...]
        b, c = lg.shape[1], lg.shape[2]
        return lg.permute(1, 0, 2, *range(3, lg.dim())).reshape(b, n * c, *lg.shape[3:])
    return _pred


def _meta_value(meta, key, default=None):
    if meta is None or key not in meta:
        return default
    value = meta[key]
    if isinstance(value, (list, tuple)):
        value = value[0]
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return value


def _save_map(path, array, affine):
    os.makedirs(osp.dirname(path), exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(array, dtype=np.float32), affine), path)


def _remove_small_components(mask: np.ndarray, min_component_voxels: int) -> np.ndarray:
    """Drop connected foreground components with <= min_component_voxels voxels."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lora_rank", type=int, default=4)
    ap.add_argument("--lora_members", type=int, default=4)
    ap.add_argument("--decoder_ensemble", action="store_true",
                    help="checkpoint uses the LoRA + independent decoder ensemble upper-bound variant")
    ap.add_argument("--lora_decoder_ensemble", action="store_true",
                    help="checkpoint uses per-member LoRA adapters plus per-member decoders")
    ap.add_argument("--tag", default="ensemble", help="label for output files (e.g. ensemble / single)")
    ap.add_argument("--model_name", default="effidec3d",
                    choices=["effidec3d", "swinv2", "voco_swinunetr", "swinunetr", "smit"],
                    help="architecture key used by the LoRA checkpoint; default is the current EffiDec3D test")
    ap.add_argument("--use_upernet", action="store_true", help="SMIT UPerNet decoder; must match training")
    ap.add_argument("--swin_depths", default=(2, 2, 2, 2), type=int, nargs=4)
    ap.add_argument("--swin_num_heads", default=(3, 6, 12, 24), type=int, nargs=4)
    ap.add_argument("--swin_use_v2", action=argparse.BooleanOptionalAction, default=True,
                    help="use MONAI SwinUNETR V2 blocks; default True for EffiDec3D/SwinV2 LoRA runs")
    ap.add_argument("--n_decoder_channels", type=int, default=48)
    ap.add_argument("--resolution_factor", type=int, default=2)
    ap.add_argument("--head_upsample", default="trilinear",
                    choices=["none", "trilinear", "upconv", "upconv_refine"])
    ap.add_argument("--data_dir", default="data_rectal")
    ap.add_argument("--json_list", default="Trainval_set1.json")
    ap.add_argument("--train_key", default="training")
    ap.add_argument("--val_key", default="validation", help="JSON split key to evaluate, e.g. validation/testing")
    ap.add_argument("--label_key", default="label")
    ap.add_argument("--fallback_label_key", default="label_plus")
    ap.add_argument("--prefer_label_plus", action="store_true")
    ap.add_argument("--a_min", type=float, default=0.0)
    ap.add_argument("--a_max", type=float, default=800.0)
    ap.add_argument("--space_x", type=float, default=1.0)
    ap.add_argument("--space_y", type=float, default=1.0)
    ap.add_argument("--space_z", type=float, default=1.0)
    ap.add_argument("--roi_x", type=int, default=96)
    ap.add_argument("--roi_y", type=int, default=96)
    ap.add_argument("--roi_z", type=int, default=64)
    ap.add_argument("--sw_batch", type=int, default=2)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--n_bins", type=int, default=15)
    ap.add_argument("--min_component_voxels", type=int, default=0,
                    help="if >0, remove connected foreground islands with <= this many voxels before "
                         "Dice/calibration/uncertainty scoring. Use 10 to match metrics_rectal.py.")
    ap.add_argument("--outdir", default="analysis/uncertainty")
    ap.add_argument("--save_maps", action="store_true",
                    help="save evaluation-space NIfTI maps for snapshots: image, label, prediction, "
                         "mean foreground probability, entropy, and ensemble variance/std when available")
    ap.add_argument("--maps_dir", default=None,
                    help="directory for --save_maps outputs; defaults to <outdir>/uncertainty_maps_<tag>")
    cli = ap.parse_args()

    # Single source of truth: take geometry/arch from the checkpoint's stored training args so eval can
    # never drift from training (this is what the roi_z=96-vs-64 slip taught us). CLI geometry becomes a
    # fallback for older checkpoints that lack stored args.
    _ck = torch.load(cli.ckpt, map_location="cpu", weights_only=False)
    _ta = _ck.get("args") if isinstance(_ck, dict) else None
    # Older checkpoints stored the raw argparse.Namespace instead of vars(args).
    if _ta is not None and not isinstance(_ta, dict):
        _ta = getattr(_ta, "__dict__", None)
    if _ta:
        for k in ("roi_x", "roi_y", "roi_z", "model_name", "swin_use_v2", "n_decoder_channels",
                  "resolution_factor", "head_upsample", "feature_size", "lora_rank", "lora_members",
                  "decoder_ensemble", "lora_decoder_ensemble"):
            if k in _ta and hasattr(cli, k):
                setattr(cli, k, _ta[k])
        print(f"[extract] geometry from checkpoint: roi=({cli.roi_x},{cli.roi_y},{cli.roi_z}) "
              f"model={cli.model_name} v2={cli.swin_use_v2} "
              f"lora_members={cli.lora_members} decoder_ensemble={cli.decoder_ensemble} "
              f"lora_decoder_ensemble={cli.lora_decoder_ensemble}")
    else:
        print("[extract][WARN] checkpoint has no stored args - using CLI geometry (verify it matches training).")

    if cli.decoder_ensemble and cli.lora_members <= 1:
        raise ValueError("--decoder_ensemble checkpoints require lora_members > 1.")
    if cli.lora_decoder_ensemble and cli.lora_members <= 1:
        raise ValueError("--lora_decoder_ensemble checkpoints require lora_members > 1.")
    if cli.decoder_ensemble and cli.lora_decoder_ensemble:
        raise ValueError("decoder_ensemble and lora_decoder_ensemble are mutually exclusive.")

    is_ens = cli.lora_members > 1
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(build_args(cli), dev); model.eval()
    _, val_loader = get_loader_v2_mri(build_args(cli))
    predictor = _ensemble_predictor(model, cli.lora_members) if is_ens else model

    edges = np.linspace(0.0, 1.0, cli.n_bins + 1)
    conf_sum = np.zeros(cli.n_bins); acc_sum = np.zeros(cli.n_bins); cnt = np.zeros(cli.n_bins)
    records = []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            x = batch["image"].to(dev)
            y = (batch["label"].cpu().numpy()[0, 0] > 0)                     # [D,H,W] GT fg
            y = _remove_small_components(y, cli.min_component_voxels)
            meta = batch.get("image_meta_dict")
            name = (osp.basename(str(meta["filename_or_obj"][0])).replace(".nii.gz", "")
                    if meta is not None and "filename_or_obj" in meta else f"case{i:03d}")
            affine = _meta_value(meta, "affine", np.eye(4))

            out = sliding_window_inference(x, (cli.roi_x, cli.roi_y, cli.roi_z), cli.sw_batch,
                                           predictor, overlap=cli.overlap, mode="gaussian")
            if is_ens:
                b, nc = out.shape[0], out.shape[1]; c = nc // cli.lora_members
                probs = torch.softmax(out.view(b, cli.lora_members, c, *out.shape[2:]), dim=2)  # [B,N,C,...]
                probs_fg = probs[0, :, 1].cpu().numpy()                       # [N,D,H,W]
                mean_fg = probs_fg.mean(0)
                var_fg = probs_fg.var(0)
                disagreement = None
            else:
                probs = torch.softmax(out, dim=1)                            # [B,C,...]
                mean_fg = probs[0, 1].cpu().numpy()
                probs_fg = None; var_fg = None

            pred = _remove_small_components(mean_fg >= 0.5, cli.min_component_voxels)
            eval_mean_fg = mean_fg.copy()
            if cli.min_component_voxels > 0:
                # Score the deployed postprocessed decision rule: removed high-probability islands
                # become background probability. GT voxels stay in the calibration ROI below.
                eval_mean_fg[~pred] = np.where(mean_fg[~pred] >= 0.5, 0.0, mean_fg[~pred])
            d = _dice(pred, y)
            roi = (eval_mean_fg >= 0.05) | y                                 # calibration region of interest
            foreground_roi_ece, _ = ece_reliability(eval_mean_fg, y, roi, n_bins=cli.n_bins)
            # pooled reliability accumulation
            p_roi = eval_mean_fg[roi]; g_roi = y[roi].astype(float)
            idx = np.clip(np.digitize(p_roi, edges) - 1, 0, cli.n_bins - 1)
            for bb in range(cli.n_bins):
                sel = idx == bb
                if sel.any():
                    cnt[bb] += sel.sum(); conf_sum[bb] += p_roi[sel].sum(); acc_sum[bb] += g_roi[sel].sum()

            region = pred
            if is_ens:
                unc_pred = float(var_fg[region].mean()) if region.any() else float(var_fg.mean())
                unc_gt = float(var_fg[y].mean()) if y.any() else float("nan")
                unc_global = float(var_fg.mean())
                dis = float(probs_fg.std(0)[region].mean()) if region.any() else float(probs_fg.std(0).mean())
            else:
                ent = binary_entropy(eval_mean_fg)
                unc_pred = float(ent[region].mean()) if region.any() else float(ent.mean())
                unc_gt = float(ent[y].mean()) if y.any() else float("nan")
                unc_global = float(ent.mean())
                dis = None

            if cli.save_maps:
                maps_dir = cli.maps_dir or osp.join(cli.outdir, f"uncertainty_maps_{cli.tag}")
                stem = name.replace(".nii.gz", "")
                ent_map = binary_entropy(mean_fg)
                _save_map(osp.join(maps_dir, f"{stem}_image.nii.gz"),
                          batch["image"].cpu().numpy()[0, 0], affine)
                _save_map(osp.join(maps_dir, f"{stem}_label.nii.gz"), y.astype(np.float32), affine)
                _save_map(osp.join(maps_dir, f"{stem}_pred.nii.gz"), pred.astype(np.float32), affine)
                _save_map(osp.join(maps_dir, f"{stem}_prob.nii.gz"), eval_mean_fg, affine)
                _save_map(osp.join(maps_dir, f"{stem}_entropy.nii.gz"), ent_map, affine)
                if is_ens:
                    std_fg = np.sqrt(np.maximum(var_fg, 0.0))
                    _save_map(osp.join(maps_dir, f"{stem}_variance.nii.gz"), var_fg, affine)
                    _save_map(osp.join(maps_dir, f"{stem}_std.nii.gz"), std_fg, affine)

            rec = {"id": name, "dice": d, "uncertainty": unc_pred,
                   "unc_pred_region": unc_pred, "unc_gt_region": unc_gt,
                   "unc_global": unc_global, "foreground_roi_ece": foreground_roi_ece,
                   "calib_roi_voxels": int(roi.sum()), "size_cc": float(y.sum()) / 1000.0}
            if dis is not None:
                rec["disagreement"] = dis
            records.append(rec)
            print(f"  {name:32s} dice={d:.3f} unc={unc_pred:.5f} fgROI-ece={foreground_roi_ece:.4f}"
                  + (f" disag={dis:.5f}" if dis is not None else ""))

    os.makedirs(cli.outdir, exist_ok=True)
    per_case = osp.join(cli.outdir, f"lora_uncertainty_{cli.tag}.csv")
    rel_csv = osp.join(cli.outdir, f"lora_reliability_{cli.tag}.csv")
    summary_json = osp.join(cli.outdir, f"lora_uncertainty_{cli.tag}_summary.json")
    fields = ["id", "dice", "uncertainty", "unc_pred_region", "unc_gt_region", "unc_global",
              "foreground_roi_ece", "calib_roi_voxels", "size_cc"] + (["disagreement"] if is_ens else [])
    with open(per_case, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in records:
            w.writerow(r)
    with open(rel_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["bin_lo", "bin_hi", "conf", "acc", "count"])
        for bb in range(cli.n_bins):
            if cnt[bb] > 0:
                w.writerow([edges[bb], edges[bb + 1], conf_sum[bb] / cnt[bb], acc_sum[bb] / cnt[bb], int(cnt[bb])])
    tot = cnt.sum()
    pooled_ece = float(np.sum((cnt / tot) * np.abs(np.divide(conf_sum, cnt, out=np.zeros_like(cnt), where=cnt > 0)
                                                   - np.divide(acc_sum, cnt, out=np.zeros_like(cnt), where=cnt > 0)))) if tot else float("nan")

    s = summarize(records)
    s["pooled_foreground_roi_ece"] = pooled_ece
    with open(summary_json, "w") as f:
        json.dump(s, f, indent=2)
    print(f"\n=== UNCERTAINTY SUMMARY [{cli.tag}] ===")
    print(json.dumps(s, indent=2))
    print(f"per-case -> {per_case}\nreliability -> {rel_csv}\nsummary -> {summary_json}")
    if is_ens:
        print("want: spearman_unc_dice<<0, flagging_auroc>>0.5, mean_disagreement>0 (not collapsed), low pooled_foreground_roi_ece")
    else:
        print("baseline (single model): compare its pooled_foreground_roi_ece / flagging_auroc against the ensemble's")


if __name__ == "__main__":
    main()
