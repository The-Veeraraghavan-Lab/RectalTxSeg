"""Rank cases where tumor voxels receive low foreground probability.

Run this on the server after probability maps have been exported. It is meant
to support a compact qualitative figure explaining reliability-diagram bins:
manual tumor is present, but one or more models assign low foreground
confidence to parts of that tumor.

Example:
python analysis/uncertainty/select_low_confidence_tumor_cases.py \
  --image-dir data_rectal/imagesTs \
  --label-dir data_rectal/labelsTs \
  --model VoxelFox=results/rectal_voxelfox_swinunetr_96x96x64_pretrained/voxelfox_swinunetr/testing \
  --model VF-EffiDec=results/rectal_effidec3d_96x96x64_pretrained/effidec3d/testing \
  --model VF-LoRA=results/rectal_effidec3d_loraS_r4_bs2/effidec3d/testing \
  --model VF-LDE4=results/rectal_effidec3d_loraLDE4/effidec3d/testing \
  --output analysis/uncertainty/low_confidence_tumor_voxels.csv \
  --case-output analysis/uncertainty/low_confidence_tumor_case_summary.csv \
  --manifest-output analysis/uncertainty/low_confidence_tumor_copy_manifest.txt \
  --top-k 12
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


def strip_nii(name: str) -> str:
    return re.sub(r"\.nii(\.gz)?$", "", name)


def base_case_id(path: Path) -> str:
    stem = strip_nii(path.name)
    stem = re.sub(r"_(prob|seg)$", "", stem)
    return re.sub(r"_z800$", "", stem)


def parse_model(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("--model must look like Label=/path/to/prob_dir")
    label, root = spec.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("model label cannot be empty")
    return label, Path(root)


def dice_from_prob(prob: np.ndarray, gt: np.ndarray, threshold: float) -> float:
    pred = prob >= threshold
    inter = float(np.logical_and(pred, gt).sum())
    denom = float(pred.sum() + gt.sum())
    return 2.0 * inter / denom if denom > 0 else np.nan


def summarize_one(prob_path: Path, label_dir: Path, threshold: float) -> dict[str, float | str]:
    case_id = base_case_id(prob_path)
    label_path = label_dir / f"{case_id}.nii.gz"
    if not label_path.exists():
        label_path = label_dir / f"{case_id}_z800.nii.gz"
    if not label_path.exists():
        raise FileNotFoundError(f"Missing label for {case_id}: {label_dir}")

    prob = nib.load(str(prob_path)).get_fdata(dtype=np.float32)
    gt = nib.load(str(label_path)).get_fdata() > 0.5
    if prob.shape != gt.shape:
        raise ValueError(f"Shape mismatch for {case_id}: {prob.shape} vs {gt.shape}")

    gt_prob = prob[gt]
    pred = prob >= threshold
    overlap = np.logical_and(pred, gt)
    return {
        "case_id": case_id,
        "prob_path": str(prob_path),
        "label_path": str(label_path),
        "gt_voxels": int(gt.sum()),
        "pred_voxels": int(pred.sum()),
        "overlap_voxels": int(overlap.sum()),
        "dice_at_threshold": dice_from_prob(prob, gt, threshold),
        "gt_prob_mean": float(np.mean(gt_prob)) if gt_prob.size else np.nan,
        "gt_prob_p10": float(np.percentile(gt_prob, 10)) if gt_prob.size else np.nan,
        "gt_prob_p50": float(np.percentile(gt_prob, 50)) if gt_prob.size else np.nan,
        "frac_gt_prob_lt_005": float(np.mean(gt_prob < 0.05)) if gt_prob.size else np.nan,
        "frac_gt_prob_lt_010": float(np.mean(gt_prob < 0.10)) if gt_prob.size else np.nan,
        "frac_gt_prob_lt_050": float(np.mean(gt_prob < 0.50)) if gt_prob.size else np.nan,
    }


def image_path_for_case(image_dir: Path, case_id: str) -> str:
    for name in (f"{case_id}_z800.nii.gz", f"{case_id}.nii.gz"):
        path = image_dir / name
        if path.exists():
            return str(path)
    return ""


def seg_path_for_prob(prob_path: str) -> str:
    path = Path(prob_path)
    seg = path.with_name(path.name.replace("_prob.nii.gz", "_seg.nii.gz"))
    return str(seg) if seg.exists() else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, default=Path("data_rectal/imagesTs"))
    parser.add_argument("--label-dir", type=Path, default=Path("data_rectal/labelsTs"))
    parser.add_argument("--model", action="append", type=parse_model, required=True)
    parser.add_argument("--output", type=Path, default=Path("analysis/uncertainty/low_confidence_tumor_voxels.csv"))
    parser.add_argument("--case-output", type=Path, default=Path("analysis/uncertainty/low_confidence_tumor_case_summary.csv"))
    parser.add_argument("--manifest-output", type=Path, default=Path("analysis/uncertainty/low_confidence_tumor_copy_manifest.txt"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=12)
    args = parser.parse_args()

    rows = []
    for model, root in args.model:
        prob_paths = sorted(root.glob("*_prob.nii.gz"))
        if not prob_paths:
            raise FileNotFoundError(f"No *_prob.nii.gz files found for {model}: {root}")
        for prob_path in prob_paths:
            row = summarize_one(prob_path, args.label_dir, args.threshold)
            row["model"] = model
            row["seg_path"] = seg_path_for_prob(row["prob_path"])
            row["image_path"] = image_path_for_case(args.image_dir, row["case_id"])
            rows.append(row)

    per_model = pd.DataFrame(rows)
    per_model["low_conf_score"] = (
        per_model["frac_gt_prob_lt_005"].fillna(0.0)
        + 0.50 * per_model["frac_gt_prob_lt_010"].fillna(0.0)
        + 0.25 * (1.0 - per_model["dice_at_threshold"].fillna(0.0))
    )
    per_model = per_model.sort_values(["low_conf_score", "frac_gt_prob_lt_005"], ascending=False)

    case_summary = (
        per_model.groupby("case_id")
        .agg(
            image_path=("image_path", "first"),
            label_path=("label_path", "first"),
            gt_voxels=("gt_voxels", "first"),
            worst_model=("model", lambda x: per_model.loc[x.index].sort_values("low_conf_score", ascending=False)["model"].iloc[0]),
            max_frac_gt_prob_lt_005=("frac_gt_prob_lt_005", "max"),
            max_frac_gt_prob_lt_010=("frac_gt_prob_lt_010", "max"),
            max_frac_gt_prob_lt_050=("frac_gt_prob_lt_050", "max"),
            min_dice=("dice_at_threshold", "min"),
            n_models_with_half_gt_below_threshold=("frac_gt_prob_lt_050", lambda x: int((x > 0.50).sum())),
            case_score=("low_conf_score", "max"),
        )
        .sort_values(["case_score", "max_frac_gt_prob_lt_005"], ascending=False)
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    per_model.to_csv(args.output, index=False)
    case_summary.to_csv(args.case_output)

    selected = list(case_summary.head(args.top_k).index)
    manifest_paths: list[str] = []
    for case_id in selected:
        crows = per_model.loc[per_model["case_id"] == case_id]
        for col in ("image_path", "label_path", "prob_path", "seg_path"):
            manifest_paths.extend([p for p in crows[col].dropna().astype(str).tolist() if p])
    unique_paths = sorted(set(manifest_paths))
    args.manifest_output.write_text("\n".join(unique_paths) + "\n")

    print(f"Wrote per-model rows: {args.output}")
    print(f"Wrote case summary:   {args.case_output}")
    print(f"Wrote copy manifest:  {args.manifest_output}")
    print("\nTop cases:")
    print(case_summary.head(args.top_k).to_string())


if __name__ == "__main__":
    main()
