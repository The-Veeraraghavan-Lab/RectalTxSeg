"""Local folder inference helper.

Point this script at one NIfTI, a folder of NIfTIs, one DICOM series folder, or
a folder containing multiple DICOM series folders. It creates a temporary
MONAI-Decathlon testing manifest and calls ``main_inference.py`` once for all
images.

The default path assumes raw DICOM/NIfTI input and uses per-volume percentile
scaling. If the input images are already prepared in [0, 800], use
``--intensity_mode fixed``.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pydicom
import SimpleITK as sitk


PYTHON = sys.executable
REPO_ROOT = Path(__file__).resolve().parents[2]
NIFTI_SUFFIXES = (".nii", ".nii.gz")
THROWAWAY_STEMS = {"ctimg", "mrimg", "scan", "img", "image"}


def is_nifti(path: Path) -> bool:
    return path.name.lower().endswith(NIFTI_SUFFIXES)


def strip_nifti_suffix(path: Path) -> str:
    name = path.name
    if name.lower().endswith(".nii.gz"):
        return name[:-7]
    if name.lower().endswith(".nii"):
        return name[:-4]
    return path.stem


def sanitize_name(text: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return clean or "case"


def unique_name(base: str, used: set[str]) -> str:
    base = sanitize_name(base)
    if len(base) < 4 or base.lower() in THROWAWAY_STEMS:
        base = f"case_{base}"
    candidate = base
    idx = 2
    while candidate in used:
        candidate = f"{base}_{idx:02d}"
        idx += 1
    used.add(candidate)
    return candidate


def find_nifti_files(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path] if is_nifti(input_path) else []

    pattern = "**/*" if recursive else "*"
    return sorted(
        path for path in input_path.glob(pattern)
        if path.is_file() and is_nifti(path)
    )


def dicom_files_in_dir(directory: Path, max_probe: Optional[int] = None) -> list[Path]:
    files = [path for path in sorted(directory.iterdir()) if path.is_file()]
    if max_probe is not None:
        files = files[:max_probe]

    dicom_files = []
    for path in files:
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        except Exception:
            continue
        modality = getattr(ds, "Modality", "")
        if modality == "SEG" or path.name.startswith("AI_seg"):
            continue
        dicom_files.append(path)
    return dicom_files


def find_dicom_series_dirs(input_path: Path, recursive: bool) -> list[Path]:
    if not input_path.is_dir():
        return []

    if dicom_files_in_dir(input_path, max_probe=30):
        return [input_path]

    pattern = "**/*" if recursive else "*"
    return sorted(
        path for path in input_path.glob(pattern)
        if path.is_dir() and dicom_files_in_dir(path, max_probe=30)
    )


def convert_dicom_series_to_nifti(dicom_dir: Path, output_nii: Path) -> Path:
    dicom_files = [str(path) for path in dicom_files_in_dir(dicom_dir)]
    if not dicom_files:
        raise RuntimeError(f"No image DICOM files found in {dicom_dir}")

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(dicom_files)
    image = reader.Execute()
    output_nii.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(output_nii))
    return output_nii


def case_name_for_nifti(path: Path, input_root: Path, used: set[str]) -> str:
    stem = strip_nifti_suffix(path)
    if stem.lower() in THROWAWAY_STEMS and path.parent != input_root:
        stem = path.parent.name

    base = sanitize_name(stem)
    if base in used:
        try:
            rel = path.relative_to(input_root)
            parts = [sanitize_name(part) for part in rel.parts[:-1]]
            base = sanitize_name("_".join([*parts, stem]))
        except ValueError:
            pass
    return unique_name(base, used)


def stage_nifti_files(nifti_files: list[Path], input_root: Path, images_dir: Path) -> list[Path]:
    used: set[str] = set()
    staged = []
    images_dir.mkdir(parents=True, exist_ok=True)

    for src in nifti_files:
        case_name = case_name_for_nifti(src, input_root, used)
        suffix = ".nii.gz" if src.name.lower().endswith(".nii.gz") else ".nii"
        dest = images_dir / f"{case_name}{suffix}"
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)
        staged.append(dest)
    return staged


def create_folder_dataset(input_path: Path, work_dir: Path, recursive: bool) -> tuple[Path, list[Path]]:
    input_path = input_path.expanduser().resolve()
    dataset_dir = work_dir.expanduser().resolve() / "dataset"
    images_dir = dataset_dir / "imagesTs"
    images_dir.mkdir(parents=True, exist_ok=True)

    input_root = input_path.parent if input_path.is_file() else input_path
    nifti_files = find_nifti_files(input_path, recursive=recursive)

    if nifti_files:
        staged_images = stage_nifti_files(nifti_files, input_root, images_dir)
    else:
        dicom_dirs = find_dicom_series_dirs(input_path, recursive=recursive)
        if not dicom_dirs:
            raise RuntimeError(f"No NIfTI or DICOM image inputs found in {input_path}")
        used: set[str] = set()
        staged_images = []
        for dicom_dir in dicom_dirs:
            case_name = unique_name(dicom_dir.name, used)
            staged_images.append(
                convert_dicom_series_to_nifti(dicom_dir, images_dir / f"{case_name}.nii.gz")
            )

    dataset = {
        "name": "folder_rectal_inference",
        "tensorImageSize": "3D",
        "testing": [{"image": f"./imagesTs/{path.name}"} for path in staged_images],
    }
    json_path = dataset_dir / "folder_inference.json"
    json_path.write_text(json.dumps(dataset, indent=2))
    return json_path, staged_images


def build_inference_command(args, json_path: Path, passthrough: list[str]) -> list[str]:
    if args.nproc_per_node > 1:
        cmd = ["torchrun", f"--nproc_per_node={args.nproc_per_node}", "main_inference.py", "--distributed"]
    else:
        cmd = [PYTHON, "main_inference.py"]

    cmd.extend([
        "--data_dir", str(json_path.parent),
        "--json_list", json_path.name,
        "--datasets", "testing",
        "--results_dir", args.results_dir,
        "--output_dir", args.output_dir,
        "--model_name", args.model_name,
        "--pretrained_model_path", args.checkpoint,
        "--in_channels", str(args.in_channels),
        "--out_channels", str(args.out_channels),
        "--feature_size", str(args.feature_size),
        "--norm_name", args.norm_name,
        "--intensity_mode", args.intensity_mode,
        "--space_x", str(args.space_x),
        "--space_y", str(args.space_y),
        "--space_z", str(args.space_z),
        "--roi_x", str(args.roi_x),
        "--roi_y", str(args.roi_y),
        "--roi_z", str(args.roi_z),
        "--sw_batch_size", str(args.sw_batch_size),
        "--infer_overlap", str(args.infer_overlap),
    ])

    if args.pred_subdir:
        cmd.extend(["--pred_subdir", args.pred_subdir])
    if args.intensity_mode == "fixed":
        cmd.extend(["--a_min", str(args.a_min), "--a_max", str(args.a_max)])
    else:
        cmd.extend([
            "--percentile_lower", str(args.percentile_lower),
            "--percentile_upper", str(args.percentile_upper),
        ])
    if args.use_tta:
        cmd.append("--use_tta")
    if args.save_probs:
        cmd.append("--save_probs")
    if args.skip_orientation:
        cmd.append("--skip_orientation")
    if args.use_upernet:
        cmd.append("--use_upernet")
    if args.swin_use_v2:
        cmd.append("--swin_use_v2")

    cmd.extend(passthrough)
    return cmd


def run_inference(args, json_path: Path, passthrough: list[str]) -> None:
    cmd = build_inference_command(args, json_path, passthrough)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def save_overlay_preview(image_path: Path, seg_path: Path, output_pdf: Path, title: str) -> Path:
    image = nib.load(str(image_path)).get_fdata()
    seg = nib.load(str(seg_path)).get_fdata()

    pos_slices = np.where(seg.sum(axis=(0, 1)) > 0)[0]
    slice_idx = int(pos_slices[len(pos_slices) // 2]) if len(pos_slices) else image.shape[2] // 2

    img_slice = image[:, :, slice_idx]
    seg_slice = seg[:, :, slice_idx]
    p1, p99 = np.percentile(img_slice, [1, 99])

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(np.rot90(img_slice), cmap="gray", vmin=p1, vmax=p99)
    if np.any(seg_slice > 0):
        ax.contour(np.rot90(seg_slice > 0), levels=[0.5], colors=["#d94801"], linewidths=2)
    ax.set_title(title)
    ax.axis("off")
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return output_pdf


def result_root(args) -> Path:
    root = Path(args.results_dir).expanduser()
    return root if root.is_absolute() else REPO_ROOT / root


def prediction_path(args, image_path: Path) -> Path:
    pred_subdir = args.pred_subdir or args.model_name
    seg_name = f"{strip_nifti_suffix(image_path)}_seg.nii.gz"
    return result_root(args) / args.output_dir / pred_subdir / "testing" / seg_name


def maybe_save_previews(args, staged_images: list[Path]) -> None:
    if not args.save_previews:
        return

    preview_dir = args.work_dir.expanduser().resolve() / "previews"
    limit = len(staged_images) if args.preview_limit <= 0 else args.preview_limit
    for image_path in staged_images[:limit]:
        seg_path = prediction_path(args, image_path)
        if not seg_path.exists():
            print(f"[preview] skipped missing segmentation: {seg_path}")
            continue
        preview_path = preview_dir / f"{strip_nifti_suffix(image_path)}_overlay.pdf"
        save_overlay_preview(image_path, seg_path, preview_path, strip_nifti_suffix(image_path))
        print(f"[preview] saved {preview_path}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run rectal tumor inference on a single image or a folder of images."
    )
    parser.add_argument("--input", required=True, type=Path,
                        help="NIfTI file, folder of NIfTIs, DICOM series folder, or folder of DICOM series folders")
    parser.add_argument("--recursive", action="store_true",
                        help="search recursively for NIfTIs or DICOM series folders")
    parser.add_argument("--work_dir", type=Path,
                        default=REPO_ROOT / "analysis" / "results" / "folder_inference_workspace",
                        help="where to write the temporary dataset JSON/images and optional previews")

    parser.add_argument("--checkpoint", default="runs/rectal_smit_128x128x64_pretrained/model_final.pt",
                        help="model checkpoint to load")
    parser.add_argument("--model_name", default="smit",
                        help="architecture key passed to main_inference.py")
    parser.add_argument("--pred_subdir", default=None,
                        help="prediction subfolder name; defaults to model_name")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--output_dir", default="folder_inference")

    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--out_channels", type=int, default=2)
    parser.add_argument("--feature_size", type=int, default=48)
    parser.add_argument("--norm_name", default="instance")
    parser.add_argument("--use_upernet", action="store_true")
    parser.add_argument("--swin_use_v2", action="store_true")

    parser.add_argument("--intensity_mode", choices=["fixed", "percentile"], default="percentile",
                        help="percentile is for raw inputs; fixed is for already prepared [0, 800] images")
    parser.add_argument("--a_min", type=float, default=0.0)
    parser.add_argument("--a_max", type=float, default=800.0)
    parser.add_argument("--percentile_lower", type=float, default=0.0)
    parser.add_argument("--percentile_upper", type=float, default=99.5)

    parser.add_argument("--space_x", type=float, default=1.0)
    parser.add_argument("--space_y", type=float, default=1.0)
    parser.add_argument("--space_z", type=float, default=1.0)
    parser.add_argument("--roi_x", type=int, default=128)
    parser.add_argument("--roi_y", type=int, default=128)
    parser.add_argument("--roi_z", type=int, default=64)
    parser.add_argument("--sw_batch_size", type=int, default=8)
    parser.add_argument("--infer_overlap", type=float, default=0.5)
    parser.add_argument("--use_tta", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip_orientation", action="store_true")
    parser.add_argument("--save_probs", action="store_true")
    parser.add_argument("--nproc_per_node", type=int, default=1,
                        help="set >1 to launch distributed inference with torchrun")

    parser.add_argument("--save_previews", action="store_true",
                        help="save PDF overlays after inference")
    parser.add_argument("--preview_limit", type=int, default=0,
                        help="maximum previews to save; 0 means all")
    return parser


def main() -> None:
    parser = build_argparser()
    args, passthrough = parser.parse_known_args()

    json_path, staged_images = create_folder_dataset(args.input, args.work_dir, args.recursive)
    print(f"Prepared {len(staged_images)} image(s)")
    print(f"Dataset JSON: {json_path}")

    run_inference(args, json_path, passthrough)
    maybe_save_previews(args, staged_images)

    pred_subdir = args.pred_subdir or args.model_name
    print("Folder inference complete.")
    print(f"Predictions: {result_root(args) / args.output_dir / pred_subdir / 'testing'}")


if __name__ == "__main__":
    main()
