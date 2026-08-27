#!/usr/bin/env python
"""Create a standalone GT-vs-AI contour overlay from NIfTI files.

Example:
  python standalone_contour_overlay.py \
    --image image_z800.nii.gz \
    --gt label.nii.gz \
    --ai prediction_seg.nii.gz \
    --out overlay.pdf

The output shows one axial slice with the manual/GT contour in green and the
AI/prediction contour in orange. The slice is chosen from the largest contour
area by default, and the view is cropped around the union of GT and AI.
"""

import argparse
import os.path as osp

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.lines import Line2D


GT_COLOR = "#2ca02c"
AI_COLOR = "#ff7f0e"


def load_image(path):
    nii = nib.load(path)
    return nii.get_fdata(), nii


def load_mask(path, like_shape):
    if not path:
        return np.zeros(like_shape, dtype=bool)
    mask = nib.load(path).get_fdata() > 0.5
    if mask.shape != like_shape:
        raise ValueError(f"Shape mismatch for {path}: {mask.shape} != {like_shape}")
    return mask


def choose_slice(gt, ai, mode):
    if mode == "middle":
        return gt.shape[2] // 2
    union = gt | ai
    if union.any():
        return int(np.argmax(union.sum(axis=(0, 1))))
    return gt.shape[2] // 2


def crop_bounds(gt_slice, ai_slice, pad):
    union = gt_slice | ai_slice
    height, width = union.shape
    ys, xs = np.where(union)
    if len(xs) == 0:
        return 0, height, 0, width

    y0, y1 = max(0, ys.min() - pad), min(height, ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(width, xs.max() + pad + 1)
    side = max(y1 - y0, x1 - x0)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    y0 = max(0, min(cy - side // 2, height - side))
    x0 = max(0, min(cx - side // 2, width - side))
    y1 = min(height, y0 + side)
    x1 = min(width, x0 + side)
    return y0, y1, x0, x1


def percentile_window(image, low, high):
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(finite, [low, high])
    if vmin == vmax:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def draw_overlay(args):
    image, image_nii = load_image(args.image)
    gt = load_mask(args.gt, image.shape)
    ai = load_mask(args.ai, image.shape)

    z = args.slice if args.slice is not None else choose_slice(gt, ai, args.slice_mode)
    if z < 0 or z >= image.shape[2]:
        raise ValueError(f"--slice must be in [0, {image.shape[2] - 1}], got {z}")

    # Transpose to display axial slices with the same orientation convention used
    # in the project figures.
    img_slice = image[:, :, z].T
    gt_slice = gt[:, :, z].T
    ai_slice = ai[:, :, z].T
    y0, y1, x0, x1 = crop_bounds(gt_slice, ai_slice, args.pad)

    vmin, vmax = percentile_window(image, args.window_low, args.window_high)
    fig, ax = plt.subplots(figsize=(args.figsize, args.figsize), constrained_layout=True)
    ax.imshow(
        img_slice[y0:y1, x0:x1],
        cmap="gray",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
    )
    if gt_slice[y0:y1, x0:x1].any():
        ax.contour(
            gt_slice[y0:y1, x0:x1],
            levels=[0.5],
            colors=[GT_COLOR],
            linewidths=args.gt_width,
            linestyles=[args.gt_linestyle],
        )
    if ai_slice[y0:y1, x0:x1].any():
        ax.contour(
            ai_slice[y0:y1, x0:x1],
            levels=[0.5],
            colors=[AI_COLOR],
            linewidths=args.ai_width,
            linestyles=[args.ai_linestyle],
        )

    if args.scale_bar_mm > 0:
        spacing = float(image_nii.header.get_zooms()[0])
        bar_px = args.scale_bar_mm / spacing
        ax.plot([5, 5 + bar_px], [5, 5], color="white", lw=3, solid_capstyle="butt")
        ax.text(
            5 + bar_px / 2,
            10,
            f"{args.scale_bar_mm / 10:g} cm",
            color="white",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    if args.title:
        ax.set_title(args.title)
    ax.set_xticks([])
    ax.set_yticks([])

    if not args.no_legend:
        handles = [
            Line2D([0], [0], color=GT_COLOR, lw=args.gt_width, ls=args.gt_linestyle),
            Line2D([0], [0], color=AI_COLOR, lw=args.ai_width, ls=args.ai_linestyle),
        ]
        ax.legend(handles, [args.gt_label, args.ai_label], loc="lower right", framealpha=0.85)

    out_path = args.out
    root, ext = osp.splitext(out_path)
    if ext.lower() != ".pdf":
        out_path = root + ".pdf"

    out_dir = osp.dirname(osp.abspath(out_path))
    if out_dir:
        import os

        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"saved {out_path} (slice z={z})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Path to the CT/MR NIfTI image.")
    parser.add_argument("--gt", required=True, help="Path to manual/ground-truth mask NIfTI.")
    parser.add_argument("--ai", required=True, help="Path to AI/prediction mask NIfTI.")
    parser.add_argument("--out", required=True, help="Output image path, e.g. overlay.pdf.")
    parser.add_argument("--slice", type=int, default=None, help="Axial slice index. Defaults to largest contour area.")
    parser.add_argument("--slice-mode", choices=["union", "middle"], default="union")
    parser.add_argument("--pad", type=int, default=25, help="Crop padding around contours, in pixels.")
    parser.add_argument("--window-low", type=float, default=2.0, help="Lower image display percentile.")
    parser.add_argument("--window-high", type=float, default=99.5, help="Upper image display percentile.")
    parser.add_argument("--scale-bar-mm", type=float, default=20.0, help="Scale bar length in mm; set 0 to hide.")
    parser.add_argument("--title", default="", help="Optional figure title.")
    parser.add_argument("--gt-label", default="Manual / GT")
    parser.add_argument("--ai-label", default="AI")
    parser.add_argument("--gt-width", type=float, default=3.0)
    parser.add_argument("--ai-width", type=float, default=2.2)
    parser.add_argument("--gt-linestyle", default="solid")
    parser.add_argument("--ai-linestyle", default="solid")
    parser.add_argument("--figsize", type=float, default=5.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--no-legend", action="store_true")
    draw_overlay(parser.parse_args())


if __name__ == "__main__":
    main()
