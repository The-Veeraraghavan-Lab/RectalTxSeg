"""
generate_representation_analysis_cka.py
──────────────────────────────────────────────────────────────────────────────
Pairwise cross-CKA analysis across SMIT / SwinUNETR model variants.

Stages and blocks are 0-indexed throughout (S0, S1, S2, S3 / B0, B1, ...).

MODES
─────
1. JSON config (default)
   python analysis/cka/generate_representation_analysis_cka.py --config analysis/cka/configs/my_experiment.json

2. Plot-only (reload saved .pt, regenerate figures)
   python analysis/cka/generate_representation_analysis_cka.py --plot-only path/to/pair_X_vs_Y.pt

JSON CONFIG SCHEMA
──────────────────
Top-level keys are global defaults; pair-level keys override them.

{
    "group_name":         "SMIT_vs_SSL",
    "split":              "validation",
    "out_channels":       2,
    "norm_name":          "instance",
    "num_workers":        4,
    "include_encdec":     false,
    "smit_ssl_path":      "pretrained_models/model_smit_ct10k.pth",
    "swin_pretrain_path": "pretrained_models/model_swinvit.pt",

    "pairs": [
        {
            "arch_a":       "smit",
            "run_a":        "rectal_smit_128x128x128_base",
            "name_a":       "SMIT-Base",
            "img_size_a":   [128, 128, 128],

            "arch_b":       "smit",
            "pretrained_b": true,
            "name_b":       "SSL",
            "img_size_b":   [128, 128, 128]
        },
        {
            "arch_a":       "smit",
            "run_a":        "rectal_smit_128x128x64_pretrained",
            "name_a":       "SMIT-ACT",
            "img_size_a":   [128, 128, 64],

            "arch_b":       "smit",
            "pretrained_b": true,
            "name_b":       "SSL",
            "img_size_b":   [128, 128, 64]
        }
    ]
}

OUTPUTS (per split)
───────────────────
  analysis/cka/results/{split}/
    pair_{name_a}_vs_{name_b}.pdf    heatmap stacked over drift  (supplemental)
    pair_{name_a}_vs_{name_b}.pt     saved matrices for --plot-only
    group_{group_name}_vs_{name_b}.pdf   grouped diagonal CKA lines (main paper)
                                         only generated when len(pairs) > 1
"""

import argparse
import json
import os
import re
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from monai import data, transforms
from monai.data import DataLoader, load_decathlon_datalist
from tqdm import tqdm

from models import smit, configs_smit
from models import swin_nvidia
from utils.monai_compat import channel_firstd


# =============================================================================
# CKA (linear kernel)
# =============================================================================

def _centering_matrix(n: int, device) -> torch.Tensor:
    return torch.eye(n, device=device) - torch.ones(n, n, device=device) / n


def _linear_hsic(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    n  = X.shape[0]
    H  = _centering_matrix(n, device=X.device)
    KX = X @ X.T
    KY = Y @ Y.T
    return torch.trace(KX @ H @ KY @ H) / ((n - 1) ** 2)


def linear_CKA(X: torch.Tensor, Y: torch.Tensor) -> float:
    X, Y    = X.float(), Y.float()
    hsic_xy = _linear_hsic(X, Y)
    hsic_xx = _linear_hsic(X, X)
    hsic_yy = _linear_hsic(Y, Y)
    return (hsic_xy / (torch.sqrt(hsic_xx * hsic_yy) + 1e-10)).item()


# =============================================================================
# Dataset / transforms
# =============================================================================

DATA_DIR   = "data_rectal"
OUTPUT_DIR = "analysis/cka/results"

DATALISTS = {
    "validation": (os.path.join(DATA_DIR, "Trainval_set1.json"), "validation"),
    "test":       (os.path.join(DATA_DIR, "Trainval_set1.json"), "testing"),
}

N_CASES = {
    "validation": 30,
    "test":       200,
}

N_CROPS         = 4
POOL_TARGET     = 8
SMIT_PATCH_SIZE = 2


def _strip_z800(filename: str) -> str:
    return filename.replace("_z800", "")


def _infer_label_from_image_path(image_path: str) -> str:
    p = image_path.replace("\\", "/")
    if "/imagesTs/" in p:
        p = p.replace("/imagesTs/", "/labelsTs/")
    elif "/imagesTr/" in p:
        p = p.replace("/imagesTr/", "/labelsTr/")
    else:
        head, tail = os.path.split(p)
        p = os.path.join(os.path.dirname(head), "labelsTs", tail).replace("\\", "/")

    d    = os.path.dirname(p)
    base = os.path.basename(p)
    if base.endswith(".nii.gz"):
        stem, ext = base[:-7], ".nii.gz"
    else:
        stem, ext = os.path.splitext(base)

    return os.path.join(d, _strip_z800(stem) + ext)


def load_datalist_with_optional_labels(
    json_path: str,
    split: str,
    data_root: str,
) -> List[dict]:
    files = load_decathlon_datalist(
        json_path,
        is_segmentation=True,
        data_list_key=split,
        base_dir=data_root,
    )
    fixed = []
    for item in files:
        it  = dict(item)
        img = it["image"]
        if isinstance(img, (list, tuple)):
            img = img[0]
        it["image"] = img

        lbl = it.get("label")
        if lbl is not None:
            if isinstance(lbl, (list, tuple)):
                lbl = lbl[0]
            it["label"] = lbl
        else:
            inferred    = _infer_label_from_image_path(img)
            it["label"] = inferred if os.path.exists(inferred) else None

        fixed.append(it)
    return fixed


def build_transform(
    img_size: Tuple[int, int, int],
    symmetric: bool = False,
    skip_orientation: bool = False,
) -> transforms.Compose:
    H, W, D    = img_size
    pad_method = "symmetric" if symmetric else "end"
    base = [
        transforms.LoadImaged(keys=["image"]),
        channel_firstd(keys=["image"]),
    ]
    if not skip_orientation:
        base.append(transforms.Orientationd(keys=["image"], axcodes="RAS"))
    base += [
        transforms.Spacingd(
            keys=["image"],
            pixdim=(1.0, 1.0, 1.0),
            mode="bilinear",
        ),
        transforms.ScaleIntensityRanged(
            keys=["image"],
            a_min=0, a_max=800,
            b_min=0, b_max=1,
            clip=True,
        ),
        transforms.CropForegroundd(keys=["image"], source_key="image", allow_smaller=True),
        transforms.SpatialPadd(
            keys=["image"],
            spatial_size=(H, W, D),
            method=pad_method,
        ),
        transforms.RandSpatialCropSamplesd(
            keys=["image"],
            roi_size=(H, W, D),
            num_samples=N_CROPS,
            random_size=False,
        ),
        transforms.ToTensord(keys=["image"]),
    ]
    return transforms.Compose(base)


def build_loader(
    files: list,
    img_size: Tuple[int, int, int],
    indices: List[int],
    num_workers: int,
    symmetric: bool = False,
    skip_orientation: bool = False,
    seed: Optional[int] = None,
) -> DataLoader:
    subset    = [files[i] for i in indices]
    transform = build_transform(img_size, symmetric=symmetric, skip_orientation=skip_orientation)
    ds        = data.Dataset(data=subset, transform=transform)

    worker_init = None
    generator   = None
    if seed is not None:
        def worker_init(worker_id):
            np.random.seed(seed + worker_id)
            torch.manual_seed(seed + worker_id)
        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init,
        generator=generator,
    )


# =============================================================================
# Model loading
# =============================================================================

DEFAULT_SMIT_SSL      = "pretrained_models/model_smit_ct10k.pth"
DEFAULT_SWIN_PRETRAIN = "pretrained_models/model_swinvit.pt"


def _load_smit_pretrained_ssl(model: torch.nn.Module, ssl_path: str) -> None:
    model_dict      = torch.load(ssl_path, map_location="cpu", weights_only=False)
    pretrained_dict = model_dict["student"]
    for k in list(pretrained_dict.keys()):
        pretrained_dict[k.replace("module.backbone.", "")] = pretrained_dict.pop(k)
    missing, unexpected = model.load_state_dict(pretrained_dict, strict=False)
    print(f"  [SMIT] loaded SSL backbone from {ssl_path}")
    if missing:
        print(f"    Missing ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"    Unexpected ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")


def _load_finetuned_checkpoint(
    model: torch.nn.Module,
    ckpt_path: str,
    strict: bool = True,
) -> None:
    ckpt       = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt))
    model.load_state_dict(state_dict, strict=strict)
    print(f"  loaded finetuned weights from {ckpt_path}")


def build_model(
    arch: str,
    img_size: Tuple[int, int, int],
    out_channels: int,
    device: torch.device,
    *,
    run_name: Optional[str],
    ckpt_path: Optional[str],
    pretrained: bool,
    smit_ssl_path: str,
    swin_pretrain_path: str,
    norm_name: str = "instance",
) -> torch.nn.Module:
    arch = arch.lower()

    if arch == "smit":
        config = configs_smit.get_SMIT_128_bias_True()
        model  = smit.SMIT_3D_Seg(
            config,
            out_channels=out_channels,
            img_size=img_size,
            norm_name=norm_name,
        )
        if pretrained:
            _load_smit_pretrained_ssl(model, smit_ssl_path)
        else:
            if ckpt_path is None:
                if run_name is None:
                    raise ValueError("Provide run_a/run_b or ckpt_a/ckpt_b for finetuned SMIT")
                ckpt_path = os.path.join("runs", run_name, "model_final.pt")
            _load_finetuned_checkpoint(model, ckpt_path, strict=True)
        return model.eval().to(device)

    if arch == "swinunetr":
        model = swin_nvidia.SwinUNETR(
            img_size=img_size,
            in_channels=1,
            out_channels=out_channels,
            feature_size=48,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=False,
            norm_name=norm_name,
        )
        if pretrained:
            model_dict = torch.load(swin_pretrain_path, map_location="cpu", weights_only=False)
            # Normalise key namespace: ct10k weights nest backbone under 'module.swinViT.'
            # but load_from() expects 'module.' directly — strip the extra prefix if present.
            sd = model_dict["state_dict"]
            if any(k.startswith("module.swinViT.") for k in sd):
                model_dict["state_dict"] = {k.replace("module.swinViT.", "module."): v for k, v in sd.items()}
            model.load_from(model_dict)
            print(f"  [SwinUNETR] loaded pretrained weights from {swin_pretrain_path}")
        else:
            if ckpt_path is None:
                if run_name is None:
                    raise ValueError("Provide run_a/run_b or ckpt_a/ckpt_b for finetuned SwinUNETR")
                ckpt_path = os.path.join("runs", run_name, "model_final.pt")
            _load_finetuned_checkpoint(model, ckpt_path, strict=True)
        return model.eval().to(device)

    raise ValueError(f"Unknown arch: {arch}")


# =============================================================================
# Layer naming / ordering  (0-indexed)
# =============================================================================

def _layer_group(name: str) -> str:
    if name.startswith("t."):  return "transformer"
    if name.startswith("enc"): return "encoder"
    if name.startswith("dec"): return "decoder"
    if name == "out":          return "out"
    return "other"


def sort_layer_names(names: List[str]) -> List[str]:
    def _key(n: str):
        m = re.fullmatch(r"t\.s(\d+)\.b(\d+)", n)
        if m: return (0, int(m.group(1)), int(m.group(2)))
        m = re.fullmatch(r"enc(\d+)", n)
        if m: return (1, int(m.group(1)), 0)
        m = re.fullmatch(r"dec(\d+)", n)
        if m: return (2, int(m.group(1)), 0)
        if n == "out": return (3, 0, 0)
        return (9, 0, 0)
    return sorted(names, key=_key)


def display_name(name: str) -> str:
    m = re.fullmatch(r"t\.s(\d+)\.b(\d+)", name)
    if m: return f"S{m.group(1)}.B{m.group(2)}"
    m = re.fullmatch(r"enc(\d+)", name)
    if m: return f"Enc{m.group(1)}"
    m = re.fullmatch(r"dec(\d+)", name)
    if m: return f"Dec{m.group(1)}"
    if name == "out": return "Out"
    return name


# =============================================================================
# Feature extraction hooks  (0-indexed stages and blocks)
# =============================================================================

class FeatureExtractor:
    def __init__(
        self,
        model: torch.nn.Module,
        arch: str,
        include_encdec: bool,
    ):
        self.model          = model
        self.arch           = arch.lower()
        self.include_encdec = include_encdec
        self.features: Dict[str, torch.Tensor] = {}
        self._hooks         = []

    def _make_hook(self, name: str):
        def hook(module, inp, out):
            self.features[name] = out.detach().cpu()
        return hook

    def _hook_module(self, module: torch.nn.Module, name: str):
        self._hooks.append(module.register_forward_hook(self._make_hook(name)))

    def register_hooks(self):
        self.remove_hooks()
        self.features = {}

        if self.arch == "smit":
            for s_idx, layer in enumerate(self.model.transformer.layers, start=0):
                for b_idx, block in enumerate(layer.blocks, start=0):
                    self._hook_module(block, f"t.s{s_idx}.b{b_idx}")

            if self.include_encdec:
                for i in [1, 2, 3, 4, 5, 10]:
                    if hasattr(self.model, f"encoder{i}"):
                        self._hook_module(getattr(self.model, f"encoder{i}"), f"enc{i}")
                for i in [5, 4, 3, 2, 1]:
                    if hasattr(self.model, f"decoder{i}"):
                        self._hook_module(getattr(self.model, f"decoder{i}"), f"dec{i}")
            return

        if self.arch == "swinunetr":
            swin = self.model.swinViT
            for stage, attr in enumerate(["layers1", "layers2", "layers3", "layers4"], start=0):
                layer_list = getattr(swin, attr, None)
                if layer_list is None:
                    continue
                blocks = layer_list[0].blocks
                for b_idx, (_, block) in enumerate(blocks.named_children(), start=0):
                    self._hook_module(block, f"t.s{stage}.b{b_idx}")

            if self.include_encdec:
                for i in [1, 2, 3, 4, 5, 10]:
                    if hasattr(self.model, f"encoder{i}"):
                        self._hook_module(getattr(self.model, f"encoder{i}"), f"enc{i}")
                for i in [5, 4, 3, 2, 1]:
                    if hasattr(self.model, f"decoder{i}"):
                        self._hook_module(getattr(self.model, f"decoder{i}"), f"dec{i}")
            return

        raise ValueError(f"Unknown arch in extractor: {self.arch}")

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []


# =============================================================================
# Pooling to common representation  (stage is 0-indexed)
# =============================================================================

def _as_feature_4d(feat: torch.Tensor) -> Optional[torch.Tensor]:
    if feat.ndim != 4:
        return None
    if feat.shape[0] <= 4096 and feat.shape[1] > 1 and feat.shape[2] > 1 and feat.shape[3] > 1:
        return feat
    if feat.shape[-1] <= 4096 and feat.shape[0] > 1 and feat.shape[1] > 1 and feat.shape[2] > 1:
        return feat.permute(3, 0, 1, 2).contiguous()
    return None


def pool_feature_to_common(
    feat: torch.Tensor,
    layer_name: str,
    img_size: Tuple[int, int, int],
) -> torch.Tensor:
    """Returns (POOL_TARGET^3, C). Stage index is 0-based."""
    if feat.ndim == 2:
        T_tok, C = feat.shape

        m = re.fullmatch(r"t\.s(\d+)\.b(\d+)", layer_name)
        if m:
            stage     = int(m.group(1))           # 0-indexed
            ds_factor = SMIT_PATCH_SIZE * (2 ** stage)
            H, W, D   = (s // ds_factor for s in img_size)
            if H * W * D == T_tok:
                vol    = feat.view(H, W, D, C).permute(3, 0, 1, 2).unsqueeze(0)
                pooled = F.adaptive_avg_pool3d(vol, output_size=POOL_TARGET)
                return pooled.squeeze(0).flatten(1).T

        p3 = POOL_TARGET ** 3
        if T_tok < p3:
            feat2 = torch.cat([feat, feat.new_zeros(p3 - T_tok, C)], dim=0)
            return feat2.view(p3, C)
        idx = torch.linspace(0, T_tok - 1, p3).long()
        return feat[idx]

    f4 = _as_feature_4d(feat)
    if f4 is not None:
        pooled = F.adaptive_avg_pool3d(f4.unsqueeze(0), output_size=POOL_TARGET)
        return pooled.squeeze(0).flatten(1).T

    raise ValueError(f"Unsupported feature shape {tuple(feat.shape)} at {layer_name}")


def extract_features(
    model: torch.nn.Module,
    extractor: FeatureExtractor,
    loader: DataLoader,
    img_size: Tuple[int, int, int],
    layer_names: List[str],
    device: torch.device,
    desc: str,
) -> Dict[str, torch.Tensor]:
    collected: Dict[str, List[torch.Tensor]] = {ln: [] for ln in layer_names}

    for batch in tqdm(loader, desc=desc, leave=False):
        img_field = batch["image"]
        if isinstance(img_field, (list, tuple)):
            imgs = torch.stack(img_field, dim=1)
        else:
            imgs = img_field.unsqueeze(1)

        B, NC  = imgs.shape[0], imgs.shape[1]
        images = imgs.view(B * NC, *imgs.shape[2:]).to(device)

        extractor.register_hooks()
        with torch.no_grad():
            _ = model(images)
        feats = extractor.features
        extractor.remove_hooks()

        for ln in layer_names:
            if ln not in feats:
                raise KeyError(f"Missing hooked feature '{ln}'. Available: {list(feats.keys())[:10]}")

        for ln in layer_names:
            raw = feats[ln].float().cpu()
            for i in range(B * NC):
                pooled = pool_feature_to_common(raw[i], ln, img_size)
                collected[ln].append(pooled.flatten())

    return {ln: torch.stack(collected[ln], dim=0) for ln in layer_names}


# =============================================================================
# CKA matrices + metrics
# =============================================================================

def compute_cross_cka(
    feats_a: Dict[str, torch.Tensor],
    feats_b: Dict[str, torch.Tensor],
    layers_a: List[str],
    layers_b: List[str],
) -> np.ndarray:
    mat = np.zeros((len(layers_a), len(layers_b)), dtype=np.float32)
    for i, la in enumerate(layers_a):
        for j, lb in enumerate(layers_b):
            mat[i, j] = linear_CKA(feats_a[la], feats_b[lb])
    return mat


def compute_cka_metrics_rect(cka: np.ndarray, band: int = 1) -> Dict[str, float]:
    nr, nc    = cka.shape
    k         = min(nr, nc)
    diag      = np.array([cka[i, i] for i in range(k)], dtype=np.float32)
    diag_mean = float(np.mean(diag)) if k > 0 else float("nan")

    I, J      = np.indices((nr, nc))
    band_mask = (np.abs(I - J) <= band) & (I < k) & (J < k)
    off_mask  = ~band_mask

    band_mean    = float(np.mean(cka[band_mask])) if np.any(band_mask) else float("nan")
    off_mean     = float(np.mean(cka[off_mask]))  if np.any(off_mask)  else float("nan")
    row_max      = np.max(cka, axis=1) if nr > 0 and nc > 0 else np.array([], dtype=np.float32)
    mean_row_max = float(np.mean(row_max)) if row_max.size else float("nan")

    path       = np.argmax(cka, axis=1) if nr > 0 and nc > 0 else np.array([], dtype=np.int64)
    ideal      = np.clip(np.arange(nr), 0, max(nc - 1, 0))
    mean_shift = float(np.mean(np.abs(path - ideal))) if path.size else float("nan")

    return {
        "diag_mean":       diag_mean,
        "band_mean":       band_mean,
        "off_mean":        off_mean,
        "mean_row_max":    mean_row_max,
        "mean_shift":      mean_shift,
        "drift_diag":      float(1.0 - diag_mean)    if np.isfinite(diag_mean)    else float("nan"),
        "drift_bestmatch": float(1.0 - mean_row_max) if np.isfinite(mean_row_max) else float("nan"),
        "diag_contrast":   float(band_mean - off_mean)
                           if (np.isfinite(band_mean) and np.isfinite(off_mean)) else float("nan"),
    }


# =============================================================================
# Plotting
# =============================================================================

def _draw_boundaries(ax, layer_list, axis="x"):
    last_group = _layer_group(layer_list[0])
    for i, name in enumerate(layer_list):
        group = _layer_group(name)
        if group != last_group:
            if axis == "x":
                ax.axvline(i - 0.5, color="white", ls="--", alpha=0.5, lw=1)
            else:
                ax.axhline(i - 0.5, color="white", ls="--", alpha=0.5, lw=1)
            last_group = group


def plot_pair(
    cka: np.ndarray,
    layers_a: List[str],
    layers_b: List[str],
    name_a: str,
    name_b: str,
    out_path: str,
    metrics: Dict[str, float],
) -> None:
    """
    Supplemental figure: CKA heatmap (top) stacked over per-layer shift (bottom).
    """
    ylabels       = [display_name(x) for x in layers_a]
    xlabels       = [display_name(x) for x in layers_b]
    nr            = len(layers_a)
    shift_per_row = np.abs(np.argmax(cka, axis=1) - np.arange(nr))
    shift_labels  = [display_name(layers_a[i]) for i in range(nr)]

    fig = plt.figure(figsize=(10, 13))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.35)

    # heatmap
    ax_heat = fig.add_subplot(gs[0])
    im = ax_heat.imshow(cka, vmin=0, vmax=1, cmap="magma", aspect="auto", origin="lower")
    _draw_boundaries(ax_heat, layers_b, axis="x")
    _draw_boundaries(ax_heat, layers_a, axis="y")
    ax_heat.set_yticks(range(len(ylabels)))
    ax_heat.set_yticklabels(ylabels, fontsize=14)
    ax_heat.set_xticks(range(len(xlabels)))
    ax_heat.set_xticklabels(xlabels, rotation=45, fontsize=14)
    ax_heat.set_xlabel(name_b, fontsize=18)
    ax_heat.set_ylabel(name_a, fontsize=18)
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.03, pad=0.02)
    cbar.set_label("Feature similarity", fontsize=12)
    cbar.ax.tick_params(labelsize=12)

    # per-layer shift
    ax_shift = fig.add_subplot(gs[1])
    colors   = ["tomato" if s > 2 else "steelblue" for s in shift_per_row]
    ax_shift.bar(range(nr), shift_per_row, color=colors, alpha=0.85, width=0.7)
    ax_shift.axhline(
        metrics["mean_shift"],
        color="dimgray", lw=1.2, ls="--",
        label=f'mean shift = {metrics["mean_shift"]:.2f}',
    )
    ax_shift.set_xticks(range(nr))
    ax_shift.set_xticklabels(shift_labels, rotation=45, fontsize=16, ha='right')
    ax_shift.set_ylabel("|argmax \n− diagonal|", fontsize=18)
    ax_shift.tick_params(axis='y', labelsize=16)
    ax_shift.set_title("Per-layer shift (best-match drift)", fontsize=16)
    ax_shift.legend(fontsize=14, frameon=False)
    ax_shift.spines["top"].set_visible(False)
    ax_shift.spines["right"].set_visible(False)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(
        out_path,
        bbox_inches="tight",
        pad_inches=0.05,
        transparent=True,
        metadata={
            "Subject": (
                f"diag={metrics['diag_mean']:.3f}, off={metrics['off_mean']:.3f}, "
                f"shift={metrics['mean_shift']:.2f}, drift(best)={metrics['drift_bestmatch']:.3f}"
            ),
            "Title": f"{name_a} vs {name_b}",
        },
    )
    plt.close(fig)
    print(f"  Saved pair  -> {out_path}")


_PAL = sns.color_palette("colorblind")

# Colour lookup — keyed on name_a exactly as written in the JSON
COLOUR_MAP = {
    "SMIT-Base":            _PAL[4],
    "SMIT-TA":              _PAL[1],
    "SMIT-ACT":             _PAL[3],
    "SMIT-ACT Scratch":     _PAL[3],   # same hue as ACT, distinguished by linestyle
    "Swin UNETR-Base":      _PAL[7],
    "Swin UNETR-TA":        _PAL[9],
    "Swin UNETR-ACT":       _PAL[0],
    "Swin UNETR-ACT Scratch": _PAL[0], # same hue as ACT, distinguished by linestyle
    "Cluster-B":            _PAL[3],   # matches SMIT-ACT (the reference model)
    "Cluster-D":            _PAL[3],
}

# Linestyle lookup — solid for everything except scratch and cluster comparisons
LINESTYLE_MAP = {
    "SMIT-ACT Scratch":       "--",
    "Swin UNETR-ACT Scratch": "--",
    "Cluster-B":              "--",
    "Cluster-D":              "--",
}

# Marker lookup — shape distinguishes clusters; everything else gets circles
MARKER_MAP = {
    "Cluster-B": "o",
    "Cluster-D": "^",
}

_FALLBACK_PALETTE = plt.cm.get_cmap("tab10")


def plot_group(
    records: List[Dict],
    group_name: str,
    out_path: str,
) -> None:
    """
    Main-paper figure: grouped diagonal CKA — one line per pair, single axis.

    Visual encoding
    ───────────────
    Colour    → model identity  (COLOUR_MAP; falls back to tab10)
    Linestyle → solid = finetuned / dashed = scratch or cluster split
    Marker    → o = default / ^ = Cluster-D

    records: list of {name_a, name_b, cka, layers_a}
    """
    fig, ax = plt.subplots(figsize=(5, 4))

    # Build union x-axis across all records
    all_layers = []
    for rec in records:
        layers_a = rec["layers_a"]
        ki       = min(rec["cka"].shape[0], rec["cka"].shape[1])
        all_layers.extend(layers_a[:ki])

    # Deduplicate preserving sort order
    seen = set()
    union_layers = []
    for l in sort_layer_names(list(dict.fromkeys(all_layers))):
        if l not in seen:
            union_layers.append(l)
            seen.add(l)

    k       = len(union_layers)
    xlabels = [display_name(l) for l in union_layers]
    pos_map = {l: i for i, l in enumerate(union_layers)}

    for idx, rec in enumerate(records):
        cka_r    = rec["cka"]
        layers_a = rec["layers_a"]
        ki       = min(cka_r.shape[0], cka_r.shape[1])
        diag     = np.array([cka_r[i, i] for i in range(ki)], dtype=np.float32)

        xs = [pos_map[l] for l in layers_a[:ki]]
        ys = np.full(k, np.nan)
        for x, y in zip(xs, diag):
            ys[x] = y

        name_a = rec["name_a"]
        color  = COLOUR_MAP.get(name_a, _FALLBACK_PALETTE(idx))
        ls     = LINESTYLE_MAP.get(name_a, "-")
        marker = MARKER_MAP.get(name_a, "o")

        ax.plot(
            range(k),
            ys,
            linestyle=ls,
            marker=marker,
            color=color,
            lw=2,
            ms=5,
            label=f'{name_a} vs {rec["name_b"]}',
        )

        # grey connector only through NaN gaps
        finite_xs = [x for x, y in enumerate(ys) if np.isfinite(y)]
        finite_ys = [ys[x] for x in finite_xs]
        for i in range(len(finite_xs) - 1):
            if finite_xs[i + 1] - finite_xs[i] > 1:
                ax.plot(
                    [finite_xs[i], finite_xs[i + 1]],
                    [finite_ys[i], finite_ys[i + 1]],
                    linestyle=":",
                    color="lightgray",
                    lw=1.2,
                    zorder=0,
                )

    ax.set_xticks(range(k))
    ax.set_xticklabels(xlabels, rotation=90, fontsize=8)
    ax.set_ylabel("Feature similarity", fontsize=10)
    ax.set_ylim(0, 1)
    ax.axhline(1.0, color="lightgray", lw=0.8, ls="--")
    ax.legend(fontsize=10, frameon=False, loc="lower left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05, transparent=True)
    plt.close(fig)
    print(f"  Saved group -> {out_path}")

# =============================================================================
# Core pipeline — run one pair for one split
# =============================================================================

def run_pair(
    split: str,
    pair_cfg: dict,
    global_cfg: dict,
    files: List[dict],
    device: torch.device,
    out_dir: str,
) -> Dict:
    arch_a     = pair_cfg["arch_a"]
    arch_b     = pair_cfg["arch_b"]
    name_a     = pair_cfg["name_a"]
    name_b     = pair_cfg["name_b"]
    img_size_a = tuple(pair_cfg["img_size_a"])
    img_size_b = tuple(pair_cfg["img_size_b"])

    def _cfg(key, default):
        return pair_cfg.get(key, global_cfg.get(key, default))

    out_channels       = _cfg("out_channels",       2)
    norm_name          = _cfg("norm_name",           "instance")
    num_workers        = _cfg("num_workers",         4)
    include_encdec     = _cfg("include_encdec",      False)
    smit_ssl_path      = _cfg("smit_ssl_path",       DEFAULT_SMIT_SSL)
    swin_pretrain_path = _cfg("swin_pretrain_path",  DEFAULT_SWIN_PRETRAIN)

    print(f"\n[{split}] {name_a} vs {name_b}")

    model_a = build_model(
        arch_a, img_size_a, out_channels, device,
        run_name=pair_cfg.get("run_a"),
        ckpt_path=pair_cfg.get("ckpt_a"),
        pretrained=pair_cfg.get("pretrained_a", False),
        smit_ssl_path=smit_ssl_path,
        swin_pretrain_path=swin_pretrain_path,
        norm_name=norm_name,
    )
    model_b = build_model(
        arch_b, img_size_b, out_channels, device,
        run_name=pair_cfg.get("run_b"),
        ckpt_path=pair_cfg.get("ckpt_b"),
        pretrained=pair_cfg.get("pretrained_b", False),
        smit_ssl_path=smit_ssl_path,
        swin_pretrain_path=swin_pretrain_path,
        norm_name=norm_name,
    )

    extractor_a = FeatureExtractor(model_a, arch_a, include_encdec)
    extractor_b = FeatureExtractor(model_b, arch_b, include_encdec)

    with torch.no_grad():
        extractor_a.register_hooks()
        _ = model_a(torch.zeros(1, 1, *img_size_a, device=device))
        layers_a = sort_layer_names(list(extractor_a.features.keys()))
        extractor_a.remove_hooks()

        extractor_b.register_hooks()
        _ = model_b(torch.zeros(1, 1, *img_size_b, device=device))
        layers_b = sort_layer_names(list(extractor_b.features.keys()))
        extractor_b.remove_hooks()

    print(f"  Layers A ({name_a}): {len(layers_a)}")
    print(f"  Layers B ({name_b}): {len(layers_b)}")

    total   = len(files)
    n_cases = min(N_CASES[split], total)
    indices = list(np.linspace(0, total - 1, n_cases, dtype=int))

    skip_ori = (split == "test") and ("rectal" in DATA_DIR.lower())

    loader_a = build_loader(
        files, img_size_a, indices,
        num_workers=num_workers,
        skip_orientation=skip_ori,
        seed=42,
    )
    loader_b = build_loader(
        files, img_size_b, indices,
        num_workers=num_workers,
        skip_orientation=skip_ori,
        seed=42,
    )

    feats_a = extract_features(
        model_a, extractor_a, loader_a, img_size_a, layers_a, device,
        desc=f"A: {name_a}",
    )
    feats_b = extract_features(
        model_b, extractor_b, loader_b, img_size_b, layers_b, device,
        desc=f"B: {name_b}",
    )

    cka     = compute_cross_cka(feats_a, feats_b, layers_a, layers_b)
    metrics = compute_cka_metrics_rect(cka, band=1)

    print(
        f"  diag={metrics['diag_mean']:.3f}  "
        f"off={metrics['off_mean']:.3f}  "
        f"shift={metrics['mean_shift']:.2f}  "
        f"drift(best)={metrics['drift_bestmatch']:.3f}"
    )

    safe_a  = name_a.replace(" ", "_")
    safe_b  = name_b.replace(" ", "_")
    out_pdf = os.path.join(out_dir, f"pair_{safe_a}_vs_{safe_b}.pdf")
    out_pt  = os.path.join(out_dir, f"pair_{safe_a}_vs_{safe_b}.pt")

    plot_pair(cka, layers_a, layers_b, name_a, name_b, out_pdf, metrics)

    torch.save(
        {
            "cka":        cka,
            "metrics":    metrics,
            "name_a":     name_a,
            "name_b":     name_b,
            "layers_a":   layers_a,
            "layers_b":   layers_b,
            "img_size_a": img_size_a,
            "img_size_b": img_size_b,
            "split":      split,
        },
        out_pt,
    )
    print(f"  Saved .pt   -> {out_pt}")

    del model_a, model_b
    torch.cuda.empty_cache()

    return {
        "name_a":   name_a,
        "name_b":   name_b,
        "cka":      cka,
        "layers_a": layers_a,
        "metrics":  metrics,
    }


# =============================================================================
# Plot-only
# =============================================================================

def replot_from_saved(
    pt_path: str,
    out_dir: str,
    name_a_override: Optional[str] = None,
    name_b_override: Optional[str] = None,
) -> None:
    saved    = torch.load(pt_path, map_location="cpu", weights_only=False)
    cka      = saved["cka"]
    layers_a = saved["layers_a"]
    layers_b = saved["layers_b"]
    name_a   = name_a_override or saved["name_a"]
    name_b   = name_b_override or saved["name_b"]
    metrics  = compute_cka_metrics_rect(cka, band=1)

    os.makedirs(out_dir, exist_ok=True)
    safe_a  = name_a.replace(" ", "_")
    safe_b  = name_b.replace(" ", "_")
    out_pdf = os.path.join(out_dir, f"pair_{safe_a}_vs_{safe_b}.pdf")
    plot_pair(cka, layers_a, layers_b, name_a, name_b, out_pdf, metrics)

    print(
        f"  diag={metrics['diag_mean']:.3f}  "
        f"off={metrics['off_mean']:.3f}  "
        f"shift={metrics['mean_shift']:.2f}  "
        f"drift(best)={metrics['drift_bestmatch']:.3f}"
    )


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cross-CKA for SMIT / SwinUNETR",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="JSON",
        help="Path to JSON config file",
    )
    p.add_argument(
        "--plot-only",
        type=str,
        default=None,
        metavar="PT_FILE",
        help="Regenerate figures from a saved .pt file (no extraction)",
    )
    p.add_argument(
        "--name-a",
        type=str,
        default=None,
        help="Override name_a label (--plot-only only)",
    )
    p.add_argument(
        "--name-b",
        type=str,
        default=None,
        help="Override name_b label (--plot-only only)",
    )
    return p.parse_args()


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # plot-only mode
    if args.plot_only:
        out_dir = os.path.dirname(args.plot_only) or "."
        print(f"Plot-only: {args.plot_only}")
        replot_from_saved(
            args.plot_only,
            out_dir,
            name_a_override=args.name_a,
            name_b_override=args.name_b,
        )
        print("Done.")
        return

    # JSON config mode
    if args.config is None:
        raise ValueError("Provide --config <file.json> or --plot-only <file.pt>")

    with open(args.config) as f:
        cfg = json.load(f)

    pairs      = cfg["pairs"]
    group_name = cfg.get("group_name", "group")
    splits_req = cfg.get("split", "both")
    splits     = ["validation", "test"] if splits_req == "both" else [splits_req]

    split_files: Dict[str, list] = {}
    for split in splits:
        json_path, list_key = DATALISTS[split]
        split_files[split]  = load_datalist_with_optional_labels(json_path, list_key, DATA_DIR)

    for split in splits:
        out_dir = os.path.join(OUTPUT_DIR, split)
        os.makedirs(out_dir, exist_ok=True)

        records = []
        for pair_cfg in pairs:
            rec = run_pair(
                split=split,
                pair_cfg=pair_cfg,
                global_cfg=cfg,
                files=split_files[split],
                device=device,
                out_dir=out_dir,
            )
            records.append(rec)

        # grouped diagonal CKA plot — main paper, only when >1 pair
        if len(records) > 1:
            name_b    = records[0]["name_b"]
            safe_b    = name_b.replace(" ", "_")
            group_pdf = os.path.join(out_dir, f"group_{group_name}_vs_{safe_b}.pdf")
            plot_group(records, group_name, group_pdf)

    print("\nDone.")


if __name__ == "__main__":
    main()
