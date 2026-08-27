#!/usr/bin/env python
"""Benchmark model inference time, peak VRAM, and patch-level FLOPs.

This script intentionally uses the same model construction and sliding-window
inference path as main_inference.py, but it does not write segmentations to disk.
The reported runtime excludes NIfTI loading and preprocessing; it measures the
preprocessed tensor once it is on the GPU.

Usage:
  python analysis/runtime/benchmark_inference.py --models voxelfox,effidec --max_cases 20
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
MPL_CACHE = ROOT / "analysis" / "local" / "runtime_benchmark" / "mpl_cache"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CACHE))

import torch
from monai import data
from monai.data import load_decathlon_datalist

from main_inference import create_transforms, load_model, run_inference_with_tta


PRESETS = {
    "voxelfox": {
        "label": "VoxelFox",
        "model_name": "swinv2",
        "pred_subdir": "voxelfox_swinunetr",
        "checkpoint": "runs/rectal_voxelfox_swinunetr_96x96x64_pretrained/model_final.pt",
        "roi": (96, 96, 64),
        "sw_batch_size": 8,
        "swin_use_v2": True,
    },
    "effidec": {
        "label": "VoxelFox-EffiDec",
        "model_name": "effidec3d",
        "pred_subdir": "effidec3d",
        "checkpoint": "runs/rectal_effidec3d_96x96x64_pretrained/model_final.pt",
        "roi": (96, 96, 64),
        "sw_batch_size": 8,
        "swin_use_v2": True,
        "n_decoder_channels": 48,
        "resolution_factor": 2,
        "head_upsample": "trilinear",
    },
    "lora": {
        "label": "LoRA",
        "model_name": "effidec3d",
        "pred_subdir": "effidec3d",
        "checkpoint": "runs/rectal_effidec3d_loraS_r4_bs2/model_final.pt",
        "roi": (96, 96, 64),
        "sw_batch_size": 8,
        "swin_use_v2": True,
        "n_decoder_channels": 48,
        "resolution_factor": 2,
        "head_upsample": "trilinear",
        "use_lora": True,
        "lora_rank": 4,
        "lora_members": 1,
    },
    "loraens": {
        "label": "LoRA-Ens",
        "model_name": "effidec3d",
        "pred_subdir": "effidec3d",
        "checkpoint": "runs/rectal_effidec3d_loraLDE4/model_final.pt",
        "roi": (96, 96, 64),
        "sw_batch_size": 2,
        "swin_use_v2": True,
        "n_decoder_channels": 48,
        "resolution_factor": 2,
        "head_upsample": "trilinear",
        "use_lora": True,
        "lora_rank": 4,
        "lora_members": 4,
        "lora_decoder_ensemble": True,
    },
}


def make_model_args(preset: dict, cli: argparse.Namespace) -> SimpleNamespace:
    rx, ry, rz = preset["roi"]
    sw_batch = cli.sw_batch_size if cli.sw_batch_size is not None else preset["sw_batch_size"]
    return SimpleNamespace(
        data_dir=cli.data_dir,
        json_list=cli.json_list,
        datasets=cli.dataset,
        results_dir="results",
        output_dir=None,
        pred_subdir=preset.get("pred_subdir"),
        pretrained_model_path=str(ROOT / preset["checkpoint"]),
        model_name=preset["model_name"],
        in_channels=1,
        out_channels=2,
        norm_name="instance",
        use_upernet=False,
        feature_size=48,
        dropout_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=False,
        spatial_dims=3,
        swin_depths=(2, 2, 2, 2),
        swin_num_heads=(3, 6, 12, 24),
        swin_use_v2=bool(preset.get("swin_use_v2", False)),
        use_lora=bool(preset.get("use_lora", False)),
        lora_rank=int(preset.get("lora_rank", 4)),
        lora_members=int(preset.get("lora_members", 1)),
        allow_roi_mismatch=False,
        decoder_ensemble=bool(preset.get("decoder_ensemble", False)),
        lora_decoder_ensemble=bool(preset.get("lora_decoder_ensemble", False)),
        n_decoder_channels=int(preset.get("n_decoder_channels", 48)),
        resolution_factor=int(preset.get("resolution_factor", 2)),
        head_upsample=str(preset.get("head_upsample", "trilinear")),
        a_min=cli.a_min,
        a_max=cli.a_max,
        b_min=0.0,
        b_max=1.0,
        space_x=1.0,
        space_y=1.0,
        space_z=1.0,
        roi_x=rx,
        roi_y=ry,
        roi_z=rz,
        infer_overlap=cli.overlap,
        sw_batch_size=sw_batch,
        use_tta=cli.use_tta,
        skip_orientation=False,
        postproc="standard",
        conf_threshold=0.70,
        save_probs=False,
        distributed=False,
        world_size=1,
        rank=0,
        local_rank=0,
        dist_url="env://",
        dist_backend="nccl",
    )


def count_params(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def make_patch_predictor(model: torch.nn.Module, args: SimpleNamespace):
    is_lora_ensemble = bool(getattr(args, "use_lora", False) and getattr(args, "lora_members", 1) > 1)
    if not is_lora_ensemble:
        return model
    from models.lora_ensemble import ensemble_reduce

    def predictor(x):
        return ensemble_reduce(model(x))

    return predictor


def estimate_patch_gflops(model: torch.nn.Module, args: SimpleNamespace, device: torch.device) -> tuple[Optional[float], str]:
    dummy = torch.zeros(1, 1, args.roi_x, args.roi_y, args.roi_z, device=device)
    predictor = make_patch_predictor(model, args)
    with torch.inference_mode():
        try:
            from fvcore.nn import FlopCountAnalysis

            flops = FlopCountAnalysis(predictor, dummy).total()
            return flops / 1e9, "fvcore"
        except Exception as fvcore_error:
            try:
                from torch.profiler import ProfilerActivity, profile

                activities = [ProfilerActivity.CPU]
                if device.type == "cuda":
                    activities.append(ProfilerActivity.CUDA)
                with profile(activities=activities, with_flops=True, record_shapes=False) as prof:
                    _ = predictor(dummy)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                flops = sum(evt.flops or 0 for evt in prof.key_averages())
                if flops > 0:
                    return flops / 1e9, "torch.profiler"
                return None, f"not_available ({type(fvcore_error).__name__}; profiler reported 0)"
            except Exception as profiler_error:
                return None, f"not_available ({type(fvcore_error).__name__}; {type(profiler_error).__name__})"


def load_cases(args: SimpleNamespace, cli: argparse.Namespace):
    datalist_json = Path(args.data_dir) / args.json_list
    files = load_decathlon_datalist(str(datalist_json), True, cli.dataset, base_dir=args.data_dir)
    if cli.max_cases is not None:
        files = files[: cli.max_cases]
    transform = create_transforms(args)
    ds = data.Dataset(files, transform=transform)
    return data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=cli.num_workers, pin_memory=True)


def benchmark_one(key: str, cli: argparse.Namespace, device: torch.device) -> dict:
    preset = PRESETS[key]
    args = make_model_args(preset, cli)
    ckpt = Path(args.pretrained_model_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint for {preset['label']}: {ckpt}")

    print(f"\n==== {preset['label']} ({key}) ====")
    print(f"checkpoint: {ckpt}")
    print(f"roi=({args.roi_x},{args.roi_y},{args.roi_z}) sw_batch={args.sw_batch_size} tta={args.use_tta}")

    model = load_model(args, device)
    total_params, trainable_params = count_params(model)
    gflops, flops_method = (None, "skipped")
    if not cli.skip_flops:
        gflops, flops_method = estimate_patch_gflops(model, args, device)

    loader = load_cases(args, cli)
    times = []
    peaks = []
    n_seen = 0

    with torch.inference_mode():
        for i, batch in enumerate(loader):
            x = batch["image"].to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            _ = run_inference_with_tta(model, x, args)
            if device.type == "cuda":
                torch.cuda.synchronize()
                peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            else:
                peak_mb = float("nan")
            elapsed = time.perf_counter() - start
            n_seen += 1
            if i >= cli.warmup_cases:
                times.append(elapsed)
                peaks.append(peak_mb)
            print(f"case {i + 1:03d}: {elapsed:.3f}s peak={peak_mb:.1f} MB")

    if not times:
        raise RuntimeError("No timed cases. Increase --max_cases or reduce --warmup_cases.")

    row = {
        "model_key": key,
        "model": preset["label"],
        "checkpoint": display_path(ckpt),
        "dataset": cli.dataset,
        "n_total_cases_loaded": n_seen,
        "n_timed_cases": len(times),
        "warmup_cases": cli.warmup_cases,
        "use_tta": int(args.use_tta),
        "roi": f"{args.roi_x}x{args.roi_y}x{args.roi_z}",
        "sw_batch_size": args.sw_batch_size,
        "overlap": args.infer_overlap,
        "total_params_m": total_params / 1e6,
        "trainable_params_m": trainable_params / 1e6,
        "avg_seconds_per_volume": statistics.mean(times),
        "std_seconds_per_volume": statistics.stdev(times) if len(times) > 1 else 0.0,
        "median_seconds_per_volume": statistics.median(times),
        "avg_peak_vram_mb": statistics.mean(peaks),
        "max_peak_vram_mb": max(peaks),
        "patch_gflops": gflops if gflops is not None else "",
        "flops_method": flops_method,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch_version": torch.__version__,
        "hostname": platform.node(),
    }
    print(json.dumps(row, indent=2))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="all",
                   help="Comma list from {voxelfox,effidec,lora,loraens}, or all.")
    p.add_argument("--data_dir", default="data_rectal")
    p.add_argument("--json_list", default="Trainval_set1.json")
    p.add_argument("--dataset", default="testing")
    p.add_argument("--max_cases", type=int, default=20)
    p.add_argument("--warmup_cases", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--sw_batch_size", type=int, default=None,
                   help="Override preset sliding-window batch size for all selected models.")
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--use_tta", action="store_true",
                   help="Match paper inference with two-view flip TTA. Omit for raw one-pass deployment timing.")
    p.add_argument("--a_min", type=float, default=0.0)
    p.add_argument("--a_max", type=float, default=800.0)
    p.add_argument("--skip_flops", action="store_true")
    p.add_argument("--output", default="analysis/outputs/runtime_benchmark/runtime_benchmark.csv")
    return p.parse_args()


def main() -> None:
    os.chdir(ROOT)
    cli = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for meaningful runtime/VRAM benchmarking.")
    device = torch.device("cuda:0")

    if cli.models == "all":
        keys = list(PRESETS)
    else:
        keys = [x.strip().lower() for x in cli.models.split(",") if x.strip()]
    unknown = [k for k in keys if k not in PRESETS]
    if unknown:
        raise ValueError(f"Unknown model keys: {unknown}. Available: {sorted(PRESETS)}")

    rows = [benchmark_one(k, cli, device) for k in keys]
    out = ROOT / cli.output
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
