#!/usr/bin/env python
"""Merge per-node runtime benchmark CSVs into one summary CSV."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    os.chdir(ROOT)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csvs", nargs="*", default=[
        "analysis/outputs/runtime_benchmark/node1_base_runtime.csv",
        "analysis/outputs/runtime_benchmark/node2_lora_runtime.csv",
        "analysis/outputs/runtime_benchmark/node3_loraens_runtime.csv",
    ])
    p.add_argument("--output", default="analysis/outputs/runtime_benchmark/runtime_summary.csv")
    args = p.parse_args()

    frames = []
    missing = []
    for item in args.csvs:
        path = Path(item)
        if path.exists():
            frames.append(pd.read_csv(path))
        else:
            missing.append(str(path))
    if not frames:
        raise FileNotFoundError(f"No benchmark CSVs found. Missing: {missing}")

    out = pd.concat(frames, ignore_index=True)
    order = ["voxelfox", "effidec", "lora", "loraens"]
    out["_order"] = out["model_key"].map({k: i for i, k in enumerate(order)})
    out = out.sort_values(["_order", "model"]).drop(columns=["_order"])

    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False)

    cols = [
        "model", "avg_seconds_per_volume", "std_seconds_per_volume",
        "avg_peak_vram_mb", "max_peak_vram_mb", "patch_gflops",
        "sw_batch_size", "use_tta", "gpu_name", "n_timed_cases",
    ]
    print(out[cols].to_string(index=False))
    if missing:
        print("\nMissing inputs:")
        for path in missing:
            print(f"  {path}")
    print(f"\nWrote {dst}")


if __name__ == "__main__":
    main()
