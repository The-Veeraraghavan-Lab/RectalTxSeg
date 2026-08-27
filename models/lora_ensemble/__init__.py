"""
LoRA + implicit LoRA-Ensemble for 3D window-attention segmentation backbones.

Adapted from LoRA-Ensemble (arXiv:2405.14438) for EffiDec3D / SMIT / SwinUNETR.
See README.md for the integration recipe and smoke-test protocol.
"""
from .lora_core import LoRA, EnsembleLoRA, InitWeight
from .inject import (
    wrap_window_attention,
    freeze_for_lora,
    count_params,
    EnsembleSeg,
    ensemble_reduce,
)

__all__ = [
    "LoRA", "EnsembleLoRA", "InitWeight",
    "wrap_window_attention", "freeze_for_lora", "count_params",
    "EnsembleSeg", "ensemble_reduce",
]
