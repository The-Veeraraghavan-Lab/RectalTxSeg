#!/usr/bin/env python
"""
Inject EnsembleLoRA into the WindowAttention blocks of a 3D Swin/SMIT segmentation backbone,
freeze the SSL-pretrained encoder, and wrap the whole model as an implicit deep ensemble.

Works for any backbone whose attention module is named `WindowAttention` and exposes
`.qkv` (Linear: dim -> 3*dim) and `.proj` (Linear: dim -> dim):
  - swinv2 / MONAI SwinUNETR-V2 (VoxelFox / VoCo): model.swinViT.* WindowAttention
  - EffiDec3D (separate config: same encoder, reduced decoder): model.swinViT.* WindowAttention
  - SMIT: model.transformer.* WindowAttention   (encoder_attr='transformer')
  - SwinUNETR: model.swinViT.* WindowAttention

Design (see lora_core.py): freeze the pretrained encoder, train only the per-member low-rank
adapters on qkv/proj; the conv skip-encoders + decoder + output head stay trainable and SHARED
across members (a shared decoder falls out for free from replicating the input batch).
"""
from __future__ import annotations
from typing import Sequence
import torch
from torch import nn, Tensor

from .lora_core import EnsembleLoRA, LoRA, InitWeight


# Adapter params are named 'w_a'/'w_b' (single LoRA) or 'w_a__'/'w_b__' (stacked
# EnsembleLoRA). The meta skeleton mirrors those names but is only structure for
# functional_call, so keep it frozen and out of optimizer param groups.
def is_lora_adapter_param(name: str) -> bool:
    if "._skeleton." in name:
        return False
    return any(t in name for t in ("w_a.", "w_b.", "w_a__", "w_b__"))


# ----------------------------------------------------------------------------- injection

def wrap_window_attention(root: nn.Module, rank: int, n_members: int,
                          targets: Sequence[str] = ("qkv", "proj"),
                          init_type: InitWeight = InitWeight.DEFAULT,
                          chunk_size: int | None = None,
                          single: bool = False) -> int:
    """
    Replace the `targets` Linear projections inside every WindowAttention under `root` with
    EnsembleLoRA (n_members) — or plain LoRA if single=True (for the param-efficiency baseline).

    Returns the number of attention modules adapted. Raises if none found (fail-loud).
    """
    n_wrapped = 0
    for m in root.modules():
        if type(m).__name__ != "WindowAttention":
            continue
        for t in targets:
            proj = getattr(m, t, None)
            if not isinstance(proj, nn.Linear):
                raise TypeError(f"WindowAttention.{t} is {type(proj).__name__}, expected nn.Linear")
            in_f, out_f = proj.in_features, proj.out_features
            if single:
                new = LoRA(proj, rank=rank, in_dim=in_f, out_dim=out_f, init_type=init_type)
            else:
                new = EnsembleLoRA(proj, rank=rank, in_dim=in_f, n_members=n_members,
                                   out_dim=out_f, init_type=init_type, chunk_size=chunk_size)
            setattr(m, t, new)
        n_wrapped += 1
    if n_wrapped == 0:
        raise RuntimeError("No WindowAttention modules found under root — check encoder_attr.")
    return n_wrapped


def freeze_for_lora(model: nn.Module, encoder_attr: str = "swinViT") -> None:
    """
    Freeze the pretrained encoder except its injected LoRA deltas; keep everything else
    (conv skip-encoders, decoder, output head) trainable and shared across members.

    LoRA delta parameters are registered with '__' in their names (see EnsembleLoRA); the frozen
    base projection is '<...>.w.weight'.
    """
    enc_prefix = encoder_attr + "."
    for name, p in model.named_parameters():
        in_encoder = name.startswith(enc_prefix) or ("." + enc_prefix) in name
        if in_encoder:
            p.requires_grad_(is_lora_adapter_param(name))
        else:
            p.requires_grad_(True)


def count_params(model: nn.Module) -> dict:
    """Trainable vs total parameter counts (for the efficiency table)."""
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"trainable": train, "total": total, "pct": 100.0 * train / max(total, 1)}


# ----------------------------------------------------------------------------- ensemble wrapper

class EnsembleSeg(nn.Module):
    """
    Wrap a LoRA-injected segmentation model as an N-member implicit ensemble.

    forward(x) replicates the input volume into N OUTERMOST blocks (member-outermost layout the
    EnsembleLoRA layers expect), runs one forward, and returns per-member logits shaped
    [N, B, C, D, H, W].  Reduce with `ensemble_reduce` for validation/inference.
    """

    def __init__(self, model: nn.Module, n_members: int):
        super().__init__()
        self.model = model
        self.n_members = n_members

    def forward(self, x: Tensor) -> Tensor:
        n, b = self.n_members, x.shape[0]
        xr = x.repeat(n, *([1] * (x.dim() - 1)))          # [N*B, C, D, H, W], member-outermost
        out = self.model(xr)                              # [N*B, out_ch, D, H, W]
        return out.view(n, b, *out.shape[1:])             # [N, B, out_ch, D, H, W]


def ensemble_reduce(out: Tensor) -> Tensor:
    """[N, B, C, ...] per-member logits -> [B, C, ...] mean class-probabilities (validation / inference)."""
    return torch.softmax(out, dim=2).mean(0)
