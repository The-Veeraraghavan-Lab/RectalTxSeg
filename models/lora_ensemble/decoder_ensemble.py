#!/usr/bin/env python
"""
LoRA + decoder-ensemble upper-bound for 3D segmentation.

Motivation: the cheap implicit variant (EnsembleSeg: per-member attention adapters + a SHARED decoder)
collapses — the shared decoder homogenises the members' features before the output, so member disagreement
~0 and the ensemble adds no uncertainty over a single model.

This variant gives each member its OWN decoder while sharing one encoder with frozen base weights plus shared
trainable LoRA adapters. All members therefore receive identical encoder features; diversity comes only from
decoder initialization and subsequent training dynamics. Treat this as a decoder deep-ensemble on shared
features, not as paper-pure LoRA-Ensemble with per-member encoder deltas.

Parameter count (the point): because the encoder is a single shared module, `model.parameters()` (which
dedups shared params) counts it ONCE — total = encoder + N·decoder. For EffiDec3D (~22M total, the decoder
being the smaller share) even N=4 decoders stays well under the full-VoxelFox-SwinUNETR baseline (~73M).
So this is the *upper-bound* ensemble: still cheaper than the full model, and (hopefully) real uncertainty.

Cost note: forward runs the shared encoder path once per member. Base encoder weights are frozen, but the
shared LoRA adapters are trainable, so this is still a grad-tracked path during training. Train with the
same per-member loss as EnsembleSeg; reduce with softmax-then-mean at eval.
"""
from __future__ import annotations
import torch
from torch import nn, Tensor

from .inject import freeze_for_lora, wrap_window_attention


class DecoderEnsembleSeg(nn.Module):
    """N segmentation members sharing ONE encoder module; forward returns per-member logits [N, B, C, ...]."""

    def __init__(self, members, encoder_attr: str = "swinViT"):
        super().__init__()
        self.members = nn.ModuleList(members)
        self.n_members = len(members)
        self.encoder_attr = encoder_attr

    def forward(self, x: Tensor) -> Tensor:
        return torch.stack([m(x) for m in self.members], dim=0)   # [N, B, C, D, H, W]


def _assert_decoder_init_diversity(members, encoder_attr: str) -> None:
    """Fail if any cloned decoder is bit-identical to member 0 after excluding the shared encoder."""
    enc_prefix = f"{encoder_attr}."
    ref = {
        k: v.detach()
        for k, v in members[0].state_dict().items()
        if not k.startswith(enc_prefix) and torch.is_floating_point(v)
    }
    if not ref:
        raise RuntimeError(f"No floating-point decoder tensors found outside '{enc_prefix}' to check diversity.")
    for idx, member in enumerate(members[1:], start=1):
        cur = {
            k: v.detach()
            for k, v in member.state_dict().items()
            if not k.startswith(enc_prefix) and torch.is_floating_point(v)
        }
        common = [k for k in ref if k in cur and ref[k].shape == cur[k].shape]
        if not common:
            raise RuntimeError(f"No comparable decoder tensors found for member {idx}; check encoder_attr={encoder_attr}.")
        if not any(not torch.equal(ref[k], cur[k]) for k in common):
            raise RuntimeError(
                f"Decoder member {idx} initialized identically to member 0. With shared encoder features, "
                "identical decoders would receive identical gradients and the ensemble would collapse from step 0."
            )


def _assert_member_lora_diversity(members, encoder_attr: str) -> None:
    """Fail if explicit members have identical LoRA A matrices after independent injection."""
    refs = {
        k: v.detach()
        for k, v in getattr(members[0], encoder_attr).state_dict().items()
        if k.endswith("w_a.weight") and torch.is_floating_point(v)
    }
    if not refs:
        raise RuntimeError("No single-member LoRA w_a tensors found; did you inject plain LoRA before checking?")
    for idx, member in enumerate(members[1:], start=1):
        cur = {
            k: v.detach()
            for k, v in getattr(member, encoder_attr).state_dict().items()
            if k.endswith("w_a.weight") and torch.is_floating_point(v)
        }
        common = [k for k in refs if k in cur and refs[k].shape == cur[k].shape]
        if not common:
            raise RuntimeError(f"No comparable LoRA w_a tensors found for member {idx}.")
        if not any(not torch.equal(refs[k], cur[k]) for k in common):
            raise RuntimeError(
                f"LoRA member {idx} initialized with identical adapters to member 0. "
                "That would collapse the paper-style member diversity from step 0."
            )


def build_decoder_ensemble(base_model: nn.Module, clone_fn, n_members: int,
                           encoder_attr: str = "swinViT") -> DecoderEnsembleSeg:
    """
    base_model : already SSL-loaded + LoRA-injected + encoder-frozen (single/shared LoRA on the encoder).
    clone_fn   : a zero-arg callable returning a FRESH model of the same architecture (fresh, random decoder).
    Returns a DecoderEnsembleSeg whose members all SHARE base_model's encoder (so it is stored/counted once),
    each with its own independently-initialised decoder.
    """
    assert n_members >= 2, "decoder ensemble needs n_members >= 2"
    shared_enc = getattr(base_model, encoder_attr)
    members = [base_model]
    for _ in range(n_members - 1):
        m = clone_fn()                               # fresh model -> fresh (random) decoder weights
        setattr(m, encoder_attr, shared_enc)         # SHARE the frozen, LoRA-adapted encoder (stored once)
        members.append(m)
    _assert_decoder_init_diversity(members, encoder_attr)
    # Encoder is frozen by the caller (freeze_for_lora on base); sharing propagates that to every member.
    return DecoderEnsembleSeg(members, encoder_attr)


def build_lora_decoder_ensemble(base_model: nn.Module, clone_fn, n_members: int, rank: int,
                                encoder_attr: str = "swinViT") -> DecoderEnsembleSeg:
    """
    Paper-closest segmentation LoRA-Ensemble:
      - every member starts from the same pretrained frozen encoder weights,
      - every member gets independently initialized plain LoRA adapters,
      - every member keeps its own independently initialized decoder/head.

    This intentionally duplicates the frozen encoder per member. It is heavier than the implicit
    EnsembleLoRA path, but avoids the shared-decoder bottleneck and mirrors the reference code's
    per-member task head more faithfully for segmentation.
    """
    assert n_members >= 2, "LoRA decoder ensemble needs n_members >= 2"
    encoder_state = {
        k: v.detach().cpu().clone()
        for k, v in getattr(base_model, encoder_attr).state_dict().items()
    }
    members = []
    for idx in range(n_members):
        m = base_model if idx == 0 else clone_fn()
        if idx > 0:
            getattr(m, encoder_attr).load_state_dict(encoder_state, strict=True)
        wrap_window_attention(getattr(m, encoder_attr), rank=rank, n_members=1, single=True)
        freeze_for_lora(m, encoder_attr=encoder_attr)
        members.append(m)
    _assert_decoder_init_diversity(members, encoder_attr)
    _assert_member_lora_diversity(members, encoder_attr)
    return DecoderEnsembleSeg(members, encoder_attr)


def count_params_dedup(model: nn.Module) -> dict:
    """Trainable/total counts with shared params counted ONCE (parameters() dedups by default)."""
    seen = set(); total = 0; train = 0
    for p in model.parameters():                     # remove_duplicate=True by default -> shared enc counted once
        if id(p) in seen:
            continue
        seen.add(id(p)); total += p.numel(); train += p.numel() if p.requires_grad else 0
    return {"trainable": train, "total": total, "pct": 100.0 * train / max(total, 1)}
