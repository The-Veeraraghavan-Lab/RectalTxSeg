#!/usr/bin/env python
"""
LoRA + implicit LoRA-Ensemble core, adapted for 3D window-attention segmentation backbones.

Adapted from Muhlematter & Halbheer et al., "LoRA-Ensemble" (arXiv:2405.14438),
https://github.com/prs-eth/LoRA-Ensemble  (models/lora.py).

Key deviations from the reference implementation (deliberate, for our setting):
  1. BATCH-FIRST, MEMBER-OUTERMOST layout. The reference assumes a sequence-first ViT
     ([seq, batch*members, dim]) with members interleaved. Our WindowAttention is batch-first
     ([rows, tokens, dim]) and window-partitioning multiplies/permutes the leading (batch) dim,
     which would scramble an interleaved member index. We instead replicate the *input volume*
     with `x.repeat(N, ...)` so members form N contiguous OUTERMOST blocks; window partition
     preserves that outer grouping, so at every attention call member = leading-block index.
  2. SHARED FROZEN PROJECTION OUTSIDE vmap. We compute the frozen base projection `w(x)` ONCE
     on the full (N*B) batch and only vmap the low-rank delta `w_b(w_a(x))` per member. This is
     faster (base matmul not replicated N times) and, crucially, guarantees the frozen backbone
     weights are never turned into N trainable copies (a subtle bug if `w` is stacked into the
     vmapped member state).

Requires torch >= 2.0 (torch.func.stack_module_state / functional_call / torch.vmap).
"""
from __future__ import annotations
import enum
import copy
from typing import Dict

import torch
from torch import nn, Tensor, vmap
from torch.func import stack_module_state, functional_call


class InitWeight(enum.Enum):
    NORMAL = 0
    KAIMING_UNIFORM = 1
    XAVIER_UNIFORM = 2
    DEFAULT = 0


def _init_lora(w_a: nn.Linear, w_b: nn.Linear, init_type: InitWeight, settings: dict | None) -> None:
    """Standard LoRA init: A ~ small random, B = 0  (=> delta W = 0 at start; members diverge via training)."""
    if init_type in (InitWeight.NORMAL, InitWeight.DEFAULT):
        mean = 0.0 if settings is None else settings.get("mean", 0.0)
        std = 0.02 if settings is None else settings.get("std", 0.02)
        nn.init.normal_(w_a.weight, mean=mean, std=std)
    elif init_type == InitWeight.KAIMING_UNIFORM:
        from math import sqrt
        a_sq = 5 if settings is None else settings.get("a_squared", 5)
        nn.init.kaiming_uniform_(w_a.weight, a=sqrt(a_sq))
    elif init_type == InitWeight.XAVIER_UNIFORM:
        gain = 1.0 if settings is None else settings.get("gain", 1.0)
        nn.init.xavier_uniform_(w_a.weight, gain=gain)
    else:
        raise ValueError(f"Invalid init type {init_type}")
    nn.init.zeros_(w_b.weight)


class LoRA(nn.Module):
    """Single LoRA-adapted projection: y = w(x) + w_b(w_a(x)). `w` is the frozen original Linear."""

    def __init__(self, w: nn.Module, rank: int, in_dim: int, out_dim: int | None = None,
                 initialize: bool = True, init_type: InitWeight = InitWeight.DEFAULT,
                 init_settings: dict | None = None) -> None:
        super().__init__()
        self.rank = rank
        self.w = w
        out_dim = in_dim if out_dim is None else out_dim
        self.w_a = nn.Linear(in_dim, rank, bias=False)
        self.w_b = nn.Linear(rank, out_dim, bias=False)
        if initialize:
            _init_lora(self.w_a, self.w_b, init_type, init_settings)

    def forward(self, x: Tensor) -> Tensor:
        return self.w(x) + self.w_b(self.w_a(x))


class _Delta(nn.Module):
    """The per-member low-rank delta only (no frozen base). This is what gets vmapped."""

    def __init__(self, in_dim: int, rank: int, out_dim: int):
        super().__init__()
        self.w_a = nn.Linear(in_dim, rank, bias=False)
        self.w_b = nn.Linear(rank, out_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w_b(self.w_a(x))


def _assert_member_a_diversity(members: list[_Delta]) -> None:
    """Fail if any two ensemble LoRA members share identical A init; B is zero by design."""
    for i, member in enumerate(members):
        wi = member.w_a.weight.detach()
        for j in range(i):
            if torch.equal(wi, members[j].w_a.weight.detach()):
                raise RuntimeError(
                    f"LoRA ensemble members {j} and {i} have identical w_a initialization. "
                    "Because w_b starts at zero, identical w_a would give identical early gradients "
                    "and collapse those members from step 0."
                )


class EnsembleLoRA(nn.Module):
    """
    Implicit ensemble of N LoRA adapters over one frozen projection `w`.

    Layout contract (see module docstring): the incoming activation `x` has leading dim
    = N * B_eff, with the N members as OUTERMOST contiguous blocks. `w(x)` (frozen, shared) is
    computed once; the N low-rank deltas are applied per-member via a single vmapped pass.
    """

    def __init__(self, w: nn.Module, rank: int, in_dim: int, n_members: int,
                 out_dim: int | None = None, init_type: InitWeight = InitWeight.DEFAULT,
                 init_settings: dict | None = None, chunk_size: int | None = None) -> None:
        super().__init__()
        out_dim = in_dim if out_dim is None else out_dim
        self.n_members = n_members
        self.out_dim = out_dim
        self.chunk_size = chunk_size

        self.w = w  # frozen original projection (shared across members)
        for p in self.w.parameters():
            p.requires_grad_(False)

        # N independent low-rank deltas; initialise each separately for diversity.
        members = []
        for _ in range(n_members):
            m = _Delta(in_dim, rank, out_dim)
            _init_lora(m.w_a, m.w_b, init_type, init_settings)
            members.append(m)
        _assert_member_a_diversity(members)

        # Stack member params for vectorised (vmap) execution.
        params, buffers = stack_module_state(members)
        # Register stacked params so they are trainable leaves tracked by the optimiser/module.
        self._param_keys = list(params.keys())
        for k, v in params.items():
            self.register_parameter(k.replace(".", "__"), nn.Parameter(v))
        # Keep the stacked buffers OUT of module registration (plain dict; _Delta has none anyway).
        object.__setattr__(self, "_buffers_stacked", buffers)
        # A meta 'skeleton' delta for functional_call — structure only, no storage. Register it via
        # object.__setattr__ so its (meta) params are NOT in .parameters()/state_dict/DDP broadcast.
        skel = copy.deepcopy(members[0]).to("meta")
        for p in skel.parameters():
            p.requires_grad_(False)
        object.__setattr__(self, "_skeleton", skel)

    def _stacked_params(self) -> Dict[str, Tensor]:
        return {k: getattr(self, k.replace(".", "__")) for k in self._param_keys}

    def _call_member(self, params: Dict[str, Tensor], buffers: Dict[str, Tensor], x: Tensor) -> Tensor:
        return functional_call(self._skeleton, (params, buffers), (x,))

    def forward(self, x: Tensor) -> Tensor:
        n = self.n_members
        assert x.shape[0] % n == 0, f"leading dim {x.shape[0]} not divisible by n_members {n}"
        b_eff = x.shape[0] // n

        base = self.w(x)                                   # [N*B_eff, ..., out_dim], shared
        xm = x.view(n, b_eff, *x.shape[1:])                # [N, B_eff, ..., in_dim]
        delta = vmap(self._call_member, in_dims=(0, 0, 0), chunk_size=self.chunk_size)(
            self._stacked_params(), self._buffers_stacked, xm)   # [N, B_eff, ..., out_dim]
        delta = delta.reshape(base.shape)
        return base + delta
