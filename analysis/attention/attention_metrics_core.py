"""
attention_metrics_core.py
==========================
Shared primitives for computing the full attention flow matrix F_l
and padding entropy fraction phi_l from cached post-softmax attention weights.

All functions are architecture-agnostic: they receive the attention tensor,
a real/padding mask, and window geometry parameters.

Definitions (matching paper exactly)
-------------------------------------
F_l = [[A^rr_l, A^rp_l],
        [A^pr_l, A^pp_l]]

A^qk_l = E_{i in Q_q}[ sum_{j in K_k} alpha_ij ]

SFR_l  = A^rr_l   (signal fidelity ratio)
A^rp_l = 1 - SFR_l  (by construction, since alpha sums to 1 over all keys)

phi_l  = E_i[H^rp_i] / E_i[H_i]
H^rp_i = -sum_{j in P} alpha_ij * log(alpha_ij + eps)   (entropy contribution, padding keys)
H_i    = -sum_j alpha_ij * log(alpha_ij + eps)           (total entropy)

Note on H^rp_i:
  When alpha_ij = 0 for j in P, the contribution is 0 (0*log(0) = 0 by convention).
  We add eps=1e-12 inside the log only to avoid log(0) when alpha_ij > 0 but tiny.
  We gate with alpha_ij itself so truly-zero entries contribute zero.
"""

import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-12   # numerical floor inside log, never adds to zero entries


# ──────────────────────────────────────────────────────────────────────────────
# Mask construction
# ──────────────────────────────────────────────────────────────────────────────

def build_token_mask_stage0(
    L_axial: int,
    roi_x: int = 128,
    roi_y: int = 128,
    roi_z: int = 128,
    patch_size: int = 2,
) -> torch.BoolTensor:
    """
    Build a boolean real-token mask at Stage 0.
    Shape: (tH, tW, tD)  True = real token.
    Padding is end-padding along the axial (z / D) dimension.
    Works for any roi_x, roi_y, roi_z and patch_size.
    """
    tH = roi_x // patch_size
    tW = roi_y // patch_size
    tD = roi_z // patch_size

    real_t_axial = int(np.ceil(L_axial / patch_size))
    real_t_axial = min(real_t_axial, tD)

    mask = torch.zeros(tH, tW, tD, dtype=torch.bool)
    mask[:, :, :real_t_axial] = True
    return mask


def propagate_mask_one_merge(mask: torch.BoolTensor) -> torch.BoolTensor:
    """
    Propagate a real/padding mask through one PatchMerging step.
    Uses max-pool: a merged token is REAL if ANY constituent is real.
    """
    H, W, D = mask.shape
    if H % 2: mask = torch.cat([mask, mask[-1:, :, :]], dim=0)
    if W % 2: mask = torch.cat([mask, mask[:, -1:, :]], dim=1)
    if D % 2: mask = torch.cat([mask, mask[:, :, -1:]], dim=2)
    m = mask.float().unsqueeze(0).unsqueeze(0)
    m = F.max_pool3d(m, kernel_size=2, stride=2)
    return m.squeeze(0).squeeze(0).bool()


def get_stage_masks(mask_s0: torch.BoolTensor, num_stages: int = 4) -> dict:
    """
    Return {stage_idx: mask_3d} for all stages, propagating through merges.
    """
    masks = {0: mask_s0}
    m = mask_s0.clone()
    for s in range(1, num_stages):
        m = propagate_mask_one_merge(m)
        masks[s] = m
    return masks


# ──────────────────────────────────────────────────────────────────────────────
# Window partitioning helper
# ──────────────────────────────────────────────────────────────────────────────

def partition_mask_into_windows(
    mask_3d: torch.BoolTensor,
    H: int, W: int, T: int,
    window_size: tuple,
    shift_size: tuple = None,
) -> torch.BoolTensor:
    """
    Pad mask_3d to multiples of window_size, apply cyclic shift if needed,
    then partition into windows.

    Returns mask_wins: (nW, N)  True = real token.
    nW = number of windows, N = tokens per window.
    """
    ws = window_size
    pH = int(np.ceil(H / ws[0])) * ws[0]
    pW = int(np.ceil(W / ws[1])) * ws[1]
    pT = int(np.ceil(T / ws[2])) * ws[2]

    mask_padded = torch.zeros(pH, pW, pT, dtype=torch.bool)
    mask_padded[:H, :W, :T] = mask_3d

    if shift_size is not None and min(shift_size) > 0:
        mask_padded = torch.roll(
            mask_padded,
            shifts=(-shift_size[0], -shift_size[1], -shift_size[2]),
            dims=(0, 1, 2),
        )

    nWh = pH // ws[0]
    nWw = pW // ws[1]
    nWt = pT // ws[2]
    N   = ws[0] * ws[1] * ws[2]

    mask_wins = mask_padded.view(nWh, ws[0], nWw, ws[1], nWt, ws[2])
    mask_wins = mask_wins.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, N)
    return mask_wins   # (nW, N)


# ──────────────────────────────────────────────────────────────────────────────
# Full flow matrix F_l
# ──────────────────────────────────────────────────────────────────────────────

def compute_flow_matrix(
    attn_nwhh: torch.Tensor,
    mask_3d: torch.BoolTensor,
    H: int, W: int, T: int,
    window_size: tuple,
    shift_size: tuple = None,
) -> dict:
    """
    Compute the full 2x2 attention flow matrix for one block.

    attn_nwhh : (nW, n_heads, N, N)  post-softmax, already on CPU or GPU
    mask_3d   : (H, W, T) bool — True = real token at this stage resolution
    H, W, T   : token grid dims at this stage
    window_size, shift_size : must match model's SwinTransformerBlock

    Returns dict with keys: Arr, Arp, Apr, App, SFR, nW_real, nW_pad_only
      Arr = A^rr  (real→real)
      Arp = A^rp  (real→pad)   = 1 - SFR
      Apr = A^pr  (pad→real)
      App = A^pp  (pad→pad)
      SFR = Arr   (alias)
      nW_real      : number of windows containing at least one real token
      nW_pad_only  : number of windows containing only padding tokens (excluded)

    Returns None if no real queries exist (degenerate case).
    """
    device = attn_nwhh.device
    mask_wins = partition_mask_into_windows(
        mask_3d, H, W, T, window_size, shift_size
    ).to(device)   # (nW, N)

    real_mask = mask_wins.float()        # (nW, N)
    pad_mask  = (~mask_wins).float()     # (nW, N)

    # windows with zero real tokens: excluded from real-query metrics
    has_real = mask_wins.any(dim=1)      # (nW,)
    has_pad  = (~mask_wins).any(dim=1)   # (nW,)
    nW_real     = has_real.sum().item()
    nW_pad_only = (~has_real).sum().item()

    total_real_queries = real_mask.sum().item()
    total_pad_queries  = pad_mask.sum().item()
    if total_real_queries == 0:
        return None

    # average over heads: (nW, N, N)
    attn_avg = attn_nwhh.mean(dim=1).clamp(min=0.0)   # match compute_phi

    # ── real-query rows ────────────────────────────────────────────────────
    # exclude pad-only windows (mirrors compute_phi behaviour)
    has_real = mask_wins.any(dim=1)          # (nW,)
    real_mask_filtered = real_mask * has_real.float().unsqueeze(1)  # (nW, N)
    pad_mask_filtered  = pad_mask  * has_real.float().unsqueeze(1)  # (nW, N)

    real_q = real_mask.unsqueeze(2)   # (nW, Nq, 1)
    pad_k  = pad_mask.unsqueeze(1)    # (nW, 1,  Nk)
    real_k = real_mask.unsqueeze(1)   # (nW, 1,  Nk)

    Arr = ((attn_avg * real_q * real_k).sum() / real_mask_filtered.sum()).item()
    Arp = ((attn_avg * real_q * pad_k ).sum() / real_mask_filtered.sum()).item()

    # ── pad-query rows ─────────────────────────────────────────────────────
    if total_pad_queries > 0:
        pad_q = pad_mask.unsqueeze(2)   # (nW, Nq, 1)
        Apr = ((attn_avg * pad_q * real_k).sum() / pad_mask.sum()).item()
        App = ((attn_avg * pad_q * pad_k ).sum() / pad_mask.sum()).item()
    else:
        Apr = float('nan')
        App = float('nan')

    return dict(
        Arr=Arr, Arp=Arp, Apr=Apr, App=App,
        SFR=Arr,
        nW_real=nW_real, nW_pad_only=nW_pad_only,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Padding entropy fraction phi_l
# ──────────────────────────────────────────────────────────────────────────────

def compute_phi(
    attn_nwhh: torch.Tensor,
    mask_3d: torch.BoolTensor,
    H: int, W: int, T: int,
    window_size: tuple,
    shift_size: tuple = None,
) -> dict:
    """
    Compute phi_l = E_i[H^rp_i] / E_i[H_i] over real query tokens.

    H^rp_i = -sum_{j in P} alpha_ij * log(alpha_ij + eps)   [entropy contribution, pad keys]
    H_i    = -sum_j alpha_ij * log(alpha_ij + eps)            [total entropy]

    Windows with ONLY padding tokens are excluded (H_i undefined).
    The 0*log(0)=0 convention is enforced by gating with alpha itself.

    Returns dict: phi, mean_H_total, mean_H_rp, mean_H_rr
    Returns None if no real queries.
    """
    device = attn_nwhh.device
    mask_wins = partition_mask_into_windows(
        mask_3d, H, W, T, window_size, shift_size
    ).to(device)   # (nW, N)

    real_mask = mask_wins.float()
    pad_mask  = (~mask_wins).float()

    total_real_q = real_mask.sum().item()
    if total_real_q == 0:
        return None

    # Average over heads first: (nW, N, N)
    attn_avg = attn_nwhh.mean(dim=1).clamp(min=0.0)   # clamp for numerical safety

    # entropy per token: (nW, N)  — using gated convention: a*log(a+eps) → 0 when a=0
    log_attn = torch.log(attn_avg + EPS)                      # (nW, Nq, Nk)
    entropy_full = -(attn_avg * log_attn).sum(dim=2)           # (nW, Nq)  H_i

    # entropy from padding keys only
    pad_k = pad_mask.unsqueeze(1)                              # (nW, 1, Nk)
    attn_pad_keys = attn_avg * pad_k                           # zero out real keys
    log_attn_pad  = torch.log(attn_pad_keys + EPS)
    # gate: only count where attn_pad_keys > 0 (enforces 0*log(0)=0)
    h_rp = -(attn_pad_keys * log_attn_pad * (attn_pad_keys > 0).float()).sum(dim=2)  # (nW, Nq)

    # gate to real queries only
    real_q_mask = real_mask   # (nW, N)
    real_q_2d   = real_q_mask.unsqueeze(2).squeeze(2)  # same as real_mask

    H_total_sum = (entropy_full * real_q_mask).sum().item()
    H_rp_sum    = (h_rp        * real_q_mask).sum().item()
    n_real_q    = real_q_mask.sum().item()

    mean_H_total = H_total_sum / n_real_q
    mean_H_rp    = H_rp_sum    / n_real_q
    mean_H_rr    = mean_H_total - mean_H_rp

    phi = mean_H_rp / mean_H_total if mean_H_total > 0 else float('nan')

    return dict(phi=phi, mean_H_total=mean_H_total,
                mean_H_rp=mean_H_rp, mean_H_rr=mean_H_rr)


# ──────────────────────────────────────────────────────────────────────────────
# Per-case aggregation across blocks within a stage
# ──────────────────────────────────────────────────────────────────────────────

def aggregate_stage_metrics(block_results: list) -> dict:
    """
    Average flow matrix entries and phi across blocks within a stage.
    block_results: list of dicts, one per block (from compute_flow_matrix / compute_phi).
    Ignores None entries (blocks with no real queries).
    """
    valid = [r for r in block_results if r is not None]
    if not valid:
        return {k: float('nan') for k in ['Arr','Arp','Apr','App','SFR','phi',
                                           'mean_H_total','mean_H_rp','mean_H_rr']}
    out = {}
    for key in ['Arr','Arp','Apr','App','SFR','phi','mean_H_total','mean_H_rp','mean_H_rr']:
        vals = [v[key] for v in valid if key in v and not np.isnan(v[key])]
        out[key] = float(np.mean(vals)) if vals else float('nan')
    return out
