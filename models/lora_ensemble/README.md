# LoRA-Ensemble for 3D rectal-tumour segmentation

Parameter-efficient fine-tuning (**plain LoRA**) + implicit deep ensembling (**LoRA-Ensemble**) for the
window-attention backbones (VoCo/SwinV2, SMIT, NVIDIA SwinUNETR). An efficient, uncertainty-aware
fine-tuning recipe whose ensemble spread can be used to flag likely segmentation failures.

Adapted from **LoRA-Ensemble** (Mühlematter, Halbheer et al., arXiv:2405.14438). The classification→3D-seg port
is our contribution. **Base model = the PRE-TRAINED encoder, kept FROZEN** (paper §1/§2); only the low-rank
adapters (+ the decoder/head) train. We do **not** start from the fine-tuned `model_final.pt`.

## Requires
`torch >= 2.0` (`torch.func.stack_module_state` / `functional_call` / `torch.vmap`); verified on torch 2.0.1.
Plus `monai`.

## Files
| File | What |
|---|---|
| `lora_core.py` | `LoRA` (single adapter) and `EnsembleLoRA` (N members, one vmapped pass). Batch-first, member-outermost; frozen base projection computed once outside vmap. |
| `inject.py` | `wrap_window_attention` (swap `qkv`/`proj` in every `WindowAttention`), `freeze_for_lora`, `count_params`, `EnsembleSeg` (replicate input → `[N,B,C,…]`), `ensemble_reduce` (→ mean prob for val). |
| `eval_uncertainty.py` | numpy-only metrics used at the eval step: Dice, case uncertainty, Spearman(unc,Dice), failure-flagging AUROC. |

## Integrated into training (already wired)
`main_training.py` + `trainer.py` now support LoRA directly — no separate script. What was added:
- **Injection** after the SSL encoder load, before optimizer/DDP: freeze the pretrained encoder, adapt
  `qkv`/`proj`, wrap as `EnsembleSeg` when `--lora_members > 1`. (`encoder_attr` = `transformer` for SMIT,
  else `swinViT`.)
- **Optimizer** now filters `requires_grad` (only adapters/decoder train).
- **Training loss** (`trainer.py`): per-member loss branch — `out [N,B,C,…]` → reshape `[N·B,C,…]`,
  `target.repeat(N)` → each member fits the target independently (NOT loss on the mean).
- **Validation**: `val_predictor` reduces members with `ensemble_reduce` (softmax-then-mean) so sliding-window
  + Dice work unchanged.

### Flags
`--use_lora` · `--lora_rank 4` · `--lora_members N` (`1` = plain LoRA / efficiency baseline; `>1` = LoRA-Ensemble).

Variant flags:
- default `--use_lora --lora_members N`: paper-style per-member LoRA adapters with the existing segmentation
  decoder/head shared across members.
- `--decoder_ensemble`: decoder-only upper-bound; shared single LoRA encoder plus N independent decoders.
- `--lora_decoder_ensemble`: closest segmentation port of the LoRA-Ensemble paper; N independent LoRA adapters
  plus N independent decoders/heads. This is the closest configuration for uncertainty comparisons.

### Run examples (from repo root, mirrors train.sh conventions)

> **The encoder checkpoint decides which model family you get.**
> `--ssl_weights_path voxelfox_swinvit.pt` produces the VoxelFox/SWIFT variants;
> `voco_b_swinvit.pt` produces the VoCo counterparts. Keep `--logdir` consistent
> with the checkpoint you pass — mixing them silently writes the wrong model into
> a run directory whose name implies the other family.

| Variant | Encoder checkpoint | `--logdir` convention |
|---|---|---|
| SWIFTe-LoRA | `voxelfox_swinvit.pt` | `rectal_effidec3d_loraS_r4_bs2` |
| SWIFTe-LDE4 | `voxelfox_swinvit.pt` | `rectal_effidec3d_loraLDE4` |
| VoCo + LoRA | `voco_b_swinvit.pt` | `rectal_effidec3d_voco_loraS_r4_bs2` |
| VoCo + LDE4 | `voco_b_swinvit.pt` | `rectal_effidec3d_voco_loraLDE4_bs2` |

Only the two `voxelfox` runs exist in the local `runs/` and `results/`. The
`*_voco_*` LoRA rows are the naming convention used on the training server and
referenced by the scripts in `analysis/uncertainty/`; those artifacts are not
mirrored locally.

**SWIFTe-LoRA** — plain LoRA on the EffiDec3D decoder, VoxelFox encoder frozen:
```
python main_training.py --logdir rectal_effidec3d_loraS_r4_bs2 \
  --model_name effidec3d --swin_use_v2 \
  --use_ssl_pretrained --ssl_weights_path voxelfox_swinvit.pt \
  --use_lora --lora_rank 4 --lora_members 1 \
  --data_dir data_rectal --json_list Trainval_set1.json --save_checkpoint \
  --out_channels 2 --norm_name instance --intensity_mode fixed --a_min 0 --a_max 800 \
  --roi_x 96 --roi_y 96 --roi_z 64 --feature_size 48 \
  --n_decoder_channels 48 --resolution_factor 2 --head_upsample trilinear \
  --use_adv_loader --use_tumor_aug --distributed
```

**SWIFTe-LDE4** — per-member adapters *and* per-member decoders (the closest
segmentation port of the LoRA-Ensemble paper). Same as above with
`--lora_members 4 --lora_decoder_ensemble` and a different `--logdir`:
```
python main_training.py --logdir rectal_effidec3d_loraLDE4 \
  --model_name effidec3d --swin_use_v2 \
  --use_ssl_pretrained --ssl_weights_path voxelfox_swinvit.pt \
  --use_lora --lora_rank 4 --lora_members 4 --lora_decoder_ensemble \
  --data_dir data_rectal --json_list Trainval_set1.json --save_checkpoint \
  --out_channels 2 --norm_name instance --intensity_mode fixed --a_min 0 --a_max 800 \
  --roi_x 96 --roi_y 96 --roi_z 64 --feature_size 48 \
  --n_decoder_channels 48 --resolution_factor 2 --head_upsample trilinear \
  --use_adv_loader --use_tumor_aug --distributed
```

Lower `--sw_batch_size` if you hit OOM — the `N·B` batch flows through the 3D
decoder. For the VoCo counterparts, swap `--ssl_weights_path` to
`voco_b_swinvit.pt` and use the `*_voco_*` logdir from the table above.

LoRA also works on the plain SwinV2 encoder (`--model_name swinv2`, dropping the
EffiDec3D decoder flags), but that configuration is not part of the SWIFT
progression.

**Guard:** `--use_lora` requires `--use_ssl_pretrained` (LoRA adapts a *pretrained* frozen encoder; freezing a
random one is almost never intended). Override deliberately with `--allow_random_lora`.

### Inference (must mirror training)
A LoRA checkpoint can't load into the plain base model — `main_inference.py` injects the same wrapping before
`load_state_dict`, so pass the **same** LoRA *and* decoder flags you trained with. Evaluating SWIFTe-LDE4:
```
torchrun --nproc_per_node=4 main_inference.py --distributed \
  --data_dir data_rectal --json_list Trainval_set1.json --datasets validation \
  --model_name effidec3d --swin_use_v2 \
  --n_decoder_channels 48 --resolution_factor 2 --head_upsample trilinear \
  --use_lora --lora_rank 4 --lora_members 4 --lora_decoder_ensemble \
  --pretrained_model_path runs/rectal_effidec3d_loraLDE4/model_final.pt \
  --out_channels 2 --norm_name instance --intensity_mode fixed --a_min 0 --a_max 800 \
  --roi_x 96 --roi_y 96 --roi_z 64 --sw_batch_size 8 --use_tta
```
For SWIFTe-LoRA use `--lora_members 1`, drop `--lora_decoder_ensemble`, and point at
`runs/rectal_effidec3d_loraS_r4_bs2`.

**You get a safety net here.** `_reconcile_ckpt_geometry` compares your eval flags against the
`args` stored in the checkpoint: architecture and LoRA mismatches print a loud `[load_model][WARN]`,
and an ROI mismatch **hard-fails** (it loads cleanly but silently corrupts results — the
`roi_z=96`-vs-`64` trap). Override deliberately with `--allow_roi_mismatch`. Checkpoints trained
before `trainer.py` began storing `args` are skipped with a warning, so for those the burden is
still on you to match the flags.

For `--lora_members > 1` the ensemble is reduced **softmax-then-mean** (returned as log-probs so the existing
argmax / `--save_probs` / `--conf_threshold` path stays correct). `--lora_members 1` loads like any model.

## Training rules baked in (from the LoRA-Ensemble paper)
- **Per-member loss, not loss-on-the-mean** (done in `trainer.py`) — averaging collapses diversity.
- **Averaging is eval-only** (softmax-then-mean via `ensemble_reduce`).
- **No α/r scaling** — folded into init (A ~ N(0,0.02), B = 0); diversity comes from each member's random A.
- **Adapt qkv + proj, all attention layers**; rank 4–8; AMP fine (use `--noamp` if `vmap`+autocast misbehaves).

## Watch when running
- **Memory** scales with `--lora_members` (inflated `N·B` batch through the 3D decoder) — drop `--sw_batch_size`.
- **Load report:** the SSL loader prints missing/unexpected key counts — a large count means an arg mismatch.
- **DDP** already uses `find_unused_parameters=True` (needed: frozen encoder params get no grad).
- **Diversity check:** once trained, feed per-case `{dice, uncertainty}` to `eval_uncertainty.summarize` — want
  `spearman_unc_dice ≪ 0` and `flagging_auroc ≫ 0.5`. If flat, raise rank / try Xavier init / per-member last
  decoder block.
