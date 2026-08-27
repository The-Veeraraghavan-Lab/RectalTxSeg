# AGENTS.md — working context for coding agents

3D rectal tumor segmentation on pelvic MRI with transformer backbones.
Read `README.md` first for the model naming and checkpoint layout; this file
covers the operational detail an agent needs to make correct changes.

## Ground rules

- Run everything from the **repository root**. Analysis scripts assume it.
- Install PyTorch first (matching your CUDA), then `pip install -r requirements.txt`.
  Surface Dice comes from `pip install git+https://github.com/google-deepmind/surface-distance.git`.
- No patient data, checkpoints, or derived tables belong in git. `.gitignore` is
  allowlist-based for `scripts/`, `analysis/features|attention|preprocessing|tumor/`,
  markdown, and JSON — new files in those areas are private by default. If you add
  something that should ship, add an explicit `!` rule.
- `train.sh` overwrites an existing run directory without warning. Use
  `RUN_OVR=<name>` to protect prior results.

## Entry points

| File | Role |
|---|---|
| `main_training.py` | Distributed training. Builds the model, loads SSL weights, injects LoRA, then hands off to `trainer.py`. |
| `trainer.py` | Training loop, validation, checkpointing. Writes `{"epoch","best_acc","state_dict","args"}`. |
| `main_inference.py` | Sliding-window inference, optional TTA, probability export. `load_model()` is the canonical checkpoint-loading path. |
| `train.sh` / `eval.sh` | Preset launchers. Identical preset names on both sides, so a model trained by one is evaluated by the other with no extra flags. |

## Architecture keys

`--model_name` selects the network only; pretraining is chosen by
`--ssl_weights_path`.

| Key | Network |
|---|---|
| `smit` | SMIT encoder + UNETR-style decoder |
| `swinunetr` | MONAI/NVIDIA SwinUNETR |
| `swinv2` | MONAI SwinUNETR-V2 (needs `--swin_use_v2`) |
| `effidec3d` | SwinV2 encoder + EffiDec3D decoder (needs `--swin_use_v2`) |

`voco_swinunetr` is a legacy alias for `swinv2`.

## Things that will bite you

**Checkpoint `args` come in two shapes.** Older checkpoints stored the raw
`argparse.Namespace`; current ones store `vars(args)`. Any code reading
`checkpoint["args"]` must normalise:

```python
targs = ckpt.get("args")
if targs is not None and not isinstance(targs, dict):
    targs = getattr(targs, "__dict__", None)
```

Most existing checkpoints store no `args` at all, so the geometry guard in
`_reconcile_ckpt_geometry` silently no-ops on them. Do not assume it protects you.

**ROI must match training.** A wrong `--roi_z` loads cleanly and silently
corrupts results. `main_inference.py` hard-fails on a mismatch when the
checkpoint has stored args; otherwise it is on you. `eval.sh` presets encode the
correct ROI per model.

**LoRA wrapping must be identical at train and eval time.** A LoRA checkpoint
will not load into a plain model. Pass the same
`--use_lora --lora_rank --lora_members [--lora_decoder_ensemble]` plus the same
decoder flags. See `models/lora_ensemble/README.md`.

**SwinUNETR key namespaces differ by checkpoint.** `main_training.py` remaps
`module.swinViT.*` → `module.*` and `mlp.linear1/2` → `mlp.fc1/fc2` before
`load_from()`. It is a no-op for NVIDIA's `model_swinvit.pt` but required for
CT10K-style checkpoints. Do not remove it.

**There are several independent model-loading paths.** `main_inference.load_model`
is one; `analysis/cka/generate_representation_analysis_cka.py` builds models
itself; the attention extractors and t-SNE script each have their own. A change
to model construction must be checked against all of them, not just inference.

The SwinV2 and EffiDec3D builders in `models/swin_voco_utils.py` read attributes
straight off `args` — `dropout_rate`, `attn_drop_rate`, `dropout_path_rate`,
`use_checkpoint`, `spatial_dims`, `swin_depths`, `swin_num_heads`. Any code that
constructs an `args` object by hand, rather than via `main_training.py`'s parser,
must supply all of them or model construction fails before weights are touched.

**Intensity mode must match between training and inference.** `fixed` for images
already scaled to `[0, 800]` (the `_z800` convention), `percentile` for raw
NIfTI. Mixing them invalidates comparisons.

## Data layout

```
data_rectal/
  Trainval_set1.json        # MONAI Decathlon-style manifest
  imagesTr/  labelsTr/      # training + validation
  imagesTs/  labelsTs/      # testing
```

Spacing 1.0 mm isotropic, RAS orientation, two output channels
(background, tumor).

## Checkpoint conventions

- `runs/<name>/model_final.pt` — the checkpoint to use
- `runs/<name>/events.out.tfevents.*` — TensorBoard scalars (`train_loss` per
  epoch, `val_acc` per `val_every`)
- `_pretrained` means SSL-initialized, `_scratch` means random init. Neither
  encodes *which* encoder was used — check the stored args, or the preset.
- Predictions land in `results/<output_dir>/<pred_subdir>/<split>/*_seg.nii.gz`

## Before you commit

```bash
python -m compileall -q .                 # syntax
bash -n train.sh && bash -n eval.sh       # launcher syntax
git ls-files | git check-ignore --no-index --stdin   # must be empty
```

The last one catches tracked files that a `.gitignore` change would silently
drop on the next fresh add.
