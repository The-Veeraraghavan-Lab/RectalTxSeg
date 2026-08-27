# RectalTxSeg

**3D rectal tumor segmentation on T2-weighted pelvic MRI with CT-pretrained transformer backbones.**

[BSPC paper](https://doi.org/10.1016/j.bspc.2026.111229) ([arXiv](https://arxiv.org/abs/2605.05522)) ·
[MIART 2026](https://miart-workshop.github.io/) ·
[model weights](https://mskcc.box.com/s/df1qdt2js9495ahtldgkooh1cigrwml8) ·
[license](LICENSE)

RectalTxSeg provides distributed training, sliding-window inference, model checkpointing, and analysis utilities for volumetric rectal-tumor segmentation. The repository supports SMIT, SwinUNETR, SwinUNETR V2, EffiDec3D, LoRA, and LoRA-decoder ensembles in a shared training and evaluation pipeline.

The code accompanies two accepted works: a cross-modality transfer study in *Biomedical Signal Processing and Control* and the SWIFT efficiency-calibration study at the MICCAI 2026 MIART workshop.

## Publications

| Work | Main question | Repository components |
|---|---|---|
| **BSPC 2026** - [*Tumor-aware augmentation with task-guided attention analysis improves rectal cancer segmentation from magnetic resonance images*](https://doi.org/10.1016/j.bspc.2026.111229) | Why can CT-pretrained transformers transfer poorly to rectal MRI, and how can task-specific augmentation and crop geometry improve that transfer? | SMIT and SwinUNETR benchmarks; scratch versus SSL initialization; tumor-aware augmentation; anisotropic cropping; attention dilution index (ADI); centered kernel alignment (CKA). |
| **MIART at MICCAI 2026** - *Parameter-Efficient Pretrained-CT-to-MRI Transfer for Rectal Cancer Segmentation: Performance-Calibration Trade-offs* | How do decoder compression, parameter-efficient adaptation, and ensembling affect segmentation performance, efficiency, and calibration? | SWIFT; SWIFTe; SWIFTe-LoRA; SWIFTe-LDE4; EffiDec3D; calibration; uncertainty; radiomics agreement; runtime benchmarking. |

The two studies share the same core data, training, inference, and metric infrastructure. The model and analysis paths specific to each paper are mapped below.

## At a Glance

| Item | Description |
|---|---|
| **Input** | 3D T2-weighted pelvic MRI, reoriented to RAS and resampled to 1.0 mm isotropic spacing. |
| **Output** | Two-channel background/tumor segmentation. |
| **Training** | PyTorch distributed training with MONAI transforms, checkpointing, and optional SSL initialization or LoRA adaptation. |
| **Inference** | Distributed sliding-window inference with optional test-time augmentation and probability-map export. |
| **Weights** | Public upstream encoders plus selected MSK-hosted pretrained and fine-tuned checkpoints. |
| **Data** | MONAI Decathlon-style JSON manifests. The clinical cohort is not publicly distributed. |

## Quick Start

Run all commands from the repository root.

### 1. Install the environment

Install PyTorch first using the build appropriate for your CUDA environment, then install the remaining dependencies:

```bash
pip install -r requirements.txt
pip install git+https://github.com/google-deepmind/surface-distance.git
```

### 2. Download a pretrained encoder

Place pretrained encoder checkpoints in `pretrained_models/` using the exact filenames expected by the launchers:

```text
pretrained_models/
  voxelfox_swinvit.pt
  model_smit_ct10k.pth
  model_swinvit.pt
  voco_b_swinvit.pt
```

See [`pretrained_models/README.md`](pretrained_models/README.md) for download commands, checksums, and upstream citations. VoxelFox, SMIT, and selected fine-tuned weights are available from the [MSK public weight folder](https://mskcc.box.com/s/df1qdt2js9495ahtldgkooh1cigrwml8).

### 3. Prepare the dataset manifest

The default launchers expect:

```text
data_rectal/
  Trainval_set1.json
  imagesTr/  labelsTr/
  imagesTs/  labelsTs/
```

### 4. Train and evaluate

The same preset name is used for training and inference:

```bash
# BSPC SMIT-ACT configuration
bash train.sh smit_act
bash eval.sh smit_act

# MIART SWIFT configuration
bash train.sh voxelfox
bash eval.sh voxelfox
```

> **Important:** `train.sh` overwrites an existing run directory without
> warning. Set `RUN_OVR=<new-name>` when an existing result must be preserved.

## Reproducing the Paper Models

### Launcher presets

`bash train.sh <preset>` and `bash eval.sh <preset>` use matching architectures, crop geometry, checkpoint names, and output directories.

| Preset | Paper role | Encoder initialization | Architecture | Run directory |
|---|---|---|---|---|
| `smit_act` | BSPC primary benchmark | `model_smit_ct10k.pth` | SMIT | `rectal_smit_128x128x64_pretrained` |
| `swin_act` | BSPC primary benchmark | `model_swinvit.pt` | SwinUNETR | `rectal_swinunetr_96x96x64_pretrained` |
| `voco` | BSPC supporting benchmark and public-initialization check | `voco_b_swinvit.pt` | SwinUNETR V2 | `rectal_voco_swinunetr_96x96x64_pretrained` |
| `voxelfox` | MIART: SWIFT | `voxelfox_swinvit.pt` | SwinUNETR V2 | `rectal_voxelfox_swinunetr_96x96x64_pretrained` |
| `effidec3d` | MIART: SWIFTe | `voxelfox_swinvit.pt` | SwinUNETR V2 + EffiDec3D | `rectal_effidec3d_96x96x64_pretrained` |

Run a scratch-initialized comparison by setting `SCRATCH=1`. The launcher skips SSL loading and changes `_pretrained` to `_scratch` in the run name:

```bash
SCRATCH=1 bash train.sh smit_act
SCRATCH=1 bash eval.sh smit_act
```

All launcher presets use the ACT recipe: an anisotropic crop, the advanced data loader, and tumor-aware augmentation. Disable only the tumor-aware augmentation for an ablation with `TUMOR_AUG=0`. Use `RUN_OVR` so the ablation does not overwrite the primary run.

```bash
TUMOR_AUG=0 RUN_OVR=rectal_effidec3d_96x96x64_pretrained_noTA \
  bash train.sh effidec3d
```

### BSPC configurations

The BSPC study compares scratch and CT-pretrained SMIT and SwinUNETR models, then evaluates tumor-aware augmentation (TA) and anisotropic cropping combined with tumor-aware augmentation (ACT). Selected SMIT checkpoints are hosted in the MSK public weight folder.

| Paper label | Run directory |
|---|---|
| SMIT-Base | `rectal_smit_128x128x128_base` |
| SMIT-TA | `rectal_smit_128x128x128_pretrained` |
| SMIT-ACT / reference | `rectal_smit_128x128x64_pretrained` |

The current `smit_act` and `swin_act` presets reproduce the ACT recipe. The Base and TA run names are retained for compatibility with the accepted-paper checkpoints and analysis configurations.

### MIART / SWIFT configurations

SWIFT model names identify trained configurations; the launcher and `--model_name` values identify the underlying code paths.

| Paper model | Configuration | Run directory |
|---|---|---|
| **SWIFT** | VoxelFox-pretrained Swin V2 encoder with the full SwinUNETR V2 decoder; end-to-end fine-tuning. | `rectal_voxelfox_swinunetr_96x96x64_pretrained` |
| **SWIFTe** | The same encoder with an EffiDec3D decoder; end-to-end fine-tuning. | `rectal_effidec3d_96x96x64_pretrained` |
| **SWIFTe-LoRA** | Frozen encoder with rank-4 LoRA adapters and a trained EffiDec3D decoder. | `rectal_effidec3d_loraS_r4_bs2` |
| **SWIFTe-LDE4** | Four member-specific rank-4 LoRA adapter-decoder pairs over a shared frozen encoder. | `rectal_effidec3d_loraLDE4` |

SWIFT and SWIFTe are available directly through `train.sh`. The LoRA variants require the matching LoRA and decoder flags; complete training and inference commands are provided in [`models/lora_ensemble/README.md`](models/lora_ensemble/README.md).

Selected fine-tuned checkpoints for all four SWIFT-family models are hosted in the MSK public weight folder. Restore a downloaded checkpoint as:

```text
runs/<run-directory>/model_final.pt
```

The VoCo-initialized `voco` configuration and `*_voco_*` analysis paths test whether the observed trends depend on the MSK VoxelFox initialization. They are not additional SWIFT models and are not part of the hosted MSK fine-tuned checkpoint release.

## Architectures and Pretrained Weights

### Architecture keys

`--model_name` selects the network architecture. Pretraining is selected separately with `--ssl_weights_path`.

| Key | Network |
|---|---|
| `smit` | SMIT encoder with a UNETR-style decoder. |
| `swinunetr` | MONAI/NVIDIA SwinUNETR. |
| `swinv2` | MONAI SwinUNETR V2; requires `--swin_use_v2`. Used by SWIFT and VoCo. |
| `effidec3d` | Swin V2 encoder with the EffiDec3D decoder; requires `--swin_use_v2`. |

`voco_swinunetr` remains available as a legacy alias for `swinv2`.

### Encoder checkpoints

All pretrained encoders are CT-pretrained and fine-tuned on MRI. This cross-modality transfer is deliberate.

| Checkpoint | Encoder | Pretraining | Source |
|---|---|---|---|
| `voxelfox_swinvit.pt` | Swin V2 | DINOv2-style self-distillation on CT10K: 10,444 public CT volumes. | MSK public weight folder |
| `model_smit_ct10k.pth` | SMIT | SMIT self-distillation on the same CT10K corpus. | MSK public weight folder |
| `model_swinvit.pt` | SwinUNETR | NVIDIA/MONAI SSL on 5,050 public CT volumes. | [MONAI research contributions](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/Pretrain) |
| `voco_b_swinvit.pt` | Swin V2 | VoCo geometric-context pretraining. | [Large-Scale-Medical](https://github.com/Luffy03/Large-Scale-Medical) |

There is no recommended “CT10K SwinUNETR” checkpoint in this public workflow. The BSPC `swin_act` preset uses NVIDIA/MONAI's `model_swinvit.pt`; SWIFT uses the VoxelFox `voxelfox_swinvit.pt` encoder.

<details>
<summary><strong>Checkpoint-loading details</strong></summary>

- SwinUNETR normalizes `module.swinViT.*` to `module.*` and `mlp.linear1/2` to `mlp.fc1/fc2` before `load_from()`. This is a no-op for NVIDIA's checkpoint but is required for CT10K-style namespaces.
- Swin V2 and EffiDec3D load through `models/swin_voco_utils.py:load_swin_encoder_pretrained`. The loader recognizes DINOv2-style `teacher`/`student` checkpoints, VoCo `state_dict` checkpoints, and the supported key prefixes.
- Only the encoder is loaded from an SSL checkpoint; the segmentation decoder is initialized separately.
- Before starting a long training run, dry-load VoxelFox with `python models/check_voxelfox_load.py` or `python models/check_effidec3d_load.py`. These checks require no GPU and report missing and unexpected keys.

</details>

## Data and Preprocessing

### Manifest format

Training and inference use MONAI Decathlon-style JSON files. Paths are relative to `--data_dir`.

```json
{
  "training": [
    {"image": "./imagesTr/case001_z800.nii.gz", "label": "./labelsTr/case001.nii.gz"}
  ],
  "validation": [
    {"image": "./imagesTs/case101_z800.nii.gz", "label": "./labelsTs/case101.nii.gz"}
  ],
  "testing": [
    {"image": "./imagesTs/case201.nii.gz"}
  ]
}
```

Labels are required for training and metric computation. Image-only entries are sufficient for inference.

### Intensity modes

The intensity mode must match the way the images were prepared and must remain consistent when models are compared.

| Input | Mode | Runtime settings |
|---|---|---|
| Prepared images scaled to `[0, 800]`, conventionally named `*_z800.nii.gz` | `fixed` | `--intensity_mode fixed --a_min 0 --a_max 800` |
| Raw NIfTI images | `percentile` | `--intensity_mode percentile --percentile_lower 0 --percentile_upper 99.5` |

The launchers default to `fixed`. The `_z800` suffix is a naming convention, not an input requirement. The script originally used to generate the prepared `z800` volumes is local-only and is not tracked in this repository.

For a checkpoint trained on prepared `z800` images, online percentile scaling is the closest supported path for inference on raw NIfTI volumes.

## Training and Inference

### Launcher overrides

The shell launchers accept environment-variable overrides without requiring edits to the scripts.

```bash
INTENSITY_MODE=percentile \
DATA_DIR=/path/to/raw_dataset \
JSON=single_case.json \
DATASETS=testing \
OUTPUT_DIR=single_case_eval \
bash eval.sh smit_act
```

Useful inference overrides include:

| Variable | Purpose |
|---|---|
| `DATA_DIR`, `JSON`, `DATASETS` | Select another dataset and manifest split. |
| `OUTPUT_DIR`, `PRED_SUBDIR` | Control the result and prediction subdirectories. |
| `INTENSITY_MODE` | Switch between prepared and raw-image normalization. |
| `PROBS=1` | Export foreground probability maps in addition to segmentations. |
| `NGPU=<n>` | Select the number of distributed inference workers. |
| `RUN_OVR=<name>` | Select or protect a specific run directory. |

Predictions are written to:

```text
results/<output_dir>/<pred_subdir>/<split>/*_seg.nii.gz
```

### Folder inference

Run inference on a single raw NIfTI, a folder of NIfTIs, a DICOM series, or a folder of DICOM series:

```bash
python analysis/preprocessing/run_folder_inference.py \
  --input /path/to/raw_nifti_folder \
  --checkpoint runs/rectal_smit_128x128x64_pretrained/model_final.pt \
  --output_dir folder_eval
```

For images already prepared in `[0, 800]`, add:

```bash
--intensity_mode fixed --a_min 0 --a_max 800
```

<details>
<summary><strong>Direct inference example</strong></summary>

```bash
python main_inference.py \
  --data_dir /path/to/raw_dataset \
  --json_list single_case.json \
  --datasets testing \
  --results_dir results \
  --output_dir single_case_eval \
  --model_name smit \
  --pretrained_model_path runs/rectal_smit_128x128x64_pretrained/model_final.pt \
  --out_channels 2 \
  --norm_name instance \
  --intensity_mode percentile \
  --percentile_lower 0 \
  --percentile_upper 99.5 \
  --roi_x 128 --roi_y 128 --roi_z 64 \
  --sw_batch_size 8 \
  --use_tta
```

</details>

## Repository Guide

| Area | Path | Purpose |
|---|---|---|
| Core | `main_training.py` | Distributed training; model construction, SSL loading, and LoRA injection. |
| Core | `trainer.py` | Training loop, validation, checkpointing, and scalar logging. |
| Core | `main_inference.py` | Canonical model-loading path and sliding-window inference. |
| Core | `utils/data_utils.py` | MONAI data loaders, augmentation, and intensity normalization. |
| Core | `train.sh`, `eval.sh` | Matching presets for common training and inference configurations. |
| Models | `models/smit.py`, `models/configs_smit.py` | SMIT encoder and decoder configurations. |
| Models | `models/swin_nvidia.py` | MONAI/NVIDIA SwinUNETR. |
| Models | `models/effidec3d.py`, `models/swin_voco_utils.py` | Swin V2 construction and the EffiDec3D decoder. |
| Models | `models/lora_ensemble/` | LoRA injection, adapter-decoder ensembles, and uncertainty evaluation. |
| Analysis | `analysis/metrics/` | Dice, HD95, and surface-Dice scoring. |
| Analysis | `analysis/cka/` | CKA representation analysis. |
| Analysis | `analysis/attention/` | Architecture-agnostic attention-flow and ADI utilities. |
| Analysis | `analysis/preprocessing/` | Folder and single-case inference helpers. |
| Analysis | `analysis/features/` | Feature and embedding extraction. |
| Analysis | `analysis/uncertainty/` | Calibration, ECE, temperature scaling, and confidence analysis. |
| Analysis | `analysis/runtime/` | Inference runtime and memory benchmarking. |
| Analysis | `analysis/tumor/` | Tumor intensity and appearance analysis. |
| Analysis | `analysis/stats/` | Parameter counting. |

See [`analysis/README.md`](analysis/README.md) for analysis-specific usage.

The shipped CKA configurations reference several accepted-paper run names that predate the current launchers, including `rectal_smit_128x128x128_base`, `rectal_swinunetr_96x96x96_base`, and `rectal_swinunetr_96x96x96_pretrained`. Update the configurations to point to your own `runs/` directories when those checkpoints are not available locally.

## Reproducibility and Safety Notes

- **Match the training ROI.** An incorrect `--roi_z` may load successfully but invalidate the result. When stored training arguments are available, `main_inference.py` hard-fails on ROI mismatches. Use `--allow_roi_mismatch` only deliberately.
- **Older checkpoints may not contain training arguments.** Current checkpoints store `args`; older checkpoints may store no arguments or an `argparse.Namespace`. If no stored arguments are available, architecture and geometry must be matched manually.
- **Mirror LoRA construction at inference.** A LoRA checkpoint cannot load into a plain model. Evaluation must use the same `--use_lora`, rank, member count, decoder-ensemble setting, and EffiDec3D flags used during training.
- **Architecture and initialization are separate.** `voco` and `voxelfox` both construct `swinv2`; their different behavior comes from `--ssl_weights_path`. A `_pretrained` run name does not identify which SSL checkpoint was used.
- **Keep intensity preprocessing consistent.** Mixing `fixed` and `percentile` modes invalidates model comparisons.
- **Run from the repository root.** Analysis scripts rely on root-relative imports and paths.

## Repository Scope and Data Availability

This repository tracks source code and reusable documentation. It does not distribute patient images, labels, derived case tables, or clinical metadata. The clinical cohort is not publicly available.

The following are generated or restored locally and excluded from git:

- images and labels under `data_rectal/`;
- pretrained encoders under `pretrained_models/`;
- trained checkpoints and TensorBoard logs under `runs/`;
- predictions under `results/`;
- derived tables, figures, embeddings, probability maps, and case bundles under
  the analysis output directories;
- radiomics extractions requiring CERR and local path configuration.

## License

MIT - see [LICENSE](LICENSE). Files derived from MONAI and MONAI research contributions retain their Apache-2.0 notices. SMIT and other third-party attribution and licensing details are documented in [NOTICE](NOTICE).

## Acknowledgements

This study was supported by the Simons Foundation and the Breast Cancer Research Foundation (through grant MATH-23-001), and the NIH ROBIN cooperative group (grant U54CA274291). This work utilized resources from the High-Performance Computing Group at Memorial Sloan Kettering Cancer Center.

## Citation

If you use this repository, please cite these two works:

```bibtex
@article{rangnekar2026tumor,
  title={Tumor-aware augmentation with task-guided attention analysis improves rectal cancer segmentation from magnetic resonance images},
  author={Rangnekar, Aneesh and Miranda, Joao and Horvat, Natally and Chahwan, Stephanie and Alrayess, Samir and Apte, Aditya and Iyer, Aditi and LoCastro, Eve and Ravella, Revathi and Gollub, Marc J and others},
  journal={Biomedical Signal Processing and Control},
  volume={128},
  pages={111229},
  year={2026},
  publisher={Elsevier}
}
```

```bibtex
@inproceedings{rangnekar2026parameterefficient,
  title     = {Parameter-Efficient Pretrained-CT-to-MRI Transfer for Rectal
               Cancer Segmentation: Performance-Calibration Trade-offs},
  author    = {Rangnekar, Aneesh and Tapias Gomez, Jorge and
               Deasy, Joseph O and Veeraraghavan, Harini},
  booktitle = {Medical Image AI in Radiation Therapy (MIART), MICCAI 2026},
  year      = {2026},
  note      = {Accepted; to appear in the MICCAI 2026 Springer workshop proceedings}
}
```
