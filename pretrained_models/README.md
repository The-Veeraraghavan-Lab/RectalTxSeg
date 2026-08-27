# Pretrained weights - drop folder

This directory ships empty. Put the SSL checkpoints here, **using these exact filenames** - `train.sh`, `eval.sh`, and the CKA driver refer to them by name, so a renamed file will not be found.

```
pretrained_models/
  voxelfox_swinvit.pt      # MSK-hosted - see below
  model_smit_ct10k.pth     # MSK-hosted - see below
  model_swinvit.pt         # public - download, see below
  voco_b_swinvit.pt        # public - download, see below
```

## MSK-hosted checkpoints

These two were pretrained in-house on **CT10K** (10,444 public 3D CT volumes) and are available from the public model-weight folder: <https://mskcc.box.com/s/df1qdt2js9495ahtldgkooh1cigrwml8>.

| File | Size | MD5 | Used by |
|---|---|---|---|
| `voxelfox_swinvit.pt` | 602 MB | `e9f79e9bb06800fac1bd322055da7200` | `train.sh voxelfox`, `train.sh effidec3d` - the SWIFT family |
| `model_smit_ct10k.pth` | 770 MB | `89038a82d29ed5462d6a49dd14c515b2` | `train.sh smit_act`, SMIT CKA config, t-SNE |

Verify after copying:

```bash
md5sum pretrained_models/voxelfox_swinvit.pt pretrained_models/model_smit_ct10k.pth
```

The same Box folder also hosts the selected fine-tuned checkpoints: the three BSPC SMIT runs (Base, TA, and ACT/reference) and the four MIART SWIFT-family models initialized from VoxelFox. Those files do **not** belong in `pretrained_models/`; restore each one under `runs/<run-name>/model_final.pt`. See the "Public fine-tuned checkpoints" section in the top-level `README.md` for the exact run-directory names.

## Public upstream checkpoints

For public weights, cite and point readers to the upstream project first. The download commands below are convenience commands that save the upstream files under the filenames expected by this repository.

### `model_swinvit.pt` - NVIDIA / MONAI SwinUNETR

SwinUNETR SSL encoder, self-supervised on 5,050 public CT volumes via
reconstruction, rotation-prediction, and contrastive pretext tasks.

Project page: <https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/Pretrain>
MONAI project: <https://github.com/Project-MONAI/MONAI>

```bash
curl -L -o pretrained_models/model_swinvit.pt \
  https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt
```

393 MB, MD5 `c8d9deb6cbbd66839bc8dced6010103f`. Used by `train.sh swin_act` and
the SwinUNETR CKA config.

Reference: Tang et al., "Self-Supervised Pre-Training of Swin Transformers for
3D Medical Image Analysis," CVPR 2022.

### `voco_b_swinvit.pt` - VoCo Base

Download **`VoCo_B_SSL_head.pt`** (53M params) and save it under our filename:

Project page: <https://github.com/Luffy03/Large-Scale-Medical>
Model hub: <https://huggingface.co/Luffy503/VoCo/tree/main>

```bash
curl -L -o pretrained_models/voco_b_swinvit.pt \
  "https://huggingface.co/Luffy503/VoCo/resolve/main/VoCo_B_SSL_head.pt?download=true"
```

Reference: Wu, Zhuang & Chen, "Large-Scale 3D Medical Image Pre-training with
Geometric Context Priors," TPAMI 2026.

> Use the **Large-Scale-Medical** repository, not the older `Luffy03/VoCo` CVPR
> repo - that one does not host checkpoints.

Used by `train.sh voco`, the initialization-generalization check.

## Notes

- Nothing in this folder is tracked by git. Only this README is.
- The SwinUNETR loader normalizes key namespaces before `load_from()`
  (`module.swinViT.*` → `module.*`, `mlp.linear1/2` → `mlp.fc1/fc2`). NVIDIA's
  checkpoint already uses `module.*`, so the remap is a no-op for it - but it
  must stay for CT10K-style checkpoints.
- `models/check_voxelfox_load.py` and `models/check_effidec3d_load.py` dry-load a
  checkpoint and report missing/unexpected key counts. Run one before a long
  training job; they need no GPU and finish in seconds.
