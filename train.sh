#!/usr/bin/env bash
# Unified training launcher.  Usage:  bash train.sh <config>
#   configs: voco | voxelfox | smit_act | swin_act | effidec3d
# All use the ACT recipe (anisotropic crop + tumor-aware aug + adv loader).
# Output run dir names match the existing convention so prior results still line up.
set -euo pipefail
CFG="${1:?usage: bash train.sh <voco|voxelfox|smit_act|swin_act|effidec3d> [extra --flags...]}"; shift

case "$CFG" in
  voco)      RUN=rectal_voco_swinunetr_96x96x64_pretrained;     MODEL=swinv2; RX=96;  RY=96;  RZ=64; SSL=voco_b_swinvit.pt;      EXTRA="--swin_use_v2" ;;
  voxelfox)  RUN=rectal_voxelfox_swinunetr_96x96x64_pretrained; MODEL=swinv2; RX=96;  RY=96;  RZ=64; SSL=voxelfox_swinvit.pt;    EXTRA="--swin_use_v2" ;;
  smit_act)  RUN=rectal_smit_128x128x64_pretrained;             MODEL=smit;           RX=128; RY=128; RZ=64; SSL=model_smit_ct10k.pth;   EXTRA="" ;;
  swin_act)  RUN=rectal_swinunetr_96x96x64_pretrained;          MODEL=swinunetr;      RX=96;  RY=96;  RZ=64; SSL=model_swinvit.pt;      EXTRA="" ;;
  # EffiDec3D decoder on the SwinV2 encoder; encoder init from VoxelFox SSL by default.
  # Decoder settings overridable by appending flags (argparse last-wins), e.g.:
  #   bash train.sh effidec3d --resolution_factor 4 --head_upsample upconv
  effidec3d) RUN=rectal_effidec3d_96x96x64_pretrained;          MODEL=effidec3d;      RX=96;  RY=96;  RZ=64; SSL=voxelfox_swinvit.pt;    EXTRA="--swin_use_v2 --n_decoder_channels 48 --resolution_factor 2 --head_upsample trilinear" ;;
  *) echo "unknown config: $CFG"; exit 1 ;;
esac

# Default epochs = 500 for every config (apples-to-apples with the baselines).
# effidec3d is ~3x lighter, so a longer run is cheap -- do it via override, not by default:
#   EPOCHS=1000 bash train.sh effidec3d      # 2x
#   EPOCHS=750  bash train.sh effidec3d      # 1.5x

# env-overridable defaults (also overridable via pass-through flags below)
BATCH="${BATCH:-2}"; VAL_EVERY="${VAL_EVERY:-50}"
INTENSITY_MODE="${INTENSITY_MODE:-fixed}"
PERCENTILE_LOWER="${PERCENTILE_LOWER:-0}"
PERCENTILE_UPPER="${PERCENTILE_UPPER:-99.5}"
# TUMOR_AUG=1 (default) -> tumour-intensity aug ON (main recipe, unchanged).
# TUMOR_AUG=0           -> ablation twin: adv loader stays, only RandTumorIntensityd is dropped.
TA_ARG="--use_tumor_aug"; [ "${TUMOR_AUG:-1}" = "0" ] && TA_ARG=""
RX="${RX_OVR:-$RX}"; RY="${RY_OVR:-$RY}"; RZ="${RZ_OVR:-$RZ}"; SSL="${SSL_OVR:-$SSL}"

# SCRATCH=1 -> random init (no SSL weights), default 1000 epochs, "_scratch" run name
if [ "${SCRATCH:-0}" = "1" ]; then
  RUN="${RUN/_pretrained/_scratch}"; EPOCHS="${EPOCHS:-1000}"; SSL_ARGS=""
else
  EPOCHS="${EPOCHS:-500}"; SSL_ARGS="--use_ssl_pretrained --ssl_weights_path $SSL"
fi
RUN="${RUN_OVR:-$RUN}"

echo ">> training $CFG -> runs/$RUN  (model=$MODEL roi=${RX}x${RY}x${RZ} ssl=$SSL epochs=$EPOCHS intensity=$INTENSITY_MODE)"
echo ">> extra args: $*"
python main_training.py \
  --logdir "$RUN" \
  --model_name "$MODEL" \
  --data_dir data_rectal \
  --json_list Trainval_set1.json \
  --save_checkpoint \
  --distributed \
  --in_channels 1 --out_channels 2 \
  --feature_size 48 \
  --norm_name instance \
  --intensity_mode "$INTENSITY_MODE" \
  --a_min 0 --a_max 800 \
  --percentile_lower "$PERCENTILE_LOWER" --percentile_upper "$PERCENTILE_UPPER" \
  --space_x 1.0 --space_y 1.0 --space_z 1.0 \
  --roi_x "$RX" --roi_y "$RY" --roi_z "$RZ" \
  --max_epochs "$EPOCHS" --batch_size "$BATCH" --val_every "$VAL_EVERY" \
  $SSL_ARGS \
  --use_adv_loader $TA_ARG \
  $EXTRA \
  "$@"   # pass-through: any extra --flag overrides the preset (argparse last-wins)
