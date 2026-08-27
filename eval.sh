#!/usr/bin/env bash
# Unified inference launcher.  Usage:  bash eval.sh <config>
#   configs: voco | voxelfox | smit_act | swin_act | effidec3d
# Writes predictions to results/<RUN>/ using the same run-dir naming as train.sh.
set -euo pipefail
CFG="${1:?usage: bash eval.sh <voco|voxelfox|smit_act|swin_act|effidec3d> [extra --flags...]}"; shift

# MODEL = architecture key (selects the network to build). PRED = prediction subfolder name
# (kept distinct so voco vs voxelfox don't collide under the same arch key).
case "$CFG" in
  voco)      RUN=rectal_voco_swinunetr_96x96x64_pretrained;     MODEL=swinv2; PRED=voco_swinunetr;     RX=96;  RY=96;  RZ=64; EXTRA="--swin_use_v2" ;;
  voxelfox)  RUN=rectal_voxelfox_swinunetr_96x96x64_pretrained; MODEL=swinv2; PRED=voxelfox_swinunetr; RX=96;  RY=96;  RZ=64; EXTRA="--swin_use_v2" ;;
  smit_act)  RUN=rectal_smit_128x128x64_pretrained;             MODEL=smit;           PRED=smit;               RX=128; RY=128; RZ=64; EXTRA="" ;;
  swin_act)  RUN=rectal_swinunetr_96x96x64_pretrained;          MODEL=swinunetr;      PRED=swinunetr;          RX=96;  RY=96;  RZ=64; EXTRA="" ;;
  # EffiDec3D — decoder settings MUST match training so the checkpoint loads cleanly.
  effidec3d) RUN=rectal_effidec3d_96x96x64_pretrained;          MODEL=effidec3d;      PRED=effidec3d;          RX=96;  RY=96;  RZ=64; EXTRA="--swin_use_v2 --n_decoder_channels 48 --resolution_factor 2 --head_upsample trilinear" ;;
  *) echo "unknown config: $CFG"; exit 1 ;;
esac

# SCRATCH=1 -> evaluate the "_scratch" run instead of "_pretrained"
[ "${SCRATCH:-0}" = "1" ] && RUN="${RUN/_pretrained/_scratch}"
RUN="${RUN_OVR:-$RUN}"

# data/output overrides (env) — evaluate another cohort without editing the script.
DATA_DIR="${DATA_DIR:-data_rectal}"
JSON="${JSON:-Trainval_set1.json}"
DATASETS="${DATASETS:-validation}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN}"     # where predictions go; model checkpoint stays runs/$RUN
PRED="${PRED_SUBDIR:-$PRED}"         # prediction subfolder name (override with PRED_SUBDIR=...)
INTENSITY_MODE="${INTENSITY_MODE:-fixed}"
PERCENTILE_LOWER="${PERCENTILE_LOWER:-0}"
PERCENTILE_UPPER="${PERCENTILE_UPPER:-99.5}"

# PROBS=1 -> also dump foreground probability maps (<key>_prob.nii.gz) for Gate-A/threshold analysis.
# (equivalent to appending --save_probs; kept as an env toggle for convenience.)
[ "${PROBS:-0}" = "1" ] && EXTRA="$EXTRA --save_probs"

# NGPU>1 -> torchrun data-parallel inference (each GPU runs a shard of cases, ~NGPU x faster).
# main_inference.py only distributes when launched via torchrun (needs RANK/WORLD_SIZE/LOCAL_RANK);
# plain `python` falls back to a single GPU.
# default NGPU = number of visible GPUs (respects CUDA_VISIBLE_DEVICES); override with NGPU=...
NGPU="${NGPU:-$(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 1)}"
[ "${NGPU:-0}" -ge 1 ] 2>/dev/null || NGPU=1
echo ">> inference $CFG | ${NGPU} GPU(s) | model=runs/$RUN | data=$DATA_DIR ($JSON:$DATASETS) intensity=$INTENSITY_MODE -> results/$OUTPUT_DIR/$PRED"
torchrun --nproc_per_node="$NGPU" main_inference.py \
  --distributed \
  --data_dir "$DATA_DIR" \
  --json_list "$JSON" \
  --datasets "$DATASETS" \
  --results_dir results \
  --output_dir "$OUTPUT_DIR" \
  --model_name "$MODEL" \
  --pred_subdir "$PRED" \
  --pretrained_model_path "runs/$RUN/model_final.pt" \
  --in_channels 1 --out_channels 2 \
  --feature_size 48 \
  --norm_name instance \
  --intensity_mode "$INTENSITY_MODE" \
  --a_min 0 --a_max 800 \
  --percentile_lower "$PERCENTILE_LOWER" --percentile_upper "$PERCENTILE_UPPER" \
  --space_x 1.0 --space_y 1.0 --space_z 1.0 \
  --roi_x "$RX" --roi_y "$RY" --roi_z "$RZ" \
  --sw_batch_size "${SW_BATCH:-8}" \
  --use_tta \
  $EXTRA \
  "$@"   # pass-through: any extra --flag overrides the preset (argparse last-wins)
