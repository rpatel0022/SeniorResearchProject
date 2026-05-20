#!/usr/bin/env bash
# Launch wrapper for src.train_qwen_cosyn that survives the failure mode
# from run3 (silent death with block-buffered stdout).
#
# Inputs (env vars, all required unless defaulted):
#   GPU             - CUDA_VISIBLE_DEVICES (default: 0)
#   OUT_DIR         - --output_dir (required)
#   LOG             - log file path (required)
#   ALIGN_W         - --alignment_loss_weight (required)
#   ALIGN_LAYER     - --alignment_layer (required)
#   ALIGN_MODE      - --alignment_loss_mode (default: lm_head_bow)
#   FINAL_NORM      - "on" or "off" for --lm_head_bow_use_final_norm (default: on)
#   EPOCHS          - --num_epochs (default: 1)
#   PER_BS          - --per_device_batch_size (default: 2)
#   ACCUM           - --grad_accum_steps (default: 8)
#   SAVE_STEPS      - --save_steps (default: 2000)
#   SAVE_LIMIT      - --save_total_limit (default: 2)
#   LOG_EVERY       - --log_every (default: 50)
#   ATTN            - --attn_implementation (default: sdpa)
#   N_TRAIN         - --num_train_samples (default: 0 = full)
#   N_VAL           - --num_val_samples (default: 0 = full)
#   RESUME_FROM     - --resume_from path (default: empty)
#
# Behavior:
#   - Sets PYTHONUNBUFFERED=1 and runs python -u so stdout is unbuffered.
#   - Pipes through tee so stderr survives if the python process dies suddenly.
#   - Detaches via nohup + disown so SSH disconnects don't kill it.
#   - Writes the PID to ${OUT_DIR}/run.pid for the watchdog.

set -euo pipefail

: "${OUT_DIR:?OUT_DIR is required}"
: "${LOG:?LOG is required}"
: "${ALIGN_W:?ALIGN_W is required}"
: "${ALIGN_LAYER:?ALIGN_LAYER is required}"

GPU="${GPU:-0}"
ALIGN_MODE="${ALIGN_MODE:-lm_head_bow}"
FINAL_NORM="${FINAL_NORM:-on}"
EPOCHS="${EPOCHS:-1}"
PER_BS="${PER_BS:-2}"
ACCUM="${ACCUM:-8}"
SAVE_STEPS="${SAVE_STEPS:-2000}"
SAVE_LIMIT="${SAVE_LIMIT:-2}"
LOG_EVERY="${LOG_EVERY:-50}"
ATTN="${ATTN:-sdpa}"
N_TRAIN="${N_TRAIN:-0}"
N_VAL="${N_VAL:-0}"
RESUME_FROM="${RESUME_FROM:-}"

REPO_ROOT="/data2/hshah057/rushi_workspace/SeniorResearchProject"
PYBIN="/data2/hshah057/miniconda/envs/rushi_vlm/bin/python"

mkdir -p "$(dirname "$LOG")" "$OUT_DIR"

FINAL_NORM_FLAG="--lm_head_bow_use_final_norm"
if [[ "$FINAL_NORM" == "off" ]]; then
  FINAL_NORM_FLAG="--no-lm_head_bow_use_final_norm"
fi

RESUME_FLAG=""
if [[ -n "$RESUME_FROM" ]]; then
  RESUME_FLAG="--resume_from $RESUME_FROM"
fi

ALIGNED_FLAG=""
if [[ "${ALIGNED_ONLY:-off}" == "on" ]]; then
  ALIGNED_FLAG="--aligned_only"
fi

cd "$REPO_ROOT"

# Stamp the launch in the log header so it's easy to grep
{
  echo "=== launch_align.sh at $(date -Iseconds) ==="
  echo "GPU=$GPU OUT_DIR=$OUT_DIR ALIGN_W=$ALIGN_W ALIGN_LAYER=$ALIGN_LAYER ALIGN_MODE=$ALIGN_MODE FINAL_NORM=$FINAL_NORM"
  echo "EPOCHS=$EPOCHS PER_BS=$PER_BS ACCUM=$ACCUM SAVE_STEPS=$SAVE_STEPS SAVE_LIMIT=$SAVE_LIMIT LOG_EVERY=$LOG_EVERY ATTN=$ATTN N_TRAIN=$N_TRAIN N_VAL=$N_VAL"
  if [[ -n "$RESUME_FROM" ]]; then echo "RESUME_FROM=$RESUME_FROM"; fi
} >> "$LOG"

# shellcheck disable=SC2086
nohup env \
  PYTHONNOUSERSITE=1 \
  PYTHONUNBUFFERED=1 \
  CUDA_VISIBLE_DEVICES="$GPU" \
  HF_HOME=/data2/hshah057/rushi_workspace/hf_cache \
  PYTHONPATH=. \
  "$PYBIN" -u -m src.train_qwen_cosyn \
    --alignment_loss_mode "$ALIGN_MODE" \
    $FINAL_NORM_FLAG \
    --alignment_loss_weight "$ALIGN_W" \
    --alignment_layer "$ALIGN_LAYER" \
    --num_epochs "$EPOCHS" \
    --per_device_batch_size "$PER_BS" \
    --grad_accum_steps "$ACCUM" \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit "$SAVE_LIMIT" \
    --log_every "$LOG_EVERY" \
    --attn_implementation "$ATTN" \
    --num_train_samples "$N_TRAIN" \
    --num_val_samples "$N_VAL" \
    --output_dir "$OUT_DIR" \
    $RESUME_FLAG \
    $ALIGNED_FLAG \
  >> "$LOG" 2>&1 &

PID=$!
echo "$PID" > "$OUT_DIR/run.pid"
disown $PID 2>/dev/null || true

echo "[launch_align.sh] started PID=$PID"
echo "[launch_align.sh] log=$LOG"
echo "[launch_align.sh] pid_file=$OUT_DIR/run.pid"
