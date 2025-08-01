#!/usr/bin/env bash
#
# train_ddp.sh  —  Launch FlowTok-MeanFlow-Sob in DDP on 4 GPUs.
# ------------------------------------------------------------------
# Edit the DATASET, EXTRA_ARGS, or CUDA_VISIBLE_DEVICES lines as needed.
#


# 1) Set the GPUs
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7  # four GPUs

# 2) Figure out how many ranks to launch
if [[ -n "${CUDA_VISIBLE_DEVICES}" ]]; then
  # Count comma-separated entries in CUDA_VISIBLE_DEVICES
  IFS=',' read -ra GPU_ARR <<< "${CUDA_VISIBLE_DEVICES}"
  NPROC=${#GPU_ARR[@]}
else
  # Fallback: ask nvidia-smi how many GPUs are visible
  NPROC=$(nvidia-smi -L | wc -l)
fi
echo "Launching with ${NPROC} processes (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-'<all>'})"

# 3) Ctrl-C handler – kill our entire process tree
cleanup () {
  echo -e "\n  Caught Ctrl-C – terminating children …"
  pkill -TERM -P $$ 2>/dev/null || true
  sleep 2
  pkill -KILL -P $$ 2>/dev/null || true
  exit 130         # 128 + SIGINT
}
trap cleanup INT

# Main command ------------------------------------------------------
torchrun \
  --standalone \
  --nproc_per_node="${NPROC}" \
  flowtok_mean_flow_sob_DDP.py train \
  --dataset flowers_blip_splits \
  --batch 16 \
  --epochs 2000 \
  --wandb online \
  "$@"

# Any extra CLI flags can be appended when you call this script, e.g.
#   ./train_ddp.sh --lr 3e-4 --wandb online
