#!/bin/bash

# Breakout training experiments with different epoch counts
# Tests: 2000, 4000, 8000 epochs

set -e  # Exit on error

# Common parameters
POLICY_NAME="Policy"
CHECKPOINT_INTERVAL=50
ENV_AGENTS=256
VEC_ENVS=8
NUM_WORKERS=8
VEC_BATCH_SIZE=8
BACKEND="Multiprocessing"
DEVICE="auto"
BPTT_HORIZON=64
TRAIN_BATCH_SIZE=32768
MINIBATCH_SIZE=8192
PADDLE_PENALTY=0.5
LIFE_PENALTY=5
USE_RNN="--use-rnn"
RNN_NAME="Recurrent"
RNN_INPUT_SIZE=1024
RNN_HIDDEN_SIZE=1024
POLICY_HIDDEN=1024
WANDB_PROJECT="pufferlib-breakout"
WANDB_GROUP="rnn-experiment"

# Epoch configurations to test
EPOCHS=(2000 4000 8000)

for EPOCH in "${EPOCHS[@]}"; do
  echo "========================================="
  echo "Starting experiment with ${EPOCH} epochs"
  echo "========================================="

  RUN_NAME="breakout_ep${EPOCH}_rnn1024"

  UV_NO_SYNC=1 uv run python scripts/train_breakout_long.py \
    --policy-name ${POLICY_NAME} \
    --total-epochs ${EPOCH} \
    --checkpoint-interval ${CHECKPOINT_INTERVAL} \
    --env-agents ${ENV_AGENTS} \
    --vec-envs ${VEC_ENVS} \
    --num-workers ${NUM_WORKERS} \
    --vec-batch-size ${VEC_BATCH_SIZE} \
    --backend ${BACKEND} \
    --device ${DEVICE} \
    --bptt-horizon ${BPTT_HORIZON} \
    --train-batch-size ${TRAIN_BATCH_SIZE} \
    --minibatch-size ${MINIBATCH_SIZE} \
    --paddle-penalty ${PADDLE_PENALTY} \
    --life-penalty ${LIFE_PENALTY} \
    ${USE_RNN} \
    --rnn-name ${RNN_NAME} \
    --rnn-input-size ${RNN_INPUT_SIZE} \
    --rnn-hidden-size ${RNN_HIDDEN_SIZE} \
    --policy-hidden ${POLICY_HIDDEN} \
    --wandb \
    --wandb-project ${WANDB_PROJECT} \
    --wandb-group ${WANDB_GROUP} \
    --wandb-name ${RUN_NAME} \
    --wandb-tags rnn epoch-${EPOCH}

  echo "Completed experiment with ${EPOCH} epochs"
  echo ""
done

echo "========================================="
echo "All experiments completed!"
echo "========================================="
