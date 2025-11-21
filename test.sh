UV_NO_SYNC=1 uv run python scripts/train_breakout_long.py \
  --policy-name Policy --total-epochs 2000 --checkpoint-interval 50 \
  --env-agents 256 --vec-envs 8 --num-workers 8 --vec-batch-size 8 \
  --backend Multiprocessing --device auto \
  --bptt-horizon 64 --train-batch-size 32768 --minibatch-size 8192 \
  --paddle-penalty 0.3 --life-penalty 1.0
