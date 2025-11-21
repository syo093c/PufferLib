#!/usr/bin/env python
"""Long-run Breakout trainer with periodic checkpoints and playback."""

import argparse
import os
import time
from pathlib import Path
import sys

import torch

from pufferlib import pufferl


def load_base_config(env_name: str):
    argv = sys.argv
    sys.argv = [argv[0]]
    try:
        return pufferl.load_config(env_name)
    finally:
        sys.argv = argv


def parse_args():
    parser = argparse.ArgumentParser(description="Train Breakout with frequent checkpoints.")
    parser.add_argument("--total-epochs", type=int, default=1_000_000, help="Number of PPO updates.")
    parser.add_argument("--checkpoint-interval", type=int, default=1000, help="Save every N epochs.")
    parser.add_argument("--policy-name", type=str, default="Policy", help="Policy class name (e.g., TransformerPolicy).")
    parser.add_argument("--env-agents", type=int, default=64, help="Agents per env (Breakout num_envs).")
    parser.add_argument("--vec-envs", type=int, default=4, help="Vectorized env replicas.")
    parser.add_argument("--num-workers", type=int, default=None, help="Workers for vector env (defaults to vec-envs).")
    parser.add_argument("--vec-batch-size", type=int, default=None, help="Batch size for vector backend (defaults to vec-envs).")
    parser.add_argument("--backend", choices=["Multiprocessing", "Serial", "PufferEnv"], default="Multiprocessing")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--paddle-penalty", type=float, default=0.0, help="Penalty per paddle collision.")
    parser.add_argument("--life-penalty", type=float, default=0.0, help="Penalty when a ball is lost.")
    parser.add_argument("--bptt-horizon", type=int, default=64)
    parser.add_argument("--train-batch-size", type=int, default=None, help="Trainer batch_size; auto if unset.")
    parser.add_argument("--minibatch-size", type=int, default=None, help="Trainer minibatch_size; auto if unset.")
    parser.add_argument("--policy-hidden", type=int, default=128)
    parser.add_argument("--overwork", action="store_true", help="Allow workers > physical cores.")
    parser.add_argument("--render-after", action="store_true", help="Render with the latest checkpoint after training.")
    parser.add_argument("--render-checkpoint", type=str, default=None, help="Render-only: path to a saved .pt model.")
    parser.add_argument("--render-steps", type=int, default=512)
    parser.add_argument("--render-fps", type=float, default=60.0)
    parser.add_argument("--sample-epochs", type=int, default=None, help="Exit early after N epochs for smoke tests.")
    return parser.parse_args()


def resolve_device(device_flag: str) -> str:
    if device_flag == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_flag


def compute_batches(total_agents: int, horizon: int, batch_size: int | None, minibatch_size: int | None):
    batch = batch_size or total_agents * horizon
    batch = max(batch, total_agents * horizon)
    if batch % horizon != 0:
        batch = (batch // horizon + 1) * horizon

    mini = minibatch_size or min(batch, max(horizon, 8192))
    if mini % horizon != 0:
        mini = (mini // horizon) * horizon
    mini = max(mini, horizon)
    mini = min(mini, batch)
    return batch, mini


def build_config(cli_args):
    cfg = load_base_config("puffer_breakout")
    total_agents = cli_args.env_agents * cli_args.vec_envs
    train_batch, minibatch = compute_batches(total_agents, cli_args.bptt_horizon,
                                             cli_args.train_batch_size, cli_args.minibatch_size)

    cfg["policy_name"] = cli_args.policy_name
    cfg["rnn_name"] = None
    cfg["train"]["use_rnn"] = False
    cfg["train"]["device"] = resolve_device(cli_args.device)
    cfg["train"]["bptt_horizon"] = cli_args.bptt_horizon
    cfg["train"]["batch_size"] = train_batch
    cfg["train"]["minibatch_size"] = minibatch
    cfg["train"]["max_minibatch_size"] = minibatch
    cfg["train"]["checkpoint_interval"] = cli_args.checkpoint_interval
    cfg["train"]["total_timesteps"] = train_batch * cli_args.total_epochs
    cfg["policy"]["hidden_size"] = cli_args.policy_hidden

    cfg["vec"]["backend"] = cli_args.backend
    cfg["vec"]["num_envs"] = cli_args.vec_envs
    cfg["vec"]["num_workers"] = cli_args.num_workers or cli_args.vec_envs
    cfg["vec"]["batch_size"] = cli_args.vec_batch_size or cli_args.vec_envs
    cfg["vec"]["overwork"] = cli_args.overwork

    cfg["env"]["num_envs"] = cli_args.env_agents
    cfg["env"]["paddle_penalty"] = cli_args.paddle_penalty
    cfg["env"]["life_penalty"] = cli_args.life_penalty
    return cfg, total_agents, train_batch


def latest_checkpoint_path(data_dir: str, env_name: str) -> str | None:
    direct = sorted(Path(data_dir).glob(f"{env_name}_*.pt"))
    if direct:
        return str(direct[-1])

    run_dirs = sorted(Path(data_dir).glob(f"{env_name}_*"))
    if not run_dirs:
        return None

    newest_run = run_dirs[-1]
    ckpts = sorted(newest_run.glob("model_*.pt"))
    if not ckpts:
        return None
    return str(ckpts[-1])


def run_training(cfg, total_epochs: int, sample_epochs: int | None):
    vecenv = pufferl.load_env("puffer_breakout", cfg)
    policy = pufferl.load_policy(cfg, vecenv, "puffer_breakout")
    trainer = pufferl.PuffeRL(cfg["train"], vecenv, policy)
    last_logs = {}
    target_epochs = sample_epochs or total_epochs

    try:
        while trainer.epoch < target_epochs:
            trainer.evaluate()
            last_logs = trainer.train() or last_logs
    finally:
        ckpt_path = trainer.close()
    return ckpt_path, last_logs


def render_checkpoint(ckpt_path: str, policy_hidden: int, steps: int, fps: float, device: str):
    from pufferlib.ocean import torch as ocean_torch
    from pufferlib.ocean.breakout import breakout

    env = breakout.Breakout(num_envs=1, render_mode="human")
    policy = ocean_torch.Policy(env, hidden_size=policy_hidden).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    policy.load_state_dict(state_dict, strict=False)
    policy.eval()

    obs, _ = env.reset()
    for _ in range(steps):
        obs_tensor = torch.as_tensor(obs, device=device, dtype=torch.float32)
        with torch.no_grad():
            logits, _ = policy(obs_tensor)
            if isinstance(logits, (list, tuple)):
                act_t = [torch.argmax(x, dim=-1) for x in logits]
                action = torch.stack(act_t, dim=-1).cpu().numpy().squeeze()
            elif isinstance(logits, torch.distributions.Distribution):
                action = logits.mean.detach().cpu().numpy().squeeze()
            else:
                action = torch.argmax(logits, dim=-1).cpu().numpy().squeeze()

        obs, _, _, _, _ = env.step(action)
        env.render()
        if fps > 0:
            time.sleep(1.0 / fps)


def main():
    args = parse_args()

    if args.render_checkpoint:
        ckpt = args.render_checkpoint
    else:
        cfg, total_agents, train_batch = build_config(args)
        print(f"[trainer] device={cfg['train']['device']} total_agents={total_agents} "
              f"batch_size={train_batch} checkpoint_interval={cfg['train']['checkpoint_interval']}")
        ckpt, logs = run_training(cfg, args.total_epochs, args.sample_epochs)
        print(f"[trainer] finished epoch={args.sample_epochs or args.total_epochs} ckpt={ckpt}")
        print(f"[trainer] last logs: {logs}")

    if args.render_after or args.render_checkpoint:
        ckpt = args.render_checkpoint or ckpt
        if not ckpt or not os.path.exists(ckpt):
            raise FileNotFoundError(f"No checkpoint found at {ckpt}")
        render_checkpoint(ckpt, args.policy_hidden, args.render_steps, args.render_fps,
                          resolve_device(args.device))


if __name__ == "__main__":
    main()
