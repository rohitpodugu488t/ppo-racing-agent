import argparse
from pathlib import Path

import numpy as np
import torch

from train_agent import (
    DEFAULT_BEST_CHECKPOINT,
    DEFAULT_LATEST_CHECKPOINT,
    build_model_from_checkpoint,
    load_checkpoint,
    make_env,
    resolve_device,
    validate_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Test the trained racing agent from train_agent.py.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_BEST_CHECKPOINT,
        help="Checkpoint path to load.",
    )
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to run.")
    parser.add_argument(
        "--render-mode",
        default="human",
        choices=["human", "rgb_array", "state_pixels", "none"],
        help="Render mode to use while evaluating.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Torch device.")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions instead of using deterministic policy output.",
    )
    return parser.parse_args()


def resolve_checkpoint_path(path: Path):
    if path.exists():
        return path
    if path == DEFAULT_BEST_CHECKPOINT and DEFAULT_LATEST_CHECKPOINT.exists():
        print(f"{path} not found, using {DEFAULT_LATEST_CHECKPOINT} instead.")
        return DEFAULT_LATEST_CHECKPOINT
    raise FileNotFoundError(f"Checkpoint not found: {path}")


def main():
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path, device)
    validate_checkpoint(checkpoint, checkpoint_path)
    model = build_model_from_checkpoint(checkpoint, device)
    model.eval()

    render_mode = None if args.render_mode == "none" else args.render_mode
    env = make_env(render_mode=render_mode)

    try:
        rewards = []
        completions = []

        for episode_idx in range(1, args.episodes + 1):
            obs, _ = env.reset()
            total_reward = 0.0
            terminated = False
            truncated = False
            final_info = {}

            while not (terminated or truncated):
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action, _, _, _ = model.act(obs_tensor, deterministic=not args.stochastic)

                action_np = action.squeeze(0).cpu().numpy().astype(np.float32)
                obs, reward, terminated, truncated, info = env.step(action_np)
                total_reward += float(reward)
                final_info = info

            completion = float(final_info.get("completion_pct", 0.0))
            reason = final_info.get("termination_reason", "unknown")
            lap_finished = bool(final_info.get("lap_finished", False))
            rewards.append(total_reward)
            completions.append(completion)

            print(
                f"Episode {episode_idx}: reward={total_reward:.2f} | "
                f"completion={completion:.1%} | lap_finished={lap_finished} | reason={reason}"
            )

        print(
            f"Average: reward={np.mean(rewards):.2f} | completion={np.mean(completions):.1%} | "
            f"best_completion={np.max(completions):.1%}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
