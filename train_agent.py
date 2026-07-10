from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

from test import CarRacing


DEFAULT_SAVE_DIR = Path("models")
DEFAULT_LATEST_CHECKPOINT = DEFAULT_SAVE_DIR / "custom_ppo_latest.pt"
DEFAULT_BEST_CHECKPOINT = DEFAULT_SAVE_DIR / "custom_ppo_best.pt"
DEFAULT_PLOT_PATH = DEFAULT_SAVE_DIR / "custom_ppo_rewards.png"
DEFAULT_TOTAL_TIMESTEPS = 1_000_000
ENV_SIGNATURE = "path_angle_obs_v4"


def parse_args():
    parser = argparse.ArgumentParser(description="Train the custom PPO racing agent on test.py.")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=DEFAULT_TOTAL_TIMESTEPS,
        help="Total environment steps to collect.",
    )
    parser.add_argument("--rollout-steps", type=int, default=2048, help="Steps collected before each PPO update.")
    parser.add_argument("--ppo-epochs", type=int, default=10, help="Gradient passes per rollout.")
    parser.add_argument("--batch-size", type=int, default=256, help="Mini-batch size for PPO updates.")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Adam learning rate.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda.")
    parser.add_argument("--clip-eps", type=float, default=0.2, help="PPO clipping epsilon.")
    parser.add_argument("--value-coef", type=float, default=0.5, help="Value-loss weight.")
    parser.add_argument("--entropy-coef", type=float, default=1e-3, help="Entropy bonus weight.")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Gradient clipping norm.")
    parser.add_argument("--hidden-size", type=int, default=128, help="Hidden size for actor and critic MLPs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Torch device.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_LATEST_CHECKPOINT,
        help="Path for the latest training checkpoint.",
    )
    parser.add_argument(
        "--best-checkpoint",
        type=Path,
        default=DEFAULT_BEST_CHECKPOINT,
        help="Path for the best training checkpoint.",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=DEFAULT_PLOT_PATH,
        help="Path for the saved training curve image.",
    )
    parser.add_argument("--save-every", type=int, default=5, help="Save the latest checkpoint every N updates.")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint if it exists.")
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda":
        return torch.device("cuda")
    if device_name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(render_mode=None):
    return CarRacing(render_mode=render_mode)


def softplus_inverse(values):
    values = torch.as_tensor(values, dtype=torch.float32)
    return torch.log(torch.expm1(values))


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 128):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size

        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.alpha_head = nn.Linear(hidden_size, action_dim)
        self.beta_head = nn.Linear(hidden_size, action_dim)
        self.value_head = nn.Linear(hidden_size, 1)

        self._init_parameters()

    def _init_parameters(self):
        for module in self.backbone:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))
                nn.init.zeros_(module.bias)

        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

        nn.init.zeros_(self.alpha_head.weight)
        nn.init.zeros_(self.beta_head.weight)

        desired_alpha = torch.full((self.action_dim,), 2.0)
        desired_beta = torch.full((self.action_dim,), 2.0)

        if self.action_dim >= 3:
            desired_alpha[0] = 4.0
            desired_beta[0] = 4.0
            desired_alpha[1] = 4.3
            desired_beta[1] = 2.1
            desired_alpha[2] = 1.15
            desired_beta[2] = 10.0

        with torch.no_grad():
            self.alpha_head.bias.copy_(softplus_inverse(desired_alpha - 1.0))
            self.beta_head.bias.copy_(softplus_inverse(desired_beta - 1.0))

    def forward(self, obs):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        features = self.backbone(obs)
        alpha = F.softplus(self.alpha_head(features)) + 1.0
        beta = F.softplus(self.beta_head(features)) + 1.0
        value = self.value_head(features).squeeze(-1)
        return alpha, beta, value

    def get_dist_and_value(self, obs):
        alpha, beta, value = self(obs)
        return Beta(alpha, beta), value

    def act(self, obs, deterministic: bool = False):
        dist, value = self.get_dist_and_value(obs)
        unit_action = dist.mean if deterministic else dist.sample()
        unit_action = unit_action.clamp(1e-4, 1.0 - 1e-4)
        log_prob = dist.log_prob(unit_action).sum(dim=-1)
        env_action = unit_to_env_action(unit_action)
        return env_action, unit_action, log_prob, value


def unit_to_env_action(unit_action: torch.Tensor) -> torch.Tensor:
    env_action = unit_action.clone()
    env_action[..., 0] = env_action[..., 0] * 2.0 - 1.0
    return env_action


def compute_gae(rewards, values, dones, last_value, gamma, gae_lambda):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_advantage = 0.0

    for step in reversed(range(len(rewards))):
        next_value = last_value if step == len(rewards) - 1 else values[step + 1]
        next_nonterminal = 1.0 - dones[step]
        delta = rewards[step] + gamma * next_value * next_nonterminal - values[step]
        last_advantage = delta + gamma * gae_lambda * next_nonterminal * last_advantage
        advantages[step] = last_advantage

    returns = advantages + values
    return advantages, returns


def save_training_plot(reward_history, plot_path: Path, window: int = 20):
    if not reward_history:
        return

    plot_path.parent.mkdir(parents=True, exist_ok=True)

    rewards = np.asarray(reward_history, dtype=np.float32)
    moving_avg = np.array(
        [rewards[max(0, idx - window + 1) : idx + 1].mean() for idx in range(len(rewards))],
        dtype=np.float32,
    )

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Episode Reward", alpha=0.45)
    plt.plot(moving_avg, label=f"Moving Avg ({window})", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Custom PPO Training Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


def checkpoint_payload(
    model,
    optimizer,
    args,
    total_steps,
    episode_count,
    reward_history,
    completion_history,
    best_reward,
    best_completion,
    best_lap_finished,
):
    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "env_signature": ENV_SIGNATURE,
        "obs_dim": model.obs_dim,
        "action_dim": model.action_dim,
        "hidden_size": model.hidden_size,
        "total_steps": total_steps,
        "episode_count": episode_count,
        "reward_history": reward_history,
        "completion_history": completion_history,
        "best_reward": best_reward,
        "best_completion": best_completion,
        "best_lap_finished": best_lap_finished,
        "config": {
            key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
        },
    }


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    args,
    total_steps,
    episode_count,
    reward_history,
    completion_history,
    best_reward,
    best_completion,
    best_lap_finished,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        checkpoint_payload(
            model,
            optimizer,
            args,
            total_steps,
            episode_count,
            reward_history,
            completion_history,
            best_reward,
            best_completion,
            best_lap_finished,
        ),
        path,
    )


def load_checkpoint(path: Path, device: torch.device):
    return torch.load(path, map_location=device)


def validate_checkpoint(checkpoint, path: Path):
    checkpoint_signature = checkpoint.get("env_signature")
    if checkpoint_signature != ENV_SIGNATURE:
        raise ValueError(
            f"Checkpoint {path} is incompatible with the current environment/observation setup. "
            "Start a fresh training run instead of resuming."
        )


def build_model_from_checkpoint(checkpoint, device: torch.device):
    model = ActorCritic(
        obs_dim=checkpoint["obs_dim"],
        action_dim=checkpoint["action_dim"],
        hidden_size=checkpoint.get("hidden_size", 128),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model


def training_status(
    update_idx,
    total_steps,
    total_timesteps,
    episode_count,
    reward_history,
    completion_history,
    best_reward,
    best_completion,
    policy_losses,
    value_losses,
):
    last_reward = reward_history[-1] if reward_history else 0.0
    avg10 = float(np.mean(reward_history[-10:])) if reward_history else 0.0
    last_completion = completion_history[-1] if completion_history else 0.0
    avg_completion = float(np.mean(completion_history[-10:])) if completion_history else 0.0
    avg_policy = float(np.mean(policy_losses)) if policy_losses else 0.0
    avg_value = float(np.mean(value_losses)) if value_losses else 0.0
    print(
        f"Update {update_idx:03d} | Steps {total_steps}/{total_timesteps} | "
        f"Episodes {episode_count} | Last {last_reward:.1f} | Avg10 {avg10:.1f} | "
        f"LastComp {last_completion:.1%} | AvgComp {avg_completion:.1%} | "
        f"Best {best_reward:.1f} | BestComp {best_completion:.1%} | "
        f"Policy {avg_policy:.4f} | Value {avg_value:.4f}"
    )


def main():
    args = parse_args()
    device = resolve_device(args.device)
    set_seed(args.seed)

    env = make_env(render_mode=None)
    try:
        initial_obs, _ = env.reset(seed=args.seed)
        obs_dim = int(initial_obs.shape[0])
        action_dim = int(env.action_space.shape[0])

        if args.resume and args.checkpoint.exists():
            checkpoint = load_checkpoint(args.checkpoint, device)
            validate_checkpoint(checkpoint, args.checkpoint)
            model = build_model_from_checkpoint(checkpoint, device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
            optimizer_state = checkpoint.get("optimizer_state")
            if optimizer_state is not None:
                optimizer.load_state_dict(optimizer_state)
            total_steps = int(checkpoint.get("total_steps", 0))
            episode_count = int(checkpoint.get("episode_count", 0))
            reward_history = list(checkpoint.get("reward_history", []))
            completion_history = list(checkpoint.get("completion_history", []))
            best_reward = float(checkpoint.get("best_reward", -np.inf))
            best_completion = float(checkpoint.get("best_completion", 0.0))
            best_lap_finished = bool(checkpoint.get("best_lap_finished", False))
            print(f"Resuming training from {args.checkpoint}")
        else:
            if args.resume:
                print(f"No checkpoint found at {args.checkpoint}, starting fresh.")
            model = ActorCritic(obs_dim=obs_dim, action_dim=action_dim, hidden_size=args.hidden_size).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
            total_steps = 0
            episode_count = 0
            reward_history = []
            completion_history = []
            best_reward = -np.inf
            best_completion = 0.0
            best_lap_finished = False

        current_obs = initial_obs
        current_episode_reward = 0.0
        update_idx = 0

        while total_steps < args.total_timesteps:
            rollout_size = min(args.rollout_steps, args.total_timesteps - total_steps)

            states = np.zeros((rollout_size, obs_dim), dtype=np.float32)
            unit_actions = np.zeros((rollout_size, action_dim), dtype=np.float32)
            rewards = np.zeros(rollout_size, dtype=np.float32)
            dones = np.zeros(rollout_size, dtype=np.float32)
            values = np.zeros(rollout_size, dtype=np.float32)
            log_probs = np.zeros(rollout_size, dtype=np.float32)

            for step in range(rollout_size):
                states[step] = current_obs
                obs_tensor = torch.as_tensor(current_obs, dtype=torch.float32, device=device).unsqueeze(0)

                with torch.no_grad():
                    env_action, unit_action, log_prob, value = model.act(obs_tensor, deterministic=False)

                action_np = env_action.squeeze(0).cpu().numpy().astype(np.float32)
                next_obs, reward, terminated, truncated, info = env.step(action_np)
                done = terminated or truncated

                unit_actions[step] = unit_action.squeeze(0).cpu().numpy()
                rewards[step] = float(reward)
                dones[step] = float(done)
                values[step] = float(value.item())
                log_probs[step] = float(log_prob.item())

                current_episode_reward += float(reward)
                total_steps += 1

                if done:
                    episode_count += 1
                    reward_history.append(current_episode_reward)

                    episode_completion = float(
                        info.get(
                            "completion_pct",
                            getattr(env, "tile_visited_count", 0) / max(1, len(getattr(env, "track", []))),
                        )
                    )
                    completion_history.append(episode_completion)
                    lap_finished = bool(info.get("lap_finished", False))

                    is_best_run = False
                    if lap_finished and not best_lap_finished:
                        is_best_run = True
                    elif lap_finished == best_lap_finished:
                        if episode_completion > best_completion + 1e-9:
                            is_best_run = True
                        elif abs(episode_completion - best_completion) <= 1e-9 and current_episode_reward > best_reward:
                            is_best_run = True

                    if is_best_run:
                        best_reward = current_episode_reward
                        best_completion = episode_completion
                        best_lap_finished = lap_finished
                        save_checkpoint(
                            args.best_checkpoint,
                            model,
                            optimizer,
                            args,
                            total_steps,
                            episode_count,
                            reward_history,
                            completion_history,
                            best_reward,
                            best_completion,
                            best_lap_finished,
                        )

                    current_obs, _ = env.reset()
                    current_episode_reward = 0.0
                else:
                    current_obs = next_obs

            if dones[-1] == 1.0:
                last_value = 0.0
            else:
                obs_tensor = torch.as_tensor(current_obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    _, last_value_tensor = model.get_dist_and_value(obs_tensor)
                last_value = float(last_value_tensor.item())

            advantages, returns = compute_gae(
                rewards=rewards,
                values=values,
                dones=dones,
                last_value=last_value,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
            )

            states_tensor = torch.as_tensor(states, dtype=torch.float32, device=device)
            unit_actions_tensor = torch.as_tensor(unit_actions, dtype=torch.float32, device=device)
            old_log_probs_tensor = torch.as_tensor(log_probs, dtype=torch.float32, device=device)
            advantages_tensor = torch.as_tensor(advantages, dtype=torch.float32, device=device)
            returns_tensor = torch.as_tensor(returns, dtype=torch.float32, device=device)

            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (
                advantages_tensor.std(unbiased=False) + 1e-8
            )

            batch_size = min(args.batch_size, rollout_size)
            policy_losses = []
            value_losses = []

            for _ in range(args.ppo_epochs):
                indices = np.random.permutation(rollout_size)

                for start in range(0, rollout_size, batch_size):
                    batch_idx = indices[start : start + batch_size]

                    batch_states = states_tensor[batch_idx]
                    batch_unit_actions = unit_actions_tensor[batch_idx]
                    batch_old_log_probs = old_log_probs_tensor[batch_idx]
                    batch_advantages = advantages_tensor[batch_idx]
                    batch_returns = returns_tensor[batch_idx]

                    dist, value = model.get_dist_and_value(batch_states)
                    new_log_probs = dist.log_prob(batch_unit_actions).sum(dim=-1)
                    entropy = dist.entropy().sum(dim=-1).mean()

                    ratio = torch.exp(new_log_probs - batch_old_log_probs)
                    surrogate_1 = ratio * batch_advantages
                    surrogate_2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * batch_advantages

                    policy_loss = -torch.min(surrogate_1, surrogate_2).mean()
                    value_loss = F.mse_loss(value, batch_returns)
                    loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()

                    policy_losses.append(float(policy_loss.item()))
                    value_losses.append(float(value_loss.item()))

            update_idx += 1
            training_status(
                update_idx=update_idx,
                total_steps=total_steps,
                total_timesteps=args.total_timesteps,
                episode_count=episode_count,
                reward_history=reward_history,
                completion_history=completion_history,
                best_reward=best_reward if best_reward != -np.inf else 0.0,
                best_completion=best_completion,
                policy_losses=policy_losses,
                value_losses=value_losses,
            )

            if update_idx % args.save_every == 0 or total_steps >= args.total_timesteps:
                save_checkpoint(
                    args.checkpoint,
                    model,
                    optimizer,
                    args,
                    total_steps,
                    episode_count,
                    reward_history,
                    completion_history,
                    best_reward,
                    best_completion,
                    best_lap_finished,
                )
                if not args.best_checkpoint.exists():
                    save_checkpoint(
                        args.best_checkpoint,
                        model,
                        optimizer,
                        args,
                        total_steps,
                        episode_count,
                        reward_history,
                        completion_history,
                        best_reward,
                        best_completion,
                        best_lap_finished,
                    )
                save_training_plot(reward_history, args.plot_path)

        save_checkpoint(
            args.checkpoint,
            model,
            optimizer,
            args,
            total_steps,
            episode_count,
            reward_history,
            completion_history,
            best_reward,
            best_completion,
            best_lap_finished,
        )
        if not args.best_checkpoint.exists():
            save_checkpoint(
                args.best_checkpoint,
                model,
                optimizer,
                args,
                total_steps,
                episode_count,
                reward_history,
                completion_history,
                best_reward,
                best_completion,
                best_lap_finished,
            )
        save_training_plot(reward_history, args.plot_path)

        print(f"Latest checkpoint: {args.checkpoint}")
        print(f"Best checkpoint: {args.best_checkpoint}")
        print(f"Training curve: {args.plot_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
