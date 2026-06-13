"""
RL_Lunar — LunarLander-v2 training entry point.
Usage:
    python train.py               # train with default settings
    python train.py --algo ppo    # choose algorithm: ppo | sac | dqn
    python train.py --timesteps 500000
"""
import argparse
import os

import gymnasium as gym
import numpy as np


def make_env(render: bool = False):
    render_mode = "human" if render else None
    return gym.make("LunarLander-v2", render_mode=render_mode)


def train(algo: str, timesteps: int, save_path: str):
    from stable_baselines3 import PPO, SAC, DQN

    env = make_env()
    print(f"Training {algo.upper()} on LunarLander-v2 for {timesteps:,} timesteps …")

    algo_map = {
        "ppo": (PPO, dict(policy="MlpPolicy", verbose=1)),
        "sac": (SAC, dict(policy="MlpPolicy", verbose=1)),
        "dqn": (DQN, dict(policy="MlpPolicy", verbose=1)),
    }
    if algo not in algo_map:
        raise ValueError(f"Unknown algo '{algo}'. Choose from: {list(algo_map)}")

    AlgoClass, kwargs = algo_map[algo]
    model = AlgoClass(env=env, **kwargs)
    model.learn(total_timesteps=timesteps)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    model.save(save_path)
    print(f"Model saved → {save_path}")
    env.close()
    return model


def evaluate(model, n_episodes: int = 10, render: bool = False):
    env = make_env(render=render)
    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        total = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        rewards.append(total)
        print(f"  Episode {ep+1:>3}: {total:+.1f}")
    env.close()
    mean, std = np.mean(rewards), np.std(rewards)
    print(f"\nMean reward: {mean:.1f} ± {std:.1f}")
    return rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default="ppo", choices=["ppo", "sac", "dqn"])
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--eval", action="store_true", help="Evaluate after training")
    parser.add_argument("--render", action="store_true", help="Render evaluation")
    parser.add_argument("--save", default="models/lunar_model")
    args = parser.parse_args()

    model = train(args.algo, args.timesteps, args.save)
    if args.eval:
        evaluate(model, render=args.render)
