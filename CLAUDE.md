# RL_Lunar — Claude Code Project Guide

## What this project is

A reinforcement learning project for training agents on the **LunarLander-v2** environment using Stable-Baselines3.

## How to run

```bash
# Install dependencies
.venv\Scripts\pip install -r requirements.txt

# Run the Streamlit web UI (primary entry point)
.venv\Scripts\streamlit run streamlit_app.py

# CLI training (legacy, uses Stable-Baselines3)
.venv\Scripts\python train.py --algo ppo --timesteps 200000 --eval

# Plot CLI training curve
.venv\Scripts\python plot_results.py --log_dir logs/
```

## Project structure

```
RL_Lunar/
├── streamlit_app.py  # ← main web UI (Streamlit + Plotly)
├── dqn.py            # Custom DQN: ReplayBuffer, QNetwork, DQNAgent
├── train.py          # CLI training (Stable-Baselines3 PPO/SAC/DQN)
├── plot_results.py   # Plot SB3 monitor logs
├── requirements.txt  # Python dependencies
└── .venv/            # Virtual environment
```

## Environment

- **Gym env**: `LunarLander-v2` (Box2D, discrete actions)
- **Observation**: 8-dim state (position, velocity, angle, leg contacts)
- **Actions**: 4 discrete (nothing, left engine, main engine, right engine)
- **Reward**: +200 for landing, −100 for crash, fuel penalty

## Algorithms

| Flag | Class | Notes |
|------|-------|-------|
| `ppo` | PPO | Default; on-policy, stable |
| `sac` | SAC | Off-policy, sample-efficient |
| `dqn` | DQN | Off-policy, discrete only |

## Dependencies

- `gymnasium[box2d]` — environment
- `stable-baselines3` — RL algorithms
- `torch` — neural network backend
- `matplotlib` — plotting
