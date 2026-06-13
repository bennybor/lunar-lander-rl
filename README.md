# 🚀 LunarLander DQN Playground

An interactive reinforcement-learning playground for the **LunarLander-v3** environment,
built as a [Streamlit](https://streamlit.io) web app. Train a Deep Q-Network from scratch,
watch it learn in real time, replay its best episodes, and fly the lander yourself.

## Features

- **Train a DQN live** — tune hyperparameters, environment physics, and the full reward
  function from the sidebar, then watch reward/loss/ε curves update as training runs.
- **Configurable physics** — gravity, wind, turbulence, engine power, landing-pad width,
  and the speed/tilt thresholds that define a successful landing.
- **Custom reward shaping** — adjust terminal rewards, per-step shaping weights
  (distance, speed, tilt, leg contact), and fuel costs, all explained on the in-app Info tab.
- **Top-10 episode replays** — the highest-reward episodes are recorded and replayed as an
  animated trajectory with per-step action and reward.
- **Manual play** — a self-contained HTML5-canvas game; fly the lander with the arrow keys
  using the same physics the agent trains on.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\streamlit run streamlit_app.py
```

The app opens at http://localhost:8501.

## Project structure

```
.
├── streamlit_app.py   # main web UI (Streamlit + Plotly + HTML5 canvas)
├── dqn.py             # custom DQN: ReplayBuffer, QNetwork, DQNAgent
├── train.py           # optional CLI training via Stable-Baselines3
├── plot_results.py    # plot SB3 monitor logs
├── requirements.txt   # Python dependencies
└── packages.txt       # system deps for Box2D (Streamlit Cloud build)
```

## The environment

`LunarLander-v3` (Box2D, discrete actions):

- **Observation** — 8-dim: x, y, vx, vy, angle, angular velocity, left-leg contact, right-leg contact
- **Actions** — 4 discrete: do nothing, left engine, main engine, right engine
- **Reward** — terminal ±100 for landing/crash, plus per-step shaping and fuel costs
  (all configurable in the app)

## Deployment

Deployable on [Streamlit Community Cloud](https://streamlit.io/cloud). Box2D requires `swig`
and a build toolchain to compile, which `packages.txt` provides to the cloud builder.

## License

MIT
