"""
LunarLander DQN — Streamlit web UI
Train a DQN agent on LunarLander-v3 and watch it play, or fly it yourself.
"""
import os
import time
import base64
import functools
import threading
import queue

import numpy as np
import gymnasium as gym
import gymnasium.envs.box2d.lunar_lander as _ll_module
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dqn import DQNAgent


@functools.lru_cache(maxsize=1)
def _dog_data_uri():
    """Return the mascot image as a base64 data URI. Returns '' if assets/dog.png
    is missing, so callers can fall back gracefully (e.g. to an emoji)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "dog.png")
    try:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return f"data:image/png;base64,{b64}"
    except OSError:
        return ""

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LunarLander DQN",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0f1e 0%, #0f172a 100%);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea {
    color: #0f172a !important;
    background: #f8fafc !important;
}
.sec-training { background:#0f2942; border-left:3px solid #38bdf8;
                border-radius:6px; padding:8px 12px; margin:6px 0 10px; }
.sec-forces   { background:#1a1042; border-left:3px solid #a78bfa;
                border-radius:6px; padding:8px 12px; margin:6px 0 10px; }
.sec-rewards  { background:#0f2e1a; border-left:3px solid #4ade80;
                border-radius:6px; padding:8px 12px; margin:6px 0 10px; }
.sec-dqn      { background:#2a1a0a; border-left:3px solid #fb923c;
                border-radius:6px; padding:8px 12px; margin:6px 0 10px; }
.sec-replay   { background:#1a1a2e; border-left:3px solid #f472b6;
                border-radius:6px; padding:8px 12px; margin:6px 0 10px; }
.sec-play     { background:#0f1f2e; border-left:3px solid #22d3ee;
                border-radius:6px; padding:8px 12px; margin:6px 0 10px; }
.sec-label    { font-size:0.85rem; font-weight:700; letter-spacing:0.05em;
                margin-bottom:2px; }
.metric-card  { background:#1e293b; border-radius:8px; padding:12px 18px;
                text-align:center; color:#e2e8f0; }
.metric-label { font-size:0.75rem; color:#94a3b8; margin-bottom:4px; }
.metric-value { font-size:1.5rem; font-weight:700; color:#38bdf8; }
</style>
""", unsafe_allow_html=True)


# ── session state init ────────────────────────────────────────────────────────
def _init_state():
    defaults = dict(
        training=False, done=False,
        episode_rewards=[], episode_lengths=[], losses=[], epsilons=[],
        engine_fires=[],   # list of [idle,left,main,right] per episode
        best_reward=-np.inf, last_episode=0, last_display_episode=0, agent=None,
        stop_flag=False, log_q=None, recorded_episodes=[],
        # manual play
        play_game_id=0,
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    _dog = _dog_data_uri()
    if _dog:
        st.markdown(
            f'<h1><img src="{_dog}" style="height:2em;width:2em;'
            f'vertical-align:-0.5em;border-radius:6px;margin-right:0.35em">'
            f'LunarLander DQN</h1>',
            unsafe_allow_html=True)
    else:
        st.title("🚀 LunarLander DQN")

    with st.expander("🎯 Training run", expanded=True):
        total_episodes = st.slider("Episodes", 100, 10000, 600, 100)
        max_steps      = st.slider("Max steps / episode", 200, 1000, 500, 50)

    with st.expander("🌪️ Environment forces", expanded=False):
        gravity          = st.slider("Gravity", -11.9, -0.5, -1.6, 0.1, format="%.1f")
        enable_wind      = st.toggle("Enable wind", value=False)
        wind_power       = st.slider("Wind power", 0.0, 20.0, 15.0, 0.5,
                                     disabled=not enable_wind, format="%.1f")
        turbulence_power = st.slider("Turbulence power", 0.0, 2.0, 1.5, 0.1,
                                     disabled=not enable_wind, format="%.1f")

    with st.expander("🚀 Engine thrust", expanded=False):
        main_engine_power = st.slider("Main engine power", 1.0, 30.0, 13.0, 0.5, format="%.1f",
                                      help="Default: 13.0")
        side_engine_power = st.slider("Side engine power", 0.1,  3.0,  0.6, 0.05, format="%.2f",
                                      help="Default: 0.6")

    with st.expander("🛬 Landing rules", expanded=False):
        pad_width = st.slider("Landing pad width", 0.5, 8.0, 2.0, 0.5, format="%.1f",
                              help="Width of the landing pad in game units (default 2)")
        max_land_speed = st.slider("Max landing speed", 0.01, 0.50, 0.09, 0.01, format="%.2f",
                                   help="Combined speed limit at touchdown (default 0.09)")
        max_land_angle = st.slider("Max landing tilt (°)", 1, 45, 13, 1,
                                   help="Maximum tilt angle at touchdown in degrees (default 13°)")

    with st.expander("💰 Reward shaping", expanded=False):
        st.markdown("**🏁 Terminal rewards**")
        land_bonus    = st.slider("Landing bonus",   0, 300,  100, 10,
                                  help="One-time reward when lander comes to rest on the pad (default +100)")
        crash_penalty = st.slider("Crash penalty", -300, 0, -100, 10,
                                  help="One-time reward on crash or out-of-bounds (default −100)")
        st.markdown("**⚖️ Per-step shaping weights**")
        st.caption("Applied as Δ(shaping) each step — higher = stronger pull toward that goal.")
        w_pos   = st.slider("Distance-to-pad penalty",  0, 300, 100, 5,
                            help="Penalises √(x²+y²) — keeps lander above the pad (default 100)")
        w_vel   = st.slider("Speed penalty",            0, 300, 100, 5,
                            help="Penalises √(vx²+vy²) — rewards slow approach (default 100)")
        w_angle = st.slider("Angle penalty",            0, 300, 100, 5,
                            help="Penalises |tilt| — rewards keeping vertical (default 100)")
        w_leg   = st.slider("Leg-contact bonus",        0,  50,  10, 1,
                            help="Reward per leg touching the ground each step (default 10)")
        st.markdown("**⛽ Fuel cost**")
        fuel_main = st.slider("Main engine fuel cost", 0.0, 1.0, 0.30, 0.01, format="%.2f",
                              help="Subtracted per step when main engine fires (default 0.30)")
        fuel_side = st.slider("Side engine fuel cost", 0.0, 0.2, 0.03, 0.005, format="%.3f",
                              help="Subtracted per step when side engine fires (default 0.03)")
        st.caption("Defaults reproduce the original LunarLander-v3 spec.")

    with st.expander("🧠 DQN hyperparameters", expanded=False):
        lr           = st.select_slider("Learning rate", [1e-5, 5e-5, 1e-4, 5e-4, 1e-3], value=1e-4,
                                        format_func=lambda x: f"{x:.0e}")
        gamma        = st.slider("Discount γ", 0.90, 0.999, 0.99, 0.001, format="%.3f")
        eps_start    = st.slider("ε start", 0.5, 1.0, 1.0, 0.05)
        eps_end      = st.slider("ε end", 0.01, 0.2, 0.05, 0.01)
        eps_decay    = st.select_slider("ε decay (steps)",
                                        [5_000, 10_000, 25_000, 50_000, 100_000], value=50_000)
        hidden       = st.select_slider("Hidden units", [64, 128, 256, 512], value=128)
        batch_size   = st.select_slider("Batch size", [32, 64, 128, 256], value=64)
        buffer_size  = st.select_slider("Replay buffer",
                                        [10_000, 25_000, 50_000, 100_000], value=50_000)
        target_update = st.select_slider("Target update (steps)",
                                         [500, 1_000, 2_000, 5_000], value=1_000)
        train_freq    = st.select_slider("Train every N steps", [1, 2, 4, 8], value=4,
                                         help="Run a gradient update once per N environment steps. "
                                              "4 is standard DQN practice and ~3-4× faster than 1.")
        st.markdown("**🎲 Replay sampling**")
        use_per = st.toggle("Prioritized replay (PER)", value=False,
                            help="Sample transitions with high TD-error more often, instead of "
                                 "uniformly. Helps most when rewards are sparse; adds some overhead.")
        per_alpha = st.slider("PER α (prioritization)", 0.0, 1.0, 0.6, 0.05,
                              disabled=not use_per,
                              help="0 = uniform sampling, 1 = fully proportional to TD-error.")
        per_beta_start = st.slider("PER β start (IS correction)", 0.0, 1.0, 0.4, 0.05,
                                   disabled=not use_per,
                                   help="Importance-sampling correction strength; annealed to 1.0 "
                                        "by the end of training.")

    st.markdown("---")
    col1, col2 = st.columns(2)
    start_btn = col1.button("▶ Train", use_container_width=True,
                             disabled=st.session_state.training)
    stop_btn  = col2.button("⏹ Stop",  use_container_width=True,
                             disabled=not st.session_state.training)
    if st.button("🔄 Reset", use_container_width=True,
                  disabled=st.session_state.training):
        for k in ["episode_rewards","episode_lengths","losses","epsilons","engine_fires",
                  "_snap_rewards","_snap_lengths","_snap_losses","_snap_epsilons","_snap_fires",
                  "done","agent","best_reward","last_episode","last_display_episode","stop_flag",
                  "log_q","recorded_episodes","_chart_fig","_chart_fig_count"]:
            if k in ("episode_rewards","episode_lengths","losses","epsilons","engine_fires","recorded_episodes",
                     "_snap_rewards","_snap_lengths","_snap_losses","_snap_epsilons","_snap_fires"):
                st.session_state[k] = []
            elif k in ("done","stop_flag"):
                st.session_state[k] = False
            elif k == "best_reward":
                st.session_state[k] = -np.inf
            elif k == "last_episode":
                st.session_state[k] = 0
            else:
                st.session_state[k] = None
        st.rerun()


# ── training thread ───────────────────────────────────────────────────────────
def _make_env(cfg):
    _ll_module.MAIN_ENGINE_POWER = cfg["main_engine_power"]
    _ll_module.SIDE_ENGINE_POWER = cfg["side_engine_power"]
    return gym.make(
        "LunarLander-v3",
        gravity=cfg["gravity"],
        enable_wind=cfg["enable_wind"],
        wind_power=cfg["wind_power"],
        turbulence_power=cfg["turbulence_power"],
    )


def _train_loop(cfg: dict, log_q: queue.Queue):
    env = _make_env(cfg)
    obs_dim  = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = DQNAgent(
        obs_dim=obs_dim, n_actions=n_actions,
        lr=cfg["lr"], gamma=cfg["gamma"],
        eps_start=cfg["eps_start"], eps_end=cfg["eps_end"], eps_decay=cfg["eps_decay"],
        buffer_size=cfg["buffer_size"], batch_size=cfg["batch_size"],
        target_update_freq=cfg["target_update"], hidden=cfg["hidden"],
        use_per=cfg["use_per"], per_alpha=cfg["per_alpha"],
        per_beta_start=cfg["per_beta_start"],
        per_beta_frames=cfg["total_episodes"] * cfg["max_steps"],
    )
    log_q.put(("agent", agent))
    top_rewards = []   # rewards of the episodes sent as replays (keep top 10)
    global_step = 0    # for train_freq cadence across episode boundaries

    for ep in range(1, cfg["total_episodes"] + 1):
        if cfg["stop_flag_ref"][0]:
            break

        obs, _info = env.reset()
        total_reward  = 0.0
        ep_losses     = []
        step          = 0
        record_this   = True   # record every episode; only top-10 are kept
        traj          = []
        prev_shaping  = None
        fires = [0, 0, 0, 0]   # idle, left, main, right

        for step in range(cfg["max_steps"]):
            action = agent.select_action(obs)
            next_obs, _env_reward, terminated, truncated, _info = env.step(action)
            done = terminated or truncated

            if terminated:
                # terminal reward uses user sliders; env gives exactly ±100
                reward = cfg["land_bonus"] if _env_reward >= 90 else cfg["crash_penalty"]
                prev_shaping = None
            else:
                # recompute shaping from scratch with user weights
                x, y, vx, vy, ang, _, ll, rl = (float(next_obs[i]) for i in range(8))
                shaping = (
                    - cfg["w_pos"]   * np.sqrt(x*x + y*y)
                    - cfg["w_vel"]   * np.sqrt(vx*vx + vy*vy)
                    - cfg["w_angle"] * abs(ang)
                    + cfg["w_leg"]   * ll
                    + cfg["w_leg"]   * rl
                )
                reward = (shaping - prev_shaping) if prev_shaping is not None else 0.0
                prev_shaping = shaping
                # fuel cost
                m_fire = 1 if action == 2 else 0
                s_fire = 1 if action in (1, 3) else 0
                reward -= cfg["fuel_main"] * m_fire + cfg["fuel_side"] * s_fire

            fires[action] += 1
            agent.push(obs, action, reward, next_obs, done)
            global_step += 1
            if global_step % cfg["train_freq"] == 0:
                loss = agent.optimize()
                if loss is not None:
                    ep_losses.append(loss)
            if record_this:
                traj.append((float(obs[0]), float(obs[1]), float(obs[4]),
                             int(obs[6]), int(obs[7]), int(action), float(reward)))
            obs = next_obs
            total_reward += reward
            if done:
                break

        avg_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
        log_q.put(("step", {"ep": ep, "reward": total_reward,
                            "length": step + 1, "loss": avg_loss,
                            "epsilon": agent.epsilon,
                            "fires": fires}))
        # keep only the 10 highest-reward episodes as replays
        if traj and (len(top_rewards) < 10 or total_reward > min(top_rewards)):
            top_rewards.append(total_reward)
            top_rewards.sort(reverse=True)
            del top_rewards[10:]
            log_q.put(("traj", {"ep": ep, "reward": total_reward, "traj": traj}))

    env.close()
    log_q.put(("done", None))


# ── start / stop ──────────────────────────────────────────────────────────────
if start_btn:
    for k in ["episode_rewards","episode_lengths","losses","epsilons","engine_fires","recorded_episodes",
              "_snap_rewards","_snap_lengths","_snap_losses","_snap_epsilons","_snap_fires"]:
        st.session_state[k] = []
    st.session_state.update(best_reward=-np.inf, last_episode=0, last_display_episode=0,
                            done=False, training=True, stop_flag=False,
                            _chart_fig=None, _chart_fig_count=None)
    log_q = queue.Queue()
    st.session_state.log_q = log_q
    stop_flag_ref = [False]
    st.session_state._stop_ref = stop_flag_ref
    cfg = dict(
        lr=lr, gamma=gamma, eps_start=eps_start, eps_end=eps_end,
        eps_decay=eps_decay, buffer_size=buffer_size, batch_size=batch_size,
        target_update=target_update, hidden=hidden,
        total_episodes=total_episodes, max_steps=max_steps, train_freq=train_freq,
        use_per=use_per, per_alpha=per_alpha, per_beta_start=per_beta_start,
        gravity=gravity, enable_wind=enable_wind,
        wind_power=wind_power, turbulence_power=turbulence_power,
        main_engine_power=main_engine_power, side_engine_power=side_engine_power,
        land_bonus=land_bonus, crash_penalty=crash_penalty,
        w_pos=w_pos, w_vel=w_vel, w_angle=w_angle, w_leg=w_leg,
        fuel_main=fuel_main, fuel_side=fuel_side,
        stop_flag_ref=stop_flag_ref,
    )
    threading.Thread(target=_train_loop, args=(cfg, log_q), daemon=True).start()
    st.rerun()

if stop_btn and st.session_state.training:
    if hasattr(st.session_state, "_stop_ref"):
        st.session_state._stop_ref[0] = True


# ── drain training queue ──────────────────────────────────────────────────────
# Drain whenever the queue exists — even if training just finished this cycle,
# so we never miss episodes that arrived before the "done" sentinel.
if st.session_state.log_q is not None:
    q = st.session_state.log_q
    try:
        while True:
            msg_type, payload = q.get_nowait()
            if msg_type == "agent":
                st.session_state.agent = payload
            elif msg_type == "step":
                st.session_state.last_episode = payload["ep"]
                st.session_state.episode_rewards.append(payload["reward"])
                st.session_state.episode_lengths.append(payload["length"])
                st.session_state.losses.append(payload["loss"])
                st.session_state.epsilons.append(payload["epsilon"])
                st.session_state.engine_fires.append(payload.get("fires", [0,0,0,0]))
                if payload["reward"] > st.session_state.best_reward:
                    st.session_state.best_reward = payload["reward"]
            elif msg_type == "traj":
                st.session_state.recorded_episodes.append(payload)
                st.session_state.recorded_episodes.sort(key=lambda r: r["reward"], reverse=True)
                del st.session_state.recorded_episodes[10:]
            elif msg_type == "done":
                st.session_state.training = False
                st.session_state.done = True
                st.session_state.log_q = None   # stop draining after this run
    except queue.Empty:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
# ── lander drawing helpers ────────────────────────────────────────────────────
def _rotate(pts, cx, cy, angle):
    c, s = np.cos(angle), np.sin(angle)
    return [(cx + p[0]*c - p[1]*s, cy + p[0]*s + p[1]*c) for p in pts]

def _poly_xy(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return xs, ys

def _make_play_fig(obs, traj, p_done, last_action):
    cx, cy, angle = float(obs[0]), float(obs[1]), float(obs[4])

    # lander geometry (local coords)
    body   = [(-0.055, 0.08), (0.055, 0.08), (0.055, -0.015), (-0.055, -0.015)]
    l_leg  = [(-0.055, -0.015), (-0.13, -0.09)]
    r_leg  = [( 0.055, -0.015), ( 0.13, -0.09)]
    l_foot = [(-0.13, -0.09),   (-0.16, -0.09)]
    r_foot = [( 0.13, -0.09),   ( 0.16, -0.09)]

    body_r   = _rotate(body,   cx, cy, angle)
    l_leg_r  = _rotate(l_leg,  cx, cy, angle)
    r_leg_r  = _rotate(r_leg,  cx, cy, angle)
    l_foot_r = _rotate(l_foot, cx, cy, angle)
    r_foot_r = _rotate(r_foot, cx, cy, angle)

    # flame geometry (local coords, grown with action)
    FLAME_MAIN  = [(-0.03, -0.015), (0.03, -0.015),
                   (0.018, -0.16),  (0.0,  -0.22), (-0.018, -0.16)]
    FLAME_LEFT  = [(0.055,  0.03),  (0.055, -0.01),
                   (0.18,  -0.01),  (0.24,  0.01),  (0.18,   0.03)]
    FLAME_RIGHT = [(-0.055, 0.03),  (-0.055, -0.01),
                   (-0.18, -0.01),  (-0.24,  0.01), (-0.18,   0.03)]

    fig = go.Figure()

    # terrain background strips
    for y0, y1, col in [(-0.15, 0.0, "#0a1628"), (0.0, 2.0, "#0d1b2a")]:
        fig.add_shape(type="rect", x0=-1.6, x1=1.6, y0=y0, y1=y1,
                      fillcolor=col, line_width=0, layer="below")

    # ground bumps (static decorations)
    for gx, gw in [(-1.1, 0.25), (-0.6, 0.15), (0.55, 0.20), (1.05, 0.30)]:
        fig.add_shape(type="rect", x0=gx-gw, x1=gx+gw, y0=-0.12, y1=0,
                      fillcolor="#1e3a5f", line_width=0, layer="below")

    # ground line
    fig.add_shape(type="line", x0=-1.6, x1=1.6, y0=0, y1=0,
                  line=dict(color="#475569", width=2))

    # landing pad
    fig.add_shape(type="rect", x0=-0.13, x1=0.13, y0=0, y1=0.025,
                  fillcolor="#14532d", line=dict(color="#4ade80", width=2))
    # pad flags
    for fx in [-0.13, 0.13]:
        fig.add_shape(type="line", x0=fx, x1=fx, y0=0, y1=0.08,
                      line=dict(color="#86efac", width=1))
    fig.add_trace(go.Scatter(x=[-0.13, 0.13], y=[0.085, 0.085],
                             mode="text", text=["🚩", "🚩"],
                             textfont=dict(size=14), showlegend=False))

    # trail
    if len(traj) > 1:
        tx = [p[0] for p in traj]
        ty = [p[1] for p in traj]
        fig.add_trace(go.Scatter(x=tx, y=ty, mode="lines",
                                 line=dict(color="rgba(56,189,248,0.35)", width=2, dash="dot"),
                                 showlegend=False))

    # engine flames
    if last_action == 2:
        fx, fy = _poly_xy(_rotate(FLAME_MAIN, cx, cy, angle))
        fx.append(fx[0]); fy.append(fy[0])
        fig.add_trace(go.Scatter(x=fx, y=fy, mode="lines", fill="toself",
                                 fillcolor="rgba(251,146,60,0.85)",
                                 line=dict(color="#fbbf24", width=1),
                                 showlegend=False))
        # inner bright core
        core = [(-0.015, -0.015), (0.015, -0.015), (0.008, -0.10), (0, -0.14), (-0.008, -0.10)]
        cx2, cy2 = _poly_xy(_rotate(core, cx, cy, angle))
        cx2.append(cx2[0]); cy2.append(cy2[0])
        fig.add_trace(go.Scatter(x=cx2, y=cy2, mode="lines", fill="toself",
                                 fillcolor="rgba(254,240,138,0.9)",
                                 line=dict(color="#fef08a", width=0),
                                 showlegend=False))
    elif last_action == 1:
        fx, fy = _poly_xy(_rotate(FLAME_LEFT, cx, cy, angle))
        fx.append(fx[0]); fy.append(fy[0])
        fig.add_trace(go.Scatter(x=fx, y=fy, mode="lines", fill="toself",
                                 fillcolor="rgba(56,189,248,0.8)",
                                 line=dict(color="#7dd3fc", width=1),
                                 showlegend=False))
    elif last_action == 3:
        fx, fy = _poly_xy(_rotate(FLAME_RIGHT, cx, cy, angle))
        fx.append(fx[0]); fy.append(fy[0])
        fig.add_trace(go.Scatter(x=fx, y=fy, mode="lines", fill="toself",
                                 fillcolor="rgba(56,189,248,0.8)",
                                 line=dict(color="#7dd3fc", width=1),
                                 showlegend=False))

    # lander body
    bx, by = _poly_xy(body_r)
    bx.append(bx[0]); by.append(by[0])
    body_color = "#ef4444" if p_done else "#e2e8f0"
    fig.add_trace(go.Scatter(x=bx, y=by, mode="lines", fill="toself",
                             fillcolor="#1e293b" if not p_done else "#450a0a",
                             line=dict(color=body_color, width=2),
                             showlegend=False))

    # legs & feet
    for seg, w in [(l_leg_r, 2), (r_leg_r, 2), (l_foot_r, 3), (r_foot_r, 3)]:
        sx, sy = _poly_xy(seg)
        fig.add_trace(go.Scatter(x=sx, y=sy, mode="lines",
                                 line=dict(color="#94a3b8", width=w),
                                 showlegend=False))

    # leg contact glow
    if int(obs[6]):
        lc = _rotate([(-0.13, -0.09)], cx, cy, angle)[0]
        fig.add_trace(go.Scatter(x=[lc[0]], y=[lc[1]], mode="markers",
                                 marker=dict(size=10, color="#4ade80",
                                             line=dict(color="#86efac", width=2)),
                                 showlegend=False))
    if int(obs[7]):
        rc = _rotate([(0.13, -0.09)], cx, cy, angle)[0]
        fig.add_trace(go.Scatter(x=[rc[0]], y=[rc[1]], mode="markers",
                                 marker=dict(size=10, color="#4ade80",
                                             line=dict(color="#86efac", width=2)),
                                 showlegend=False))

    # window / cockpit
    wd = _rotate([(0, 0.04)], cx, cy, angle)[0]
    fig.add_trace(go.Scatter(x=[wd[0]], y=[wd[1]], mode="markers",
                             marker=dict(size=8, color="#38bdf8",
                                         line=dict(color="#7dd3fc", width=1)),
                             showlegend=False))

    # result overlay
    if p_done:
        msg = "💥 CRASHED" if cy < 0.05 else "🎉 LANDED!"
        fig.add_annotation(x=0, y=0.95, text=msg, showarrow=False,
                           font=dict(size=36, color="#fbbf24"),
                           bgcolor="rgba(0,0,0,0.6)", borderpad=12)

    fig.update_layout(
        height=620,
        paper_bgcolor="#060d1a",
        plot_bgcolor="#060d1a",
        font=dict(color="#e2e8f0"),
        showlegend=False,
        margin=dict(l=30, r=30, t=20, b=20),
        xaxis=dict(range=[-1.5, 1.5], gridcolor="#0f2035",
                   zeroline=False, showticklabels=True),
        yaxis=dict(range=[-0.14, 1.75], gridcolor="#0f2035",
                   zeroline=False, scaleanchor="x", scaleratio=1),
    )
    return fig


def _make_game_html(game_id, gravity, main_power, side_power,
                    enable_wind, wind_power, turb_power,
                    pad_width=2.0, max_land_speed=0.09, max_land_angle=13):
    W, H = 920, 700
    wind_js   = "true" if enable_wind else "false"
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#060d1a;display:flex;flex-direction:column;align-items:center;
       font-family:'Courier New',monospace;user-select:none;overflow:hidden}}
  #bar{{display:flex;gap:8px;padding:7px 0 5px;width:{W}px;align-items:center}}
  button{{background:#1e293b;color:#e2e8f0;border:2px solid #334155;border-radius:6px;
          padding:5px 14px;font-family:'Courier New',monospace;font-size:13px;
          font-weight:bold;cursor:pointer;transition:background .15s}}
  button:hover{{background:#334155}}
  #clock{{color:#22d3ee;font-size:17px;font-weight:bold;margin-left:6px;
          font-variant-numeric:tabular-nums}}
  #scorespan{{color:#94a3b8;font-size:12px;margin-left:6px}}
  #hint{{color:#334155;font-size:11px;margin-left:auto;text-align:right}}
</style>
</head>
<body>
<div id="bar">
  <button id="btnStart" onclick="togglePause()"><span id="btnLabel">&#9654; Start</span></button>
  <button onclick="giveUp()">&#128128; Give Up</button>
  <span id="clock">00:00</span>
  <span id="scorespan">step 0 &nbsp;|&nbsp; reward +0.0</span>
  <span id="hint">&#8593; main &nbsp;&#8592;&#8594; side &nbsp;&middot;&nbsp; hold key = engine ON, release = OFF &nbsp;&middot;&nbsp; SPACE pause &nbsp;&middot;&nbsp; R restart</span>
</div>
<canvas id="c" width="{W}" height="{H}" tabindex="0"></canvas>
<script>
const GAME_ID={game_id};
const W={W},H={H};
const CFG={{gravity:{gravity},mainPower:{main_power},sidePower:{side_power},
           wind:{wind_js},windPower:{wind_power},turbPower:{turb_power},
           padHalf:{pad_width}*0.065,
           maxSpeed:{max_land_speed},maxAngleRad:{max_land_angle}*Math.PI/180}};

// ── coordinate mapping ──────────────────────────────────────────────────────
const XL=-1.6,XR=1.6,YB=-0.12,YT=3.0;
function cx(x){{return (x-XL)/(XR-XL)*W;}}
function cy(y){{return H-(y-YB)/(YT-YB)*H;}}
function sc(d){{return d/(XR-XL)*W;}}

// ── physics constants (tuned to match gymnasium LunarLander-v3 feel) ────────
const FPS=50, DT=1/FPS;
// per-step velocity change in normalised coords
const GRAV      = CFG.gravity * 0.00022;   // at -10 → slow ~8s free-fall from top
const MTHRUST   = CFG.mainPower * 0.00038; // at 13  → overcomes gravity comfortably
const STORQUE   = CFG.sidePower * 0.0016;  // at 0.6 → 0.00096 rad/step
const DAMP_V    = 0.994;
const DAMP_ANG  = 0.990;

// ── stars (pre-computed) ────────────────────────────────────────────────────
const STARS=[];
for(let i=0;i<160;i++)
  STARS.push({{x:Math.random()*W,y:Math.random()*H*0.8,r:Math.random()*1.4+0.4,a:Math.random()*.7+.2}});

// ── terrain (pre-computed once per GAME_ID) ─────────────────────────────────
const RNG=(function(){{
  let s=GAME_ID*1234567+42;
  return ()=>{{s=(s*1664525+1013904223)&0xffffffff;return(s>>>0)/0xffffffff;}};
}})();
const TERRAIN=[];
const TNPTS=24;
for(let i=0;i<=TNPTS;i++){{
  const x=XL+(XR-XL)*i/TNPTS;
  let y=Math.abs(x)<0.14?0:((RNG()-.5)*.10+(Math.abs(x)>.9?RNG()*.08:0));
  TERRAIN.push({{x,y}});
}}

// ── game state ───────────────────────────────────────────────────────────────
let lander,trail,activeAction,burstTimer,paused,gameOver,crashReasons,startT,elapsed,totalReward,steps,windIdx;
const BURST_STEPS=50; // 1 second at 50 fps

function initGame(){{
  const rx=(RNG()-.5)*.18, ry=(RNG()-.5)*.08;
  lander={{x:rx,y:2.5+ry,vx:(RNG()-.5)*.02,vy:-.003,angle:(RNG()-.5)*.05,omega:0,legL:false,legR:false}};
  trail=[]; activeAction=0; burstTimer=0; paused=true; gameOver=null; crashReasons=[];
  startT=null; elapsed=0; totalReward=0; steps=0; windIdx=RNG()*1000;
  document.getElementById('btnLabel').textContent='▶ Start';
}}
initGame();

// ── input ────────────────────────────────────────────────────────────────────
const KMAP={{ArrowUp:2,ArrowLeft:1,ArrowRight:3,ArrowDown:0}};
document.getElementById('c').focus();
document.addEventListener('keydown',e=>{{
  if(e.key===' '){{e.preventDefault();togglePause();return;}}
  if(e.key==='r'||e.key==='R'){{initGame();return;}}
  const a=KMAP[e.key];
  if(a===undefined)return;
  e.preventDefault();
  activeAction=a;
}});
document.addEventListener('keyup',e=>{{
  const a=KMAP[e.key];
  if(a!==undefined&&activeAction===a)activeAction=0;
}});

function togglePause(){{
  if(gameOver)return;
  paused=!paused;
  if(!paused&&!startT)startT=performance.now();
  document.getElementById('btnLabel').textContent=paused?'▶ Start':'❚❚ Pause';
}}
function giveUp(){{gameOver='crashed';crashReasons=['Mission aborted (Give Up)'];paused=false;totalReward-=50;}}

// ── physics step ─────────────────────────────────────────────────────────────
function physStep(){{
  const s=lander;
  let{{x,y,vx,vy,angle,omega}}=s;

  // wind
  if(CFG.wind&&!s.legL&&!s.legR){{
    windIdx+=1;
    vx+=Math.tanh(Math.sin(.02*windIdx)+Math.sin(Math.PI*.01*windIdx))*CFG.windPower*.00004;
    omega+=Math.tanh(Math.sin(.02*windIdx*1.3)+Math.sin(Math.PI*.01*windIdx*1.3))*CFG.turbPower*.00006;
  }}

  // gravity
  vy+=GRAV;

  // engines
  let mFire=0,sFire=0;
  if(activeAction===2){{mFire=1;vx+=Math.sin(angle)*MTHRUST;vy+=Math.cos(angle)*MTHRUST;}}
  if(activeAction===1){{sFire=1;omega+=STORQUE;vx-=Math.cos(angle)*STORQUE*.25;}}
  if(activeAction===3){{sFire=1;omega-=STORQUE;vx+=Math.cos(angle)*STORQUE*.25;}}

  // damping
  vx*=DAMP_V; vy*=DAMP_V; omega*=DAMP_ANG;

  x+=vx; y+=vy; angle+=omega;

  // leg contact — exact foot world-y from canvas geometry
  // foot local canvas offsets (from drawLander): (±0.175·S, +0.10·S)
  // world_game_y = lander.y - (lx·sin(a) + ly·cos(a)) / (H/(YT-YB))
  // F = S·(YT-YB)/H = W·(YT-YB)/(H·(XR-XL))
  const _F=W*(YT-YB)/(H*(XR-XL));
  const lly=y-(-0.175*_F*Math.sin(angle)+0.10*_F*Math.cos(angle));
  const rly=y-( 0.175*_F*Math.sin(angle)+0.10*_F*Math.cos(angle));
  const legL=lly<=0.02, legR=rly<=0.02;

  // fuel cost
  totalReward-=(mFire*.28+sFire*.025);
  steps++;

  // termination
  let over=null;
  if(Math.abs(x)>=1.45){{
    over='crashed'; totalReward-=100;
    crashReasons=['Flew out of bounds (|x| ≥ 1.45)'];
  }} else if(y<=0.03){{
    const spd=Math.sqrt(vx*vx+vy*vy);
    const tiltOk  = Math.abs(angle)<=CFG.maxAngleRad;
    const speedOk = spd<=CFG.maxSpeed;
    const legsOk  = legL&&legR;
    const padOk   = Math.abs(x)<=CFG.padHalf;
    if(tiltOk&&speedOk&&legsOk&&padOk){{
      over='landed'; totalReward+=100;
    }} else {{
      over='crashed'; totalReward-=100;
      crashReasons=[];
      if(!legsOk){{
        const legMsg=(!legL&&!legR)?'Neither leg touched the ground'
                    :legL?'Only left leg touched  (right leg still up)'
                         :'Only right leg touched  (left leg still up)';
        crashReasons.push(legMsg);
      }}
      if(!speedOk) crashReasons.push('Speed too high  ('+spd.toFixed(3)+' > '+CFG.maxSpeed.toFixed(2)+' limit)');
      if(!tiltOk)  crashReasons.push('Too tilted  ('+Math.abs(angle*180/Math.PI).toFixed(1)+'° > '+(CFG.maxAngleRad*180/Math.PI).toFixed(1)+'° limit)');
      if(!padOk)   crashReasons.push('Missed the pad  (|x|='+Math.abs(x).toFixed(3)+' > pad half='+CFG.padHalf.toFixed(3)+')');
    }}
    y=0.03;
  }}

  lander={{x,y:Math.max(y,0.02),vx,vy,angle,omega,legL,legR}};
  trail.push({{x,y:lander.y}});
  if(trail.length>400)trail.shift();
  if(over)gameOver=over;
}}

// ── rendering ────────────────────────────────────────────────────────────────
const canvas=document.getElementById('c');
const ctx=canvas.getContext('2d');

function render(ts){{
  ctx.clearRect(0,0,W,H);

  // background
  const bg=ctx.createLinearGradient(0,0,0,H);
  bg.addColorStop(0,'#030810'); bg.addColorStop(1,'#0a1628');
  ctx.fillStyle=bg; ctx.fillRect(0,0,W,H);

  // stars
  STARS.forEach(s=>{{
    const tw=.7+.3*Math.sin(ts*.0009+s.x*.1);
    ctx.globalAlpha=s.a*tw;
    ctx.fillStyle='#e2e8f0';
    ctx.beginPath(); ctx.arc(s.x,s.y,s.r,0,6.283); ctx.fill();
  }});
  ctx.globalAlpha=1;

  // ground fill
  const gy=cy(0);
  ctx.fillStyle='#0b1a2e'; ctx.fillRect(0,gy,W,H-gy);

  // terrain
  ctx.fillStyle='#162840';
  ctx.beginPath(); ctx.moveTo(0,H);
  TERRAIN.forEach(p=>ctx.lineTo(cx(p.x),cy(p.y)));
  ctx.lineTo(W,H); ctx.closePath(); ctx.fill();
  ctx.strokeStyle='#1e3a5f'; ctx.lineWidth=1.5;
  ctx.beginPath();
  TERRAIN.forEach((p,i)=>i?ctx.lineTo(cx(p.x),cy(p.y)):ctx.moveTo(cx(p.x),cy(p.y)));
  ctx.stroke();

  // landing pad
  const px1=cx(-CFG.padHalf),px2=cx(CFG.padHalf),py=cy(.025),ph=9;
  ctx.fillStyle='rgba(74,222,128,.08)'; ctx.fillRect(px1-12,py-18,px2-px1+24,32);
  ctx.fillStyle='#14532d'; ctx.strokeStyle='#4ade80'; ctx.lineWidth=2;
  ctx.beginPath(); ctx.rect(px1,py,px2-px1,ph); ctx.fill(); ctx.stroke();
  // lights
  [px1+5,(px1+px2)/2,px2-5].forEach(lx=>{{
    ctx.fillStyle='#4ade80'; ctx.beginPath(); ctx.arc(lx,py+ph/2,3,0,6.283); ctx.fill();
  }});
  // flags
  drawFlag(px1,py); drawFlag(px2,py);

  // trail
  if(trail.length>1){{
    ctx.beginPath();
    trail.forEach((p,i)=>i?ctx.lineTo(cx(p.x),cy(p.y)):ctx.moveTo(cx(p.x),cy(p.y)));
    ctx.strokeStyle='rgba(56,189,248,.22)'; ctx.lineWidth=1.5;
    ctx.setLineDash([4,5]); ctx.stroke(); ctx.setLineDash([]);
  }}

  // lander
  const S=sc(1);
  const lx=cx(lander.x), ly=cy(lander.y);
  ctx.save(); ctx.translate(lx,ly); ctx.rotate(lander.angle);
  drawLander(S,lander.legL,lander.legR,gameOver);
  if(activeAction!==0&&!gameOver)drawFlame(S,activeAction,ts);
  ctx.restore();

  // HUD top bar (timer, engine indicators)
  drawHUD(ts);

  // overlays
  if(gameOver)drawGameOver();
  else if(paused&&!startT)drawStartPrompt();
  else if(paused)drawPaused();

  // telemetry panel always on top
  drawTelemetry();
}}

function drawFlag(fx,fy){{
  ctx.strokeStyle='#86efac'; ctx.lineWidth=1.5;
  ctx.beginPath(); ctx.moveTo(fx,fy); ctx.lineTo(fx,fy-18); ctx.stroke();
  ctx.fillStyle='#4ade80';
  ctx.beginPath(); ctx.moveTo(fx,fy-18); ctx.lineTo(fx+10,fy-13); ctx.lineTo(fx,fy-8); ctx.closePath(); ctx.fill();
}}

function drawLander(S,legL,legR,over){{
  const lc=legL?'#4ade80':'#94a3b8', rc=legR?'#4ade80':'#94a3b8';
  // legs
  [[lc,-1],[rc,1]].forEach(([col,side])=>{{
    ctx.strokeStyle=col; ctx.lineWidth=sc(.022);
    ctx.beginPath(); ctx.moveTo(side*sc(.058),sc(.012)); ctx.lineTo(side*sc(.13),sc(.10)); ctx.stroke();
    ctx.lineWidth=sc(.028);
    ctx.beginPath(); ctx.moveTo(side*sc(.13),sc(.10)); ctx.lineTo(side*sc(.175),sc(.10)); ctx.stroke();
  }});
  // body
  ctx.fillStyle=over==='crashed'?'#3b0a0a':'#1e293b';
  ctx.strokeStyle=over==='crashed'?'#ef4444':'#cbd5e1';
  ctx.lineWidth=sc(.012);
  ctx.beginPath(); ctx.rect(-sc(.058),-sc(.082),sc(.116),sc(.094)); ctx.fill(); ctx.stroke();
  // nozzle
  ctx.fillStyle='#475569';
  ctx.beginPath(); ctx.moveTo(-sc(.026),sc(.012)); ctx.lineTo(sc(.026),sc(.012));
  ctx.lineTo(sc(.018),sc(.038)); ctx.lineTo(-sc(.018),sc(.038)); ctx.closePath(); ctx.fill();
  // cockpit
  ctx.fillStyle=over?'#334155':'#38bdf8'; ctx.strokeStyle='#7dd3fc'; ctx.lineWidth=sc(.009);
  ctx.beginPath(); ctx.arc(0,-sc(.038),sc(.026),0,6.283); ctx.fill(); ctx.stroke();
}}

function drawFlame(S,action,ts){{
  const flicker=.85+.15*Math.sin(ts*.07);
  if(action===2){{
    const fl=sc(.16)*flicker, fw=sc(.038);
    ctx.fillStyle='rgba(251,146,60,.88)'; ctx.shadowColor='#fb923c'; ctx.shadowBlur=18;
    ctx.beginPath();
    ctx.moveTo(-fw,sc(.038));
    ctx.quadraticCurveTo(-fw*.4,sc(.038)+fl*.55,0,sc(.038)+fl);
    ctx.quadraticCurveTo(fw*.4,sc(.038)+fl*.55,fw,sc(.038)); ctx.closePath(); ctx.fill();
    const cl=fl*.55;
    ctx.fillStyle='rgba(254,240,138,.96)'; ctx.shadowBlur=8;
    ctx.beginPath();
    ctx.moveTo(-fw*.38,sc(.038));
    ctx.quadraticCurveTo(-fw*.15,sc(.038)+cl*.5,0,sc(.038)+cl);
    ctx.quadraticCurveTo(fw*.15,sc(.038)+cl*.5,fw*.38,sc(.038)); ctx.closePath(); ctx.fill();
    ctx.shadowBlur=0;
  }} else {{
    const side=action===1?1:-1;
    const fl=sc(.13)*flicker, fw=sc(.022);
    const bx=side*-sc(.058), by=-sc(.048);
    ctx.fillStyle='rgba(56,189,248,.88)'; ctx.shadowColor='#38bdf8'; ctx.shadowBlur=14;
    ctx.beginPath();
    ctx.moveTo(bx,by-fw);
    ctx.quadraticCurveTo(bx-side*fl*.55,by,bx-side*fl,by);
    ctx.quadraticCurveTo(bx-side*fl*.55,by,bx,by+fw); ctx.closePath(); ctx.fill();
    ctx.shadowBlur=0;
  }}
}}

function telColor(val, warnThresh, dangerThresh){{
  const a=Math.abs(val);
  if(a>=dangerThresh) return '#ef4444';
  if(a>=warnThresh)   return '#fbbf24';
  return '#4ade80';
}}

function drawTelemetry(){{
  const s=lander;
  const tiltDeg=(s.angle*180/Math.PI);
  const tiltLimit=CFG.maxAngleRad*180/Math.PI;
  const speed=Math.sqrt(s.vx*s.vx+s.vy*s.vy);

  const rows=[
    {{lbl:'H',     val:s.y.toFixed(3),         raw:s.y,      warn:0.4,             danger:0.15}},
    {{lbl:'Tilt',  val:tiltDeg.toFixed(1)+'°',  raw:tiltDeg,  warn:tiltLimit*0.7,   danger:tiltLimit}},
    {{lbl:'Vx',    val:s.vx.toFixed(3),         raw:s.vx,     warn:CFG.maxSpeed*.6, danger:CFG.maxSpeed}},
    {{lbl:'Vy',    val:s.vy.toFixed(3),         raw:s.vy,     warn:CFG.maxSpeed*.6, danger:CFG.maxSpeed}},
    {{lbl:'Speed', val:speed.toFixed(3),         raw:speed,    warn:CFG.maxSpeed*.6, danger:CFG.maxSpeed}},
  ];

  const ROW_H=30, FONT_SZ=16, PAD=14;
  const pw=230, ph=PAD+rows.length*ROW_H+28+PAD;
  const px=14, py=H-ph-10;

  // panel — brighter border after game over so it stands out
  ctx.fillStyle=gameOver?'rgba(4,10,22,0.96)':'rgba(4,10,22,0.88)';
  roundRect(px,py,pw,ph,10); ctx.fill();
  ctx.strokeStyle=gameOver?'#38bdf8':'#1e3a5f'; ctx.lineWidth=gameOver?2:1.5;
  roundRect(px,py,pw,ph,10); ctx.stroke();

  // header
  ctx.font='bold 11px Courier New';
  ctx.fillStyle=gameOver?'#38bdf8':'#334155';
  ctx.textAlign='left'; ctx.fillText('TELEMETRY', px+PAD, py+13);

  // divider
  ctx.strokeStyle=gameOver?'#1e3a5f':'rgba(30,58,95,0.6)'; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(px+8,py+18); ctx.lineTo(px+pw-8,py+18); ctx.stroke();

  ctx.font=`bold ${{FONT_SZ}}px Courier New`;
  rows.forEach((r,i)=>{{
    const ry=py+PAD+18+i*ROW_H;
    ctx.fillStyle='#94a3b8'; ctx.textAlign='left';
    ctx.fillText(r.lbl, px+PAD, ry);
    const col=telColor(r.raw, r.warn, r.danger);
    ctx.fillStyle=col; ctx.textAlign='right';
    ctx.fillText(r.val, px+pw-PAD, ry);
    ctx.strokeStyle='rgba(30,58,95,0.5)'; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(px+8,ry+5); ctx.lineTo(px+pw-8,ry+5); ctx.stroke();
  }});

  // speed bar — limit marker
  const barY=py+ph-18, barX=px+PAD, barW=pw-PAD*2;
  ctx.fillStyle='#0f172a'; roundRect(barX,barY,barW,10,4); ctx.fill();
  const frac=Math.min(speed/CFG.maxSpeed/1.5,1);
  const barCol=speed>=CFG.maxSpeed?'#ef4444':speed>=CFG.maxSpeed*.6?'#fbbf24':'#4ade80';
  if(frac>0){{ctx.fillStyle=barCol; roundRect(barX,barY,barW*frac,10,4); ctx.fill();}}
  // limit tick
  const limitX=barX+barW/1.5;
  ctx.strokeStyle='#e2e8f0'; ctx.lineWidth=2;
  ctx.beginPath(); ctx.moveTo(limitX,barY-2); ctx.lineTo(limitX,barY+12); ctx.stroke();
  ctx.strokeStyle='#1e3a5f'; ctx.lineWidth=1;
  roundRect(barX,barY,barW,10,4); ctx.stroke();

  ctx.textAlign='left';
}}

function drawHUD(ts){{
  // timer & score (top-left)
  const mm=String(Math.floor(elapsed/60)).padStart(2,'0');
  const ss=String(Math.floor(elapsed%60)).padStart(2,'0');
  ctx.font='bold 21px Courier New'; ctx.fillStyle='#22d3ee'; ctx.textAlign='left';
  ctx.fillText('⏱ '+mm+':'+ss, 14, 30);
  ctx.font='13px Courier New'; ctx.fillStyle='#64748b';
  ctx.fillText('step '+steps, 14, 48);
  const rSign=totalReward>=0?'+':'';
  ctx.fillText('reward '+rSign+totalReward.toFixed(1), 14, 64);

  // ── engine indicators (top-right) ─────────────────────────────────────────
  const engs=[{{a:1,lbl:'LEFT',col:'#38bdf8',x:W-204}},
              {{a:2,lbl:'MAIN',col:'#fb923c',x:W-134}},
              {{a:3,lbl:'RIGHT',col:'#38bdf8',x:W-62}}];
  engs.forEach(e=>{{
    const on=activeAction===e.a&&!gameOver;
    ctx.fillStyle=on?e.col:'#1e293b';
    ctx.strokeStyle=on?e.col:'#334155'; ctx.lineWidth=1.5;
    roundRect(e.x-32,8,62,22,5); ctx.fill(); ctx.stroke();
    ctx.font='bold 11px Courier New'; ctx.textAlign='center';
    ctx.fillStyle=on?'#0f172a':'#475569';
    ctx.fillText(e.lbl, e.x, 23);
  }});
  ctx.textAlign='left';
}}

function roundRect(x,y,w,h,r){{
  ctx.beginPath(); ctx.moveTo(x+r,y);
  ctx.lineTo(x+w-r,y); ctx.arcTo(x+w,y,x+w,y+r,r);
  ctx.lineTo(x+w,y+h-r); ctx.arcTo(x+w,y+h,x+w-r,y+h,r);
  ctx.lineTo(x+r,y+h); ctx.arcTo(x,y+h,x,y+h-r,r);
  ctx.lineTo(x,y+r); ctx.arcTo(x,y,x+r,y,r); ctx.closePath();
}}

function drawGameOver(){{
  ctx.fillStyle='rgba(0,0,0,.60)'; ctx.fillRect(0,0,W,H);
  const landed=gameOver==='landed';
  const col=landed?'#4ade80':'#ef4444';
  const msg=landed?'LANDED!':'CRASHED';

  // title
  ctx.font='bold 52px Courier New'; ctx.fillStyle=col;
  ctx.shadowColor=col; ctx.shadowBlur=28; ctx.textAlign='center';
  ctx.fillText(msg,W/2,H/2-80); ctx.shadowBlur=0;

  // reward
  ctx.font='bold 20px Courier New'; ctx.fillStyle='#e2e8f0';
  const rs=totalReward>=0?'+':'';
  ctx.fillText('Reward: '+rs+totalReward.toFixed(1),W/2,H/2-44);

  if(!landed&&crashReasons.length>0){{
    // reason panel
    const lineH=28, pad=18;
    const panelW=Math.min(W-60, 560);
    const panelH=pad*2+crashReasons.length*lineH+4;
    const panelX=W/2-panelW/2, panelY=H/2-20;

    ctx.fillStyle='rgba(69,10,10,0.85)';
    roundRect(panelX,panelY,panelW,panelH,10); ctx.fill();
    ctx.strokeStyle='#ef4444'; ctx.lineWidth=1.5;
    roundRect(panelX,panelY,panelW,panelH,10); ctx.stroke();

    ctx.font='bold 12px Courier New'; ctx.fillStyle='#fca5a5';
    ctx.textAlign='left';
    ctx.fillText('WHY IT CRASHED:', panelX+pad, panelY+pad);

    ctx.font='15px Courier New'; ctx.fillStyle='#fecaca';
    crashReasons.forEach((r,i)=>{{
      ctx.fillText('✕  '+r, panelX+pad, panelY+pad+16+lineH*(i+1));
    }});
  }}

  ctx.font='13px Courier New'; ctx.fillStyle='#64748b'; ctx.textAlign='center';
  ctx.fillText('Press R to restart',W/2,H-30);
  ctx.textAlign='left';
}}
function drawStartPrompt(){{
  ctx.fillStyle='rgba(0,0,0,.38)'; ctx.fillRect(0,0,W,H);
  ctx.font='bold 28px Courier New'; ctx.fillStyle='#22d3ee'; ctx.textAlign='center';
  ctx.fillText('Press SPACE or click Start',W/2,H/2-8);
  ctx.font='13px Courier New'; ctx.fillStyle='#64748b';
  ctx.fillText('Arrow keys: ↑ main engine  ←→ side  — hold = ON, release = OFF',W/2,H/2+24);
  ctx.textAlign='left';
}}
function drawPaused(){{
  ctx.font='bold 24px Courier New'; ctx.fillStyle='rgba(34,211,238,.75)'; ctx.textAlign='center';
  ctx.fillText('⏸ PAUSED — SPACE to resume',W/2,H/2);
  ctx.textAlign='left';
}}

// ── update UI elements outside canvas ────────────────────────────────────────
function updateDOM(){{
  const mm=String(Math.floor(elapsed/60)).padStart(2,'0');
  const ss=String(Math.floor(elapsed%60)).padStart(2,'0');
  document.getElementById('clock').textContent=mm+':'+ss;
  const rs=totalReward>=0?'+':'';
  document.getElementById('scorespan').textContent='step '+steps+'  |  reward '+rs+totalReward.toFixed(1);
}}

// ── game loop ─────────────────────────────────────────────────────────────────
let lastTs=0, accum=0;
const STEP_MS=1000/FPS;

function loop(ts){{
  requestAnimationFrame(loop);
  if(!paused&&!gameOver){{
    const delta=Math.min(ts-lastTs,120);
    accum+=delta;
    while(accum>=STEP_MS&&!gameOver){{physStep();accum-=STEP_MS;}}
    if(startT!==null) elapsed=(ts-startT)/1000;
    updateDOM();
  }}
  lastTs=ts;
  render(ts);
}}
requestAnimationFrame(loop);
</script>
</body>
</html>"""


tab_train, tab_play, tab_info = st.tabs(["📈 Training", "🕹️ Manual Play", "ℹ️ Info"])


# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — Training
# ──────────────────────────────────────────────────────────────────────────────
with tab_train:
    st.title("🚀 LunarLander-v3 — DQN Training")

    rewards  = st.session_state.episode_rewards
    ep_count = len(rewards)

    # snapshot approach — take a new snapshot every 20 episodes; always render from snapshot
    _new_since_display = st.session_state.last_episode - st.session_state.last_display_episode
    _time_to_snapshot = (
        not st.session_state.training
        or st.session_state.done
        or _new_since_display >= 20
    )
    if _time_to_snapshot and ep_count > 0:
        st.session_state.last_display_episode = st.session_state.last_episode
        st.session_state._snap_rewards   = list(st.session_state.episode_rewards)
        st.session_state._snap_lengths   = list(st.session_state.episode_lengths)
        st.session_state._snap_losses    = list(st.session_state.losses)
        st.session_state._snap_epsilons  = list(st.session_state.epsilons)
        st.session_state._snap_fires     = list(st.session_state.engine_fires)

    _snap_rewards  = st.session_state.get("_snap_rewards",  [])
    _snap_lengths  = st.session_state.get("_snap_lengths",  [])
    _snap_losses   = st.session_state.get("_snap_losses",   [])
    _snap_epsilons = st.session_state.get("_snap_epsilons", [])
    _snap_fires    = st.session_state.get("_snap_fires",    [])
    _snap_count    = len(_snap_rewards)

    if st.session_state.training:
        _next    = st.session_state.last_display_episode + 20
        _pending = _new_since_display
        st.info(
            f"Training… episode {st.session_state.last_episode} / {total_episodes}"
            + (f"  —  charts refresh at episode {_next} (+{20 - _pending} to go)"
               if not _time_to_snapshot else "")
        )
    elif st.session_state.done:
        _snap_count_used = len(_snap_rewards)
        st.success(f"Training complete — {ep_count} episodes")
    else:
        st.info("Configure parameters in the sidebar, then press ▶ Train.")

    # metrics
    c1, c2, c3, c4 = st.columns(4)
    def _metric(col, label, value):
        col.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div>'
                     f'<div class="metric-value">{value}</div></div>', unsafe_allow_html=True)
    _metric(c1, "Episodes done",  ep_count if ep_count else "—")
    _metric(c2, "Last reward",    f"{rewards[-1]:.1f}" if rewards else "—")
    _metric(c3, "Best reward",    f"{st.session_state.best_reward:.1f}" if rewards else "—")
    _metric(c4, "ε (epsilon)",    f"{st.session_state.epsilons[-1]:.3f}" if st.session_state.epsilons else "—")
    st.markdown("<br>", unsafe_allow_html=True)

    # training charts — only rendered every 20 episodes while training
    def _smooth(data, w=20):
        return np.convolve(data, np.ones(w)/w, mode="valid") if len(data) >= w else np.array(data)

    # rebuild the figure only when the snapshot actually changed; reuse it otherwise
    _fig_stale = (st.session_state.get("_chart_fig") is None
                  or st.session_state.get("_chart_fig_count") != _snap_count)
    if _snap_count >= 2 and _fig_stale:
        fig = make_subplots(rows=2, cols=2,
            subplot_titles=("Episode Reward","Smoothed Reward (20-ep)","Loss","Epsilon"),
            vertical_spacing=0.14)
        xs = list(range(1, _snap_count + 1))
        fig.add_trace(go.Scatter(x=xs, y=_snap_rewards, mode="lines",
                                 line=dict(color="#38bdf8", width=1)), row=1, col=1)
        sm = _smooth(_snap_rewards)
        fig.add_trace(go.Scatter(x=list(range(20, _snap_count+1)), y=sm, mode="lines",
                                 line=dict(color="#f472b6", width=2)), row=1, col=2)
        fig.add_hline(y=200, line_dash="dash", line_color="#4ade80",
                      annotation_text="solved (200)", row=1, col=2)
        fig.add_trace(go.Scatter(x=xs, y=_snap_losses, mode="lines",
                                 line=dict(color="#fb923c", width=1)), row=2, col=1)
        fig.add_trace(go.Scatter(x=xs, y=_snap_epsilons, mode="lines",
                                 line=dict(color="#a78bfa", width=2)), row=2, col=2)
        fig.update_layout(height=520, paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
                          font=dict(color="#e2e8f0"), showlegend=False,
                          margin=dict(l=40, r=20, t=50, b=30))
        for r in (1,2):
            for c in (1,2):
                fig.update_xaxes(gridcolor="#334155", row=r, col=c)
                fig.update_yaxes(gridcolor="#334155", row=r, col=c)
        st.session_state._chart_fig = fig
        st.session_state._chart_fig_count = _snap_count

    if _snap_count >= 2:
        st.plotly_chart(st.session_state._chart_fig, use_container_width=True)
    elif _snap_count == 0:
        st.markdown("<div style='height:160px;display:flex;align-items:center;"
                    "justify-content:center;color:#475569;font-size:1.1rem;'>"
                    "Charts appear after the first episode.</div>", unsafe_allow_html=True)

    # episode table — always rendered from snapshot
    if _snap_count >= 1:
        st.subheader("Recent Episodes")
        n = min(20, _snap_count)
        _fires_n = _snap_fires[-n:] if _snap_fires else [[0,0,0,0]] * n
        st.dataframe({
            "Episode": list(range(_snap_count - n + 1, _snap_count + 1)),
            "Reward":  [f"{r:.1f}"  for r in _snap_rewards[-n:]],
            "Steps":   _snap_lengths[-n:],
            "Loss":    [f"{l:.4f}"  for l in _snap_losses[-n:]],
            "ε":       [f"{e:.3f}"  for e in _snap_epsilons[-n:]],
            "🔵 Idle":  [f[0] for f in _fires_n],
            "⬅ Left":  [f[1] for f in _fires_n],
            "🔥 Main":  [f[2] for f in _fires_n],
            "➡ Right": [f[3] for f in _fires_n],
        }, use_container_width=True, hide_index=True)

    # replay — also gated to every 20 episodes while training
    recorded = st.session_state.recorded_episodes
    if recorded and _time_to_snapshot:
        st.markdown('<div class="sec-replay"><span class="sec-label" style="color:#f472b6">'
                    '🏆 TOP 10 EPISODES — REPLAY</span></div>', unsafe_allow_html=True)
        options = [f"#{i+1}  Ep {r['ep']}  (reward {r['reward']:+.0f})"
                   for i, r in enumerate(recorded)]
        col_sel, col_info = st.columns([3, 1])
        sel = col_sel.selectbox("Select recorded episode", options,
                                index=0, label_visibility="collapsed")
        rec  = recorded[options.index(sel)]
        traj = rec["traj"]

        col_info.markdown(f'<div class="metric-card" style="padding:8px">'
                          f'<div class="metric-label">Total reward</div>'
                          f'<div class="metric-value" style="font-size:1.1rem">'
                          f'{rec["reward"]:+.1f}</div></div>', unsafe_allow_html=True)

        xs_r  = [t[0] for t in traj]
        ys_r  = [t[1] for t in traj]
        acts  = [t[5] for t in traj]
        srews = [t[6] for t in traj]
        n_s   = len(traj)

        ACT_LABELS = ["nothing","left engine","main engine","right engine"]
        ACT_COLORS = ["#475569","#38bdf8","#4ade80","#f472b6"]

        rfig = go.Figure()
        rfig.add_trace(go.Scatter(x=xs_r, y=ys_r, mode="lines",
                                  line=dict(color="#334155", width=1), showlegend=False))
        rfig.add_shape(type="rect", x0=-0.1, x1=0.1, y0=-0.02, y1=0.02,
                       fillcolor="#4ade80", line_width=0, opacity=0.6)
        frames = [go.Frame(
            data=[
                go.Scatter(x=xs_r, y=ys_r, mode="lines", line=dict(color="#334155",width=1)),
                go.Scatter(x=xs_r[:i+1], y=ys_r[:i+1], mode="lines",
                           line=dict(color="#38bdf8", width=2)),
                go.Scatter(x=[xs_r[i]], y=[ys_r[i]], mode="markers+text",
                           marker=dict(size=14, color=ACT_COLORS[acts[i]],
                                       symbol="triangle-up",
                                       line=dict(color="#e2e8f0", width=1)),
                           text=[ACT_LABELS[acts[i]]], textposition="top center",
                           textfont=dict(color="#e2e8f0", size=11)),
            ],
            name=str(i),
            layout=go.Layout(title_text=f"Step {i+1}/{n_s} | {ACT_LABELS[acts[i]]} | reward {srews[i]:+.2f}"),
        ) for i in range(n_s)]
        rfig.frames = frames
        rfig.add_trace(go.Scatter(x=xs_r[:1], y=ys_r[:1], mode="lines",
                                  line=dict(color="#38bdf8", width=2), showlegend=False))
        rfig.add_trace(go.Scatter(x=[xs_r[0]], y=[ys_r[0]], mode="markers+text",
                                  marker=dict(size=14, color=ACT_COLORS[acts[0]],
                                              symbol="triangle-up",
                                              line=dict(color="#e2e8f0", width=1)),
                                  text=[ACT_LABELS[acts[0]]], textposition="top center",
                                  textfont=dict(color="#e2e8f0", size=11), showlegend=False))
        rfig.update_layout(
            height=440, paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
            font=dict(color="#e2e8f0"), showlegend=False,
            margin=dict(l=40, r=20, t=50, b=70),
            xaxis=dict(range=[-1.5,1.5], gridcolor="#334155", title="x"),
            yaxis=dict(range=[-0.3,1.8], gridcolor="#334155", title="y"),
            title=dict(text=f"Episode {rec['ep']}", font=dict(color="#e2e8f0")),
            updatemenus=[dict(type="buttons", showactive=False, y=-0.15, x=0.5,
                              xanchor="center",
                              buttons=[
                                  dict(label="▶ Play", method="animate",
                                       args=[None, dict(frame=dict(duration=60, redraw=True),
                                                        fromcurrent=True, mode="immediate")]),
                                  dict(label="⏸ Pause", method="animate",
                                       args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                          mode="immediate")]),
                              ])],
            sliders=[dict(
                steps=[dict(args=[[str(i)], dict(frame=dict(duration=0, redraw=True),
                                                  mode="immediate")],
                            label=str(i), method="animate") for i in range(n_s)],
                active=0, y=-0.08, x=0.05, len=0.9,
                currentvalue=dict(prefix="Step: ", font=dict(color="#e2e8f0")),
                font=dict(color="#94a3b8"),
            )],
        )
        st.plotly_chart(rfig, use_container_width=True)
    elif ep_count > 0 or st.session_state.training:
        st.caption("The 10 highest-reward episodes are kept for replay — the list appears after the first episode.")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — Manual Play (self-contained JS canvas, no Streamlit reruns during play)
# ──────────────────────────────────────────────────────────────────────────────
with tab_play:
    c_new, c_info = st.columns([1, 6])
    if c_new.button("🆕 New Game", use_container_width=True):
        st.session_state.play_game_id += 1
        st.rerun()
    c_info.markdown(
        "<span style='color:#64748b;font-size:0.85rem'>"
        "Physics parameters (gravity, engine power, wind) are taken from the sidebar — "
        "click **New Game** after changing them.</span>",
        unsafe_allow_html=True,
    )
    game_html = _make_game_html(
        game_id=st.session_state.play_game_id,
        gravity=gravity,
        main_power=main_engine_power,
        side_power=side_engine_power,
        enable_wind=enable_wind,
        wind_power=wind_power,
        turb_power=turbulence_power,
        pad_width=pad_width,
        max_land_speed=max_land_speed,
        max_land_angle=max_land_angle,
    )
    components.html(game_html, height=780, scrolling=False)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — Info
# ──────────────────────────────────────────────────────────────────────────────
with tab_info:
    st.markdown("""
<style>
.info-card {
    background: #0f172a;
    border-radius: 10px;
    padding: 20px 26px;
    margin-bottom: 18px;
}
.info-card h3 { color: #e2e8f0; margin-top: 0; }
.cond-row {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    margin: 10px 0;
    padding: 10px 14px;
    border-radius: 7px;
}
.cond-icon { font-size: 1.5rem; line-height: 1.3; flex-shrink: 0; }
.cond-text { color: #cbd5e1; font-size: 0.92rem; line-height: 1.55; }
.cond-text b { color: #e2e8f0; }
.land-row  { background: #0d2318; border-left: 3px solid #4ade80; }
.crash-row { background: #2a0d0d; border-left: 3px solid #ef4444; }
.note-row  { background: #0f1e35; border-left: 3px solid #38bdf8; }
.tag-land  { display:inline-block; background:#14532d; color:#4ade80;
             border-radius:4px; padding:1px 8px; font-size:.8rem; font-weight:700; }
.tag-crash { display:inline-block; background:#450a0a; color:#ef4444;
             border-radius:4px; padding:1px 8px; font-size:.8rem; font-weight:700; }
</style>

<div class="info-card">
<h3>🛬 What counts as a successful landing?</h3>

All <b>four</b> conditions must be true at the moment the lander touches the ground:

<div class="cond-row land-row">
  <span class="cond-icon">🦵</span>
  <span class="cond-text"><b>Both legs are on the ground</b><br>
  The left and right landing legs must both make contact simultaneously.
  Touching down on one leg only counts as a crash.</span>
</div>

<div class="cond-row land-row">
  <span class="cond-icon">💨</span>
  <span class="cond-text"><b>Total speed ≤ 0.09 (normalised)</b><br>
  The combined horizontal + vertical speed must be low — a hard or fast approach crashes the lander.
  Slow down with the main engine before touchdown.</span>
</div>

<div class="cond-row land-row">
  <span class="cond-icon">📐</span>
  <span class="cond-text"><b>Tilt angle ≤ ~13°</b><br>
  The lander must be nearly upright (|angle| &lt; 0.22 rad ≈ 12.6°).
  A sideways landing is a crash even if speed is low.</span>
</div>

<div class="cond-row land-row">
  <span class="cond-icon">📍</span>
  <span class="cond-text"><b>Within the horizontal boundary</b><br>
  The lander must not have drifted past the left or right edge of the world (|x| &lt; 1.45).
  Flying out of bounds instantly crashes the episode.</span>
</div>
</div>

<div class="info-card">
<h3>💥 What causes a crash?</h3>

Any of the following ends the episode as a crash:

<div class="cond-row crash-row">
  <span class="cond-icon">🏃</span>
  <span class="cond-text"><b>Touching down too fast</b> — speed &gt; 0.09 at ground contact.</span>
</div>
<div class="cond-row crash-row">
  <span class="cond-icon">↗️</span>
  <span class="cond-text"><b>Landing at an angle</b> — tilted more than ~13° when the lander hits the ground.</span>
</div>
<div class="cond-row crash-row">
  <span class="cond-icon">🦿</span>
  <span class="cond-text"><b>One-legged touchdown</b> — only one leg contacts the ground on impact.</span>
</div>
<div class="cond-row crash-row">
  <span class="cond-icon">🚧</span>
  <span class="cond-text"><b>Out of bounds</b> — drifting past |x| = 1.45 (the horizontal edges of the world).</span>
</div>
<div class="cond-row crash-row">
  <span class="cond-icon">💀</span>
  <span class="cond-text"><b>Give Up button</b> — manually ending the episode counts as a crash and applies a −50 reward penalty.</span>
</div>
</div>

<div class="info-card">
<h3>🎯 Reward structure</h3>

All values below are <b>defaults</b> — every one of them is adjustable in the sidebar under
<b>💰 Reward shaping</b>.

<br><br><b>Terminal rewards</b> (once, when the episode ends):

<div class="cond-row note-row">
  <span class="cond-icon">✅</span>
  <span class="cond-text"><b>Successful landing: +100</b> &nbsp;(sidebar: <i>Landing bonus</i>)</span>
</div>
<div class="cond-row crash-row">
  <span class="cond-icon">❌</span>
  <span class="cond-text"><b>Crash / out of bounds: −100</b> &nbsp;(sidebar: <i>Crash penalty</i>)<br>
  An episode that simply runs out of steps mid-air gets <b>neither</b> — it keeps whatever
  shaping reward it accumulated.</span>
</div>

<b>Per-step shaping rewards</b> — the agent is rewarded every step for <i>improving</i> its
situation, and penalised for making it worse. This is why an episode can score positive
without ever landing:

<div class="cond-row note-row">
  <span class="cond-icon">📍</span>
  <span class="cond-text"><b>Getting closer to the pad: positive</b> &nbsp;(weight 100, sidebar: <i>Distance-to-pad penalty</i>)<br>
  Each step, reward = weight × decrease in distance √(x²+y²). Moving away costs the same amount.</span>
</div>
<div class="cond-row note-row">
  <span class="cond-icon">🐢</span>
  <span class="cond-text"><b>Slowing down: positive</b> &nbsp;(weight 100, sidebar: <i>Speed penalty</i>)<br>
  Reward for reducing total speed √(vx²+vy²); speeding up is penalised.</span>
</div>
<div class="cond-row note-row">
  <span class="cond-icon">📐</span>
  <span class="cond-text"><b>Levelling out: positive</b> &nbsp;(weight 100, sidebar: <i>Angle penalty</i>)<br>
  Reward for reducing |tilt|; tilting further is penalised.</span>
</div>
<div class="cond-row note-row">
  <span class="cond-icon">🦵</span>
  <span class="cond-text"><b>Leg contact: +10 per leg</b> &nbsp;(sidebar: <i>Leg-contact bonus</i>)<br>
  A one-time jump in shaping when a leg first touches the ground (lost again if it lifts off).</span>
</div>

<b>Fuel costs</b> (every step an engine fires):

<div class="cond-row crash-row">
  <span class="cond-icon">🔥</span>
  <span class="cond-text"><b>Main engine: −0.30 per step</b> &nbsp;(sidebar: <i>Main engine fuel cost</i>)<br>
  Use thrust sparingly — hovering burns reward fast.</span>
</div>
<div class="cond-row crash-row">
  <span class="cond-icon">↔️</span>
  <span class="cond-text"><b>Side engine: −0.03 per step</b> &nbsp;(sidebar: <i>Side engine fuel cost</i>)<br>
  Side thrusters are cheap; use them freely for attitude control.</span>
</div>
</div>

<div class="info-card">
<h3>🧮 Worked example — one step of braking</h3>

Every step, the app computes a <b>shaping score</b> describing how good the current situation is
(default weights of 100 shown):

<pre style="background:#1e293b;border-radius:7px;padding:12px 16px;color:#a5f3fc;font-size:0.88rem;overflow-x:auto;">shaping = −100 × distance − 100 × speed − 100 × |tilt| + 10 × legs_touching
reward  = (shaping_now − shaping_previous) − fuel</pre>

<b>Scenario:</b> the lander fires the main engine for one step and slows down.

<br><br><b>Before the step:</b> distance to pad = 0.90, speed = 0.80, tilt = 0.05 rad, no legs touching.

<pre style="background:#1e293b;border-radius:7px;padding:12px 16px;color:#cbd5e1;font-size:0.88rem;overflow-x:auto;">shaping_previous = −100(0.90) − 100(0.80) − 100(0.05) = −90 − 80 − 5 = −175.0</pre>

<b>After the step:</b> speed dropped 0.80 → 0.75 (braking), distance dropped 0.90 → 0.88
(still descending), tilt unchanged.

<pre style="background:#1e293b;border-radius:7px;padding:12px 16px;color:#cbd5e1;font-size:0.88rem;overflow-x:auto;">shaping_now = −100(0.88) − 100(0.75) − 100(0.05) = −88 − 75 − 5 = −168.0

reward = (−168.0) − (−175.0) − 0.30 fuel = +7.0 − 0.3 = <b style="color:#4ade80">+6.70</b></pre>

<b>Where the reward came from:</b>

<table style="width:100%;border-collapse:collapse;color:#cbd5e1;font-size:0.91rem;">
<thead>
<tr style="border-bottom:1px solid #334155;">
  <th style="text-align:left;padding:7px 10px;color:#94a3b8;">Component</th>
  <th style="text-align:left;padding:7px 10px;color:#94a3b8;">Before</th>
  <th style="text-align:left;padding:7px 10px;color:#94a3b8;">After</th>
  <th style="text-align:left;padding:7px 10px;color:#94a3b8;">Contribution</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid #1e293b;">
  <td style="padding:6px 10px;">Speed</td><td style="padding:6px 10px;font-family:monospace;">0.80</td>
  <td style="padding:6px 10px;font-family:monospace;">0.75</td>
  <td style="padding:6px 10px;color:#4ade80;">100 × 0.05 = <b>+5.0</b></td>
</tr>
<tr style="border-bottom:1px solid #1e293b;background:#0f1829;">
  <td style="padding:6px 10px;">Distance</td><td style="padding:6px 10px;font-family:monospace;">0.90</td>
  <td style="padding:6px 10px;font-family:monospace;">0.88</td>
  <td style="padding:6px 10px;color:#4ade80;">100 × 0.02 = <b>+2.0</b></td>
</tr>
<tr style="border-bottom:1px solid #1e293b;">
  <td style="padding:6px 10px;">Tilt</td><td style="padding:6px 10px;font-family:monospace;">0.05</td>
  <td style="padding:6px 10px;font-family:monospace;">0.05</td>
  <td style="padding:6px 10px;">0.0</td>
</tr>
<tr style="border-bottom:1px solid #1e293b;background:#0f1829;">
  <td style="padding:6px 10px;">Fuel (main engine)</td><td style="padding:6px 10px;"></td>
  <td style="padding:6px 10px;"></td>
  <td style="padding:6px 10px;color:#ef4444;"><b>−0.3</b></td>
</tr>
<tr>
  <td style="padding:6px 10px;"><b>Total</b></td><td style="padding:6px 10px;"></td>
  <td style="padding:6px 10px;"></td>
  <td style="padding:6px 10px;color:#4ade80;"><b>+6.7</b></td>
</tr>
</tbody>
</table>

<br>
<div class="cond-row note-row">
  <span class="cond-icon">↕️</span>
  <span class="cond-text"><b>The flip side:</b> the same math punishes speeding up. In free fall, speed
  rising 0.75 → 0.80 costs <b>−5</b> on that step even with no engine firing. This is why a gentle,
  controlled descent accumulates a positive total over hundreds of steps without ever touching down.</span>
</div>
<div class="cond-row note-row">
  <span class="cond-icon">🚫</span>
  <span class="cond-text"><b>No reward farming:</b> because the reward is a <i>difference</i>, hovering
  in place earns nothing — zero change means zero shaping reward, while fuel still costs. The agent only
  earns by actually improving distance, speed, or tilt relative to the previous step.</span>
</div>
</div>

<div class="info-card">
<h3>🧠 Observation state (what the agent sees)</h3>

The DQN agent receives an 8-dimensional vector every step from the environment:

<table style="width:100%;border-collapse:collapse;color:#cbd5e1;font-size:0.91rem;">
<thead>
<tr style="border-bottom:1px solid #334155;">
  <th style="text-align:left;padding:7px 10px;color:#94a3b8;">Index</th>
  <th style="text-align:left;padding:7px 10px;color:#94a3b8;">Observation</th>
  <th style="text-align:left;padding:7px 10px;color:#94a3b8;">Range</th>
  <th style="text-align:left;padding:7px 10px;color:#94a3b8;">Meaning</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid #1e293b;">
  <td style="padding:6px 10px;color:#38bdf8;font-weight:700;">0</td>
  <td style="padding:6px 10px;"><b>x</b></td>
  <td style="padding:6px 10px;font-family:monospace;">−1.5 … 1.5</td>
  <td style="padding:6px 10px;">Horizontal position (0 = centre of pad)</td>
</tr>
<tr style="border-bottom:1px solid #1e293b;background:#0f1829;">
  <td style="padding:6px 10px;color:#38bdf8;font-weight:700;">1</td>
  <td style="padding:6px 10px;"><b>y</b></td>
  <td style="padding:6px 10px;font-family:monospace;">−1.5 … 1.5</td>
  <td style="padding:6px 10px;">Vertical position (0 = ground level)</td>
</tr>
<tr style="border-bottom:1px solid #1e293b;">
  <td style="padding:6px 10px;color:#38bdf8;font-weight:700;">2</td>
  <td style="padding:6px 10px;"><b>vx</b></td>
  <td style="padding:6px 10px;font-family:monospace;">−5 … 5</td>
  <td style="padding:6px 10px;">Horizontal velocity</td>
</tr>
<tr style="border-bottom:1px solid #1e293b;background:#0f1829;">
  <td style="padding:6px 10px;color:#38bdf8;font-weight:700;">3</td>
  <td style="padding:6px 10px;"><b>vy</b></td>
  <td style="padding:6px 10px;font-family:monospace;">−5 … 5</td>
  <td style="padding:6px 10px;">Vertical velocity (negative = falling)</td>
</tr>
<tr style="border-bottom:1px solid #1e293b;">
  <td style="padding:6px 10px;color:#38bdf8;font-weight:700;">4</td>
  <td style="padding:6px 10px;"><b>angle</b></td>
  <td style="padding:6px 10px;font-family:monospace;">−π … π</td>
  <td style="padding:6px 10px;">Body angle in radians (0 = upright)</td>
</tr>
<tr style="border-bottom:1px solid #1e293b;background:#0f1829;">
  <td style="padding:6px 10px;color:#38bdf8;font-weight:700;">5</td>
  <td style="padding:6px 10px;"><b>ω (omega)</b></td>
  <td style="padding:6px 10px;font-family:monospace;">−5 … 5</td>
  <td style="padding:6px 10px;">Angular velocity (spin rate)</td>
</tr>
<tr style="border-bottom:1px solid #1e293b;">
  <td style="padding:6px 10px;color:#38bdf8;font-weight:700;">6</td>
  <td style="padding:6px 10px;"><b>left leg</b></td>
  <td style="padding:6px 10px;font-family:monospace;">0 or 1</td>
  <td style="padding:6px 10px;">1 = left leg is touching the ground</td>
</tr>
<tr style="background:#0f1829;">
  <td style="padding:6px 10px;color:#38bdf8;font-weight:700;">7</td>
  <td style="padding:6px 10px;"><b>right leg</b></td>
  <td style="padding:6px 10px;font-family:monospace;">0 or 1</td>
  <td style="padding:6px 10px;">1 = right leg is touching the ground</td>
</tr>
</tbody>
</table>

<br>
<div class="cond-row note-row">
  <span class="cond-icon">🎮</span>
  <span class="cond-text"><b>Actions:</b> 0 = do nothing &nbsp;|&nbsp; 1 = left thruster &nbsp;|&nbsp; 2 = main engine &nbsp;|&nbsp; 3 = right thruster</span>
</div>
</div>

<div class="info-card">
<h3>🎲 Initial state (how each episode starts)</h3>

The starting state is <b>not fixed</b> — it is randomised by the environment on every
episode reset, and training runs are unseeded (so each full run differs too). Three things
vary from episode to episode:

<div class="cond-row note-row">
  <span class="cond-icon">📍</span>
  <span class="cond-text"><b>Starting position</b> — the lander always spawns near the
  top-centre of the screen, with the exact x slightly jittered.</span>
</div>
<div class="cond-row note-row">
  <span class="cond-icon">💥</span>
  <span class="cond-text"><b>Initial velocity</b> — on reset a <b>random force</b> is applied to
  the lander's centre of mass, so it always begins already drifting in a random direction at a
  random speed. It is never handed to the agent motionless — this is the largest source of
  episode-to-episode variation.</span>
</div>
<div class="cond-row note-row">
  <span class="cond-icon">⛰️</span>
  <span class="cond-text"><b>Terrain</b> — the jagged moon surface on either side is regenerated
  each episode. The landing pad stays fixed at the centre, but the surrounding hills change.</span>
</div>

What stays constant: the pad location, gravity, and every physics parameter you set in the sidebar.

<div class="cond-row note-row">
  <span class="cond-icon">🎯</span>
  <span class="cond-text"><b>Why randomise?</b> It forces the agent to learn a <i>general</i>
  landing policy that works from any starting drift and over any terrain, rather than memorising one
  trajectory. It is also why single-episode rewards stay noisy even late in training (a bad initial
  impulse is simply harder to recover from) — so the smoothed/averaged reward curve, not any one
  episode, is the signal to watch.</span>
</div>
</div>

<div class="info-card">
<h3>🕹️ Controls (Manual Play tab)</h3>

| Key | Action |
|-----|--------|
| **↑ Arrow Up** | Main engine (thrust upward) |
| **← Arrow Left** | Left side engine (rotates right / drifts left) |
| **→ Arrow Right** | Right side engine (rotates left / drifts right) |
| **SPACE** | Start / Pause |
| **R** | Restart game |

**Hold to fire:** hold a key down to keep the engine running; releasing the key cuts the engine immediately.
</div>
""", unsafe_allow_html=True)


# ── auto-refresh while training ───────────────────────────────────────────────
if st.session_state.training:
    time.sleep(1.0)
    st.rerun()
