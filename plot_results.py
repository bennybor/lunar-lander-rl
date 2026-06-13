"""Plot training curves from Stable-Baselines3 monitor logs."""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy


def smooth(y, window=50):
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="valid")


def plot(log_dir: str, title: str = "LunarLander Training"):
    x, y = ts2xy(load_results(log_dir), "timesteps")
    plt.figure(figsize=(10, 4))
    plt.plot(x, y, alpha=0.3, color="steelblue", label="raw")
    if len(y) >= 50:
        plt.plot(x[49:], smooth(y), color="steelblue", label="smoothed (50 ep)")
    plt.xlabel("Timesteps")
    plt.ylabel("Episode reward")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "training_curve.png"), dpi=150)
    plt.show()
    print(f"Plot saved to {log_dir}/training_curve.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default="logs/")
    args = parser.parse_args()
    plot(args.log_dir)
