"""Plot a recorded hover test. Requires matplotlib for this review command only."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(source, output):
    data = json.loads(source.read_text())
    physics = data["physics"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.88, data["jumps"]))
    rmse = []
    for jump in range(1, data["jumps"] + 1):
        samples = [r for r in data["measurements"] if r["jump"] == jump]
        age = np.array([r["age"] for r in samples])
        rise = data["presses"][jump - 1]["launch_y"] - np.array([r["y"] for r in samples])
        axes[0].plot(age, rise, color=colors[jump - 1], lw=1, alpha=0.65)
        errors = np.array([[r["error_x"], r["error_y"]] for r in samples])
        rmse.append(np.sqrt(np.mean(errors**2, axis=0)))
    age = np.linspace(0.06, 1.02, 150)
    axes[0].plot(
        age,
        physics["jump"] * age - physics["gravity"] * age**2 / 2,
        color="#151c26",
        lw=2.5,
        linestyle="--",
        label="Frozen model",
    )
    axes[0].plot([], [], color=colors[8], lw=1.8, label="Observed: 20 jumps")
    axes[0].set(
        title="Predicted parabola vs observed jumps",
        xlabel="Seconds after reset",
        ylabel="Rise from predicted launch position (pixels)",
        xlim=(0, 1.04),
    )
    axes[0].legend(frameon=False, loc="lower center")
    rmse = np.array(rmse)
    n = np.arange(1, data["jumps"] + 1)
    axes[1].bar(n - 0.18, rmse[:, 0], 0.36, color="#478ab0", label="Horizontal")
    axes[1].bar(n + 0.18, rmse[:, 1], 0.36, color="#d8913a", label="Vertical")
    axes[1].set(
        title="Prediction error for each full jump",
        xlabel="Jump number",
        ylabel="RMS error (pixels)",
        xticks=n,
    )
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(axis="y", alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
    total = data["rmse_xy"]
    fig.suptitle(
        f"{data['jumps']} completed jumps | RMS: x {total[0]:.1f}px, y {total[1]:.1f}px"
        " | board normalized to 420px wide",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def plot_spatial(source, output):
    data = json.loads(source.read_text())
    fig, axes = plt.subplots(4, 5, figsize=(14, 9), constrained_layout=True)
    residuals = []
    for jump, ax in enumerate(axes.ravel(), 1):
        samples = [r for r in data["measurements"] if r["jump"] == jump]
        x = np.array([r["x"] for r in samples])
        y = np.array([r["y"] for r in samples])
        curve = np.polyfit(x, y, 2)
        error = float(np.sqrt(np.mean((np.polyval(curve, x) - y) ** 2)))
        residuals.append(error)
        domain = np.linspace(x.min(), x.max(), 100)
        ax.scatter(x, y, s=7, color="#227591", alpha=0.75)
        ax.plot(domain, np.polyval(curve, domain), color="#dd8223", lw=1.6)
        ax.set_title(f"Jump {jump}: fit RMS {error:.1f}px", fontsize=9)
        ax.invert_yaxis()
        ax.grid(alpha=0.15)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7)
    fig.suptitle(
        "20 jumps: camera-corrected observations and fitted parabolas\n"
        f"Median fit residual {np.median(residuals):.2f}px | "
        "Individual curve fits, not ahead-of-time forecasts",
        fontsize=14,
        fontweight="bold",
    )
    fig.supxlabel("Horizontal position (board normalized to 420px wide)")
    fig.supylabel("Vertical position with camera movement subtracted")
    fig.savefig(output, dpi=160, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spatial", action="store_true")
    args = parser.parse_args()
    if args.spatial:
        plot_spatial(args.source, args.output)
    else:
        plot(args.source, args.output)
