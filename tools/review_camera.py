"""Plot measured camera onset and the recorded 107-point bar collision."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def review(onset, failure, output):
    source = np.load(onset)
    data, target = source["data"], source["target"]
    tau = float(source["tau"][np.argmin(source["loss"])])
    scene = [0.0]
    for i in range(1, len(data)):
        dt = data[i, 0] - data[i - 1, 0]
        velocity = (target[i] - target[i - 1]) / dt
        scene.append(
            target[i]
            - tau * velocity
            + (scene[-1] - target[i - 1] + tau * velocity) * np.exp(-dt / tau)
        )
    scene = np.array(scene)
    began = data[np.argmax(data[:, 3] > 0), 0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    t = data[:, 0] - began
    axes[0].plot(t, target, "--", color="#87919c", label="Instant camera target")
    axes[0].plot(t, scene, color="#287b95", lw=2, label=f"Easing fit ({tau * 1000:.0f}ms)")
    axes[0].scatter(
        t, data[:, 3], color="#b77024", s=20, label="Measured static-bar shift", zorder=3
    )
    axes[0].set(
        xlim=(-0.08, 0.22),
        xlabel="Seconds from observed camera start",
        ylabel="Background displacement (pixels)",
        title="Measured camera start",
    )
    axes[0].legend(frameon=False, fontsize=8)
    report = json.loads(failure.read_text())
    path = np.array(report["path"])
    pt = report["planner_latency"] + report["step"] * np.arange(1, len(path) + 1)
    predicted = (
        report["bar_y_initial"]
        + report["bar_height"]
        + np.maximum.accumulate(np.maximum(0, 371.5 - path[:, 1]))
    )
    actual = np.array(
        [[m["t"], m["bar_y"] + report["bar_height"], m["body_top"]] for m in report["measured"]]
    )
    axes[1].plot(pt, predicted, "--", color="#287b95", lw=2, label="Old plan: bar underside")
    axes[1].plot(
        actual[:, 0], actual[:, 1], "o-", color="#c85843", ms=4, label="Measured bar underside"
    )
    axes[1].plot(actual[:, 0], actual[:, 2], color="#303a42", label="Character top")
    axes[1].set(
        xlim=(0, 0.43),
        ylim=(150, 365),
        xlabel="Seconds after decision snapshot",
        ylabel="Screen Y (pixels)",
        title="Why the 107-point run ended",
    )
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    contact = actual[actual[:, 1] >= actual[:, 2]]
    if len(contact):
        axes[1].scatter(contact[0, 0], contact[0, 2], color="#c85843", s=65, zorder=5)
        axes[1].annotate(
            "Bar reaches the character",
            (contact[0, 0], contact[0, 2]),
            xytext=(0.11, 340),
            fontsize=9,
            arrowprops={"arrowstyle": "->", "color": "#c85843"},
        )
    for ax in axes:
        ax.grid(alpha=0.15)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Camera smoothing and the recorded bar collision", fontsize=15, fontweight="bold")
    fig.savefig(output, dpi=170, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onset", type=Path, required=True)
    parser.add_argument("--failure", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    review(args.onset, args.failure, args.output)
