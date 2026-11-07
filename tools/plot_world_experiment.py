"""Optional publication-style figure from a saved experiment (requires matplotlib)."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results")
    parser.add_argument("output_prefix")
    args = parser.parse_args()
    data = json.loads(Path(args.results).read_text(encoding="utf-8"))["aggregates"]
    keys = ("normal", "short_outage", "shifted")
    labels = ("No outage", "Short outage", "Shifted")
    baseline_mae = [data[k]["offline_same_decisions"]["errors"]["world"]["completion_mae_ticks"] for k in keys]
    learned_mae = [data[k]["offline_same_decisions"]["errors"]["learned"]["completion_mae_ticks"] for k in keys]
    baseline_flow = [data[k]["policies"]["world"]["mean_restricted_flow_ticks"] for k in keys]
    learned_flow = [data[k]["policies"]["learned"]["mean_restricted_flow_ticks"] for k in keys]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.hashsalt": "wm-sample-transport"})
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), constrained_layout=False)
    x, width = np.arange(len(keys)), 0.34
    for ax, first, second, title in ((axes[0], baseline_mae, learned_mae, "Prediction error on the same decisions"),
                                     (axes[1], baseline_flow, learned_flow, "Closed-loop task flow")):
        bars1 = ax.bar(x - width / 2, first, width, label="Online mean baseline", color="#355a7a")
        bars2 = ax.bar(x + width / 2, second, width, label="Learned residual", color="#d4743c")
        ax.bar_label(bars1, fmt="%.1f", padding=3, fontsize=8)
        ax.bar_label(bars2, fmt="%.1f", padding=3, fontsize=8)
        ax.set_xticks(x, labels)
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.set_ylabel("Simulation steps (10 s assumed)")
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, max(first + second) * 1.22)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("Simulated sample transport: prediction versus dispatch", x=.06, ha="left", fontsize=14, fontweight="bold")
    fig.text(.06, .03, "Synthetic tasks and disturbances; 10 held-out seeds per condition. Timing is an estimate, not clinical data.",
             fontsize=9, color="#526479")
    fig.subplots_adjust(left=.07, right=.98, top=.82, bottom=.17, wspace=.35)
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".svg"), metadata={"Date": None})
    fig.savefig(prefix.with_suffix(".png"), dpi=160)
    print("Wrote", prefix.with_suffix(".svg"), "and", prefix.with_suffix(".png"))


if __name__ == "__main__":
    main()
