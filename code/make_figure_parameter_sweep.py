"""
make_figure_parameter_sweep.py
================================
Figures for the parameter-space generalization sweep (Section 4.3):
  Figure 5: one-at-a-time sensitivity of ARI_A/ARI_B/ARI_C/Silhouette
             to each of the 4 mechanism-strength parameters.
  Figure 6: 2D grid (relational_fraction x ratio) heatmaps of ARI_B and
             of the Theorem 1 safety margin (bound - observed error),
             confirming the bound holds everywhere tested.

Reads only parameter_sweep_results.json; reruns nothing.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
BASE = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE, "..", "data")
FIG_DIR = os.path.join(BASE, "..", "figures")

PALETTE = {
    "naive": "#E15759", "informed": "#4E79A7", "gold": "#F1B434",
    "gray": "#9DA3AA", "green": "#59A14F", "purple": "#B07AA1",
    "dark": "#2B2F36", "grid": "#E4E7EB", "bg_panel": "#FAFAFB",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12,
    "axes.titlesize": 13.5, "axes.titleweight": "bold",
    "axes.labelsize": 11.5, "axes.edgecolor": "#B8BEC6",
    "axes.linewidth": 0.9, "axes.grid": True,
    "grid.color": PALETTE["grid"], "grid.linewidth": 0.8,
    "axes.axisbelow": True, "xtick.color": PALETTE["dark"],
    "ytick.color": PALETTE["dark"], "text.color": PALETTE["dark"],
    "axes.labelcolor": PALETTE["dark"], "legend.frameon": False,
    "legend.fontsize": 9.5, "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3)


def savefig(name):
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{name}", dpi=180, bbox_inches="tight")
    plt.close()
    print("wrote", name)


d = json.load(open(f"{DATA_DIR}/parameter_sweep_results.json"))

# ===========================================================================
# Figure 5: one-at-a-time sensitivity
# ===========================================================================
rows = d["one_at_a_time"]["summary"]
params = ["dist_std", "ratio", "relational_fraction", "scale_fraction"]
titles = {
    "dist_std": "Mechanism A spread (dist_std)\noriginal = 1.6",
    "ratio": "Mechanism B ratio\noriginal = 2.3",
    "relational_fraction": "Mechanism B minority fraction\noriginal = 0.12",
    "scale_fraction": "Mechanism C minority fraction\noriginal = 0.04",
}
orig_val = {"dist_std": 1.6, "ratio": 2.3, "relational_fraction": 0.12, "scale_fraction": 0.04}

fig, axes = plt.subplots(1, 4, figsize=(16, 3.4))
for ax, p in zip(axes, params):
    sub = sorted([r for r in rows if r["param"] == p], key=lambda r: r["value"])
    xs = [r["value"] for r in sub]
    ax.plot(xs, [r["mean_ari_A"] for r in sub], "o-", color=PALETTE["naive"], label="ARI (A, distance)")
    ax.plot(xs, [r["mean_ari_B"] for r in sub], "o-", color=PALETTE["gold"], label="ARI (B, relational)")
    ax.plot(xs, [r["mean_ari_C"] for r in sub], "o-", color=PALETTE["purple"], label="ARI (C, scale)")
    ax.plot(xs, [r["mean_silhouette"] for r in sub], "s--", color=PALETTE["informed"], label="Silhouette")
    ax.axvline(orig_val[p], color=PALETTE["gray"], linestyle=":", linewidth=1.2)
    ax.set_title(titles[p], fontsize=10.5)
    ax.set_ylim(-0.05, 0.65)
    style_axes(ax)
axes[0].set_ylabel("ARI / Silhouette")
axes[0].legend(loc="upper left", fontsize=8, ncol=1)
fig.suptitle("Figure 9. One-at-a-time sensitivity of naive-pipeline recovery (ARI) and\n"
             "Silhouette to each mechanism-strength parameter (N=20 seeds/level; dotted line = original paper value)",
             fontsize=11, y=1.08)
savefig("fig09_parameter_sensitivity.png")

# ===========================================================================
# Figure 6: 2D grid heatmaps
# ===========================================================================
grid_rows = d["grid_2d"]["summary"]
fracs = d["grid_2d"]["relational_fraction_levels"]
ratios = d["grid_2d"]["ratio_levels"]

ari_b_grid = np.zeros((len(fracs), len(ratios)))
margin_grid = np.zeros((len(fracs), len(ratios)))
for r in grid_rows:
    i = fracs.index(r["relational_fraction"])
    j = ratios.index(r["ratio"])
    ari_b_grid[i, j] = r["mean_ari_B"]
    margin_grid[i, j] = r["mean_theorem1_bound_evr"] - r["mean_observed_evr_error"]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

im0 = axes[0].imshow(ari_b_grid, cmap="RdYlGn", vmin=0, vmax=0.4, aspect="auto")
axes[0].set_xticks(range(len(ratios))); axes[0].set_xticklabels(ratios)
axes[0].set_yticks(range(len(fracs))); axes[0].set_yticklabels(fracs)
axes[0].set_xlabel("ratio"); axes[0].set_ylabel("relational_fraction")
axes[0].set_title("Naive-pipeline ARI, mechanism B\n(green = more recoverable; still far below\nhand-informed ceiling of 0.915 everywhere)", fontsize=10)
for i in range(len(fracs)):
    for j in range(len(ratios)):
        axes[0].text(j, i, f"{ari_b_grid[i,j]:.2f}", ha="center", va="center", fontsize=8, color=PALETTE["dark"])
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(margin_grid, cmap="Blues", vmin=0, vmax=0.06, aspect="auto")
axes[1].set_xticks(range(len(ratios))); axes[1].set_xticklabels(ratios)
axes[1].set_yticks(range(len(fracs))); axes[1].set_yticklabels(fracs)
axes[1].set_xlabel("ratio"); axes[1].set_ylabel("relational_fraction")
axes[1].set_title("Theorem 1 safety margin\n(bound - observed EVR error;\npositive everywhere = bound holds in all 30 cells)", fontsize=10)
for i in range(len(fracs)):
    for j in range(len(ratios)):
        axes[1].text(j, i, f"{margin_grid[i,j]:.3f}", ha="center", va="center", fontsize=8, color=PALETTE["dark"])
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

fig.suptitle("Figure 10. 2D parameter grid (relational_fraction x ratio), N=15 seeds/cell, 450 runs total",
             fontsize=11, y=1.03)
savefig("fig10_parameter_grid_theorem1.png")
