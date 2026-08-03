"""
make_figures_v4_radial_update.py
==================================
Fourth pass on figures, triggered by Section 5.6's new radial-distance-
from-centroid baseline (radial_centroid_baseline.py), added to stress-test
and correct the paper's own "invisible to Euclidean distance" language.

This script regenerates ONLY the two figures affected by that addition,
in the same shared visual style as make_figures_v3.py, reading only
already-saved JSON outputs in data/ (including the new
radial_centroid_baseline_results.json) -- it does not re-run any
experiment, so every number matches what Section 5.6 and the revised
Section 6.1/6.5 text already report.

  Figure 9  -- adds a third bar (radial-centroid oracle ARI) alongside
               the existing naive / hand-informed bars for mechanisms
               B and C.
  Figure 17 -- adds a new row ("Radial-centroid (oracle ceiling)*") to
               the method x mechanism detection matrix.

All other figures (1-8, 10-16) are unchanged and are NOT regenerated
here; see make_figures_v3.py for those.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
BASE = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE, "..", "data")
FIG_DIR = os.path.join(BASE, "..", "figures")

# ---------------------------------------------------------------------
# Shared visual style (identical to make_figures_v3.py)
# ---------------------------------------------------------------------
PALETTE = {
    "naive": "#E15759",
    "informed": "#4E79A7",
    "gold": "#F1B434",
    "gray": "#9DA3AA",
    "green": "#59A14F",
    "purple": "#B07AA1",
    "dark": "#2B2F36",
    "grid": "#E4E7EB",
    "bg_panel": "#FAFAFB",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 13.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 11.5,
    "axes.edgecolor": "#B8BEC6",
    "axes.linewidth": 0.9,
    "axes.grid": True,
    "grid.color": PALETTE["grid"],
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "xtick.color": PALETTE["dark"],
    "ytick.color": PALETTE["dark"],
    "text.color": PALETTE["dark"],
    "axes.labelcolor": PALETTE["dark"],
    "legend.frameon": False,
    "legend.fontsize": 10,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def style_axes(ax, top=False, right=False):
    ax.spines["top"].set_visible(top)
    ax.spines["right"].set_visible(right)
    ax.tick_params(length=3)


def value_labels(ax, bars, fmt="{:.2f}", dy=0.012, size=9.5, color=None):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h),
                    textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=size, color=color or PALETTE["dark"], fontweight="medium")


def savefig(name):
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{name}", dpi=180, bbox_inches="tight")
    plt.close()
    print("wrote", name)


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
results = json.load(open(f"{DATA_DIR}/results.json"))
baselines = json.load(open(f"{DATA_DIR}/stronger_baselines_results.json"))
ics = json.load(open(f"{DATA_DIR}/ics_kurtosis_screening_results.json"))
radial = json.load(open(f"{DATA_DIR}/radial_centroid_baseline_results.json"))

# ======================================================================
# Figure 9 (revised) -- ARI comparison, naive vs hand-informed vs
# radial-centroid oracle, seed 42 / N=80 oracle numbers
# ======================================================================
mechanisms = ["Distance\n(4 main groups)", "Relational\nsubgroup", "Scale/density\nsubgroup"]
naive_vals = [results["naive"]["ARI_class_label__vs__naive_k4"],
              results["naive"]["ARI_relational_hidden__vs__naive_k4"],
              results["naive"]["ARI_scale_hidden__vs__naive_k4"]]
informed_vals = [results["informed"]["ARI_class_label__distance_kmeans_on_feature_1_2"],
                  results["informed"]["ARI_relational_hidden__polar_dbscan_on_feature_3_4"],
                  results["informed"]["ARI_scale_hidden__knn_density_gmm_on_feature_5_6"]]
# Radial-centroid oracle: not applicable to mechanism A (distance mechanism
# already has an off-center compact blob per class; radius-from-global-
# centroid is not the relevant statistic there), so plot as n/a for A.
radial_vals = [np.nan,
               radial["mechanism_B_relational"]["radius_oracle_ari_mean"],
               radial["mechanism_C_scale"]["radius_oracle_ari_mean"]]
ceiling_scale = results["informed"]["ARI_scale_hidden__theoretical_ceiling_same_signal"]

fig, ax = plt.subplots(figsize=(9.4, 5.3))
x = np.arange(3)
w = 0.27
b1 = ax.bar(x - w, naive_vals, w, label="Naive PCA+KMeans", color=PALETTE["naive"],
            edgecolor="white", linewidth=0.6, zorder=3)
b2 = ax.bar(x, informed_vals, w, label="Hand-informed (mechanism-aware)", color=PALETTE["informed"],
            edgecolor="white", linewidth=0.6, zorder=3)
b3 = ax.bar(x + w, radial_vals, w, label="Radial-distance-from-centroid\n(oracle ceiling, Sec. 5.6)",
            color=PALETTE["green"], edgecolor="white", linewidth=0.6, zorder=3)
ax.scatter([2], [ceiling_scale], marker="*", s=260, color=PALETTE["gold"],
           edgecolor=PALETTE["dark"], linewidth=0.7, zorder=5,
           label="Theoretical ceiling (scale mechanism,\nkNN-density signal)")
value_labels(ax, b1)
value_labels(ax, b2)
value_labels(ax, [b for b in b3 if not np.isnan(b.get_height())])
ax.set_ylabel("ARI (vs. ground truth)")
ax.set_xticks(x)
ax.set_xticklabels(mechanisms)
ax.set_ylim(-0.05, 1.12)
ax.axhline(0, color=PALETTE["dark"], linewidth=0.8, zorder=2)
ax.legend(loc="upper left", ncol=1, fontsize=8.8)
ax.set_title("Naive vs. hand-informed vs. radial-centroid recovery (seed 42 / N=80 oracle)")
style_axes(ax)
savefig("fig16_ari_comparison.png")

# ======================================================================
# Figure 17 (revised) -- Method x Mechanism detection-matrix heatmap,
# with a new "Radial-centroid (oracle)" row
# ======================================================================
row_labels = [
    "Naive PCA + KMeans",
    "Sparse PCA",
    "Kernel PCA (RBF)",
    "t-SNE",
    "Hand-informed\n(mechanism-specific)",
    "ICS kurtosis routing\n(oracle ceiling)*",
    "Radial-centroid\n(oracle ceiling)*",
]
col_labels = ["Mechanism A\n(distance)", "Mechanism B\n(relational)", "Mechanism C\n(scale/density)"]

matrix = np.array([
    [results["naive"]["ARI_class_label__vs__naive_k4"],
     results["naive"]["ARI_relational_hidden__vs__naive_k4"],
     results["naive"]["ARI_scale_hidden__vs__naive_k4"]],
    [baselines["sparse_pca"]["ari_at_k4"],
     baselines["sparse_pca"]["ari_relational_hidden_at_k4"],
     baselines["sparse_pca"]["ari_scale_hidden_at_k4"]],
    [baselines["kernel_pca_rbf"]["ari_at_k4"],
     baselines["kernel_pca_rbf"]["ari_relational_hidden_at_k4"],
     baselines["kernel_pca_rbf"]["ari_scale_hidden_at_k4"]],
    [baselines["tsne"]["ari_at_k4"],
     baselines["tsne"]["ari_relational_hidden_at_k4"],
     baselines["tsne"]["ari_scale_hidden_at_k4"]],
    [results["informed"]["ARI_class_label__distance_kmeans_on_feature_1_2"],
     results["informed"]["ARI_relational_hidden__polar_dbscan_on_feature_3_4"],
     results["informed"]["ARI_scale_hidden__knn_density_gmm_on_feature_5_6"]],
    [np.nan,
     ics["multi_seed_summary_N80"]["mean_oracle_ari_true_relational_pair"],
     ics["multi_seed_summary_N80"]["mean_oracle_ari_true_scale_pair_negative_control"]],
    [np.nan,
     radial["mechanism_B_relational"]["radius_oracle_ari_mean"],
     radial["mechanism_C_scale"]["radius_oracle_ari_mean"]],
])

fig, ax = plt.subplots(figsize=(8.8, 7.2))
masked = np.ma.masked_invalid(matrix)
cmap = plt.cm.RdYlGn.copy()
cmap.set_bad(color="#E4E7EB")
im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect="auto")

ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels, fontsize=10.5)
ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels, fontsize=10)
ax.xaxis.set_ticks_position("top"); ax.xaxis.set_label_position("top")

for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        v = matrix[i, j]
        if np.isnan(v):
            ax.text(j, i, "n/a", ha="center", va="center", fontsize=10.5, color="#6B7280")
        else:
            txt_color = "white" if (v > 0.62 or v < 0.18) else "#1A1D21"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=11.5,
                    color=txt_color, fontweight="bold")

for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=2.5)
ax.tick_params(which="minor", length=0)
ax.tick_params(which="major", length=0)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cbar.set_label("ARI (vs. ground truth)")
ax.set_title("Which methods see which hiding mechanism? (all numbers from this paper)",
              fontsize=12.5, fontweight="bold", pad=14)
fig.text(0.02, -0.03,
         "*ICS and radial-centroid rows report oracle ceilings on their respective signals (Sections 5.5, 5.6),\n"
         "not automatic end-to-end results; all other rows are fully automatic, label-free pipeline outputs\n"
         "at k=4, seed 42. Radial-centroid n/a for mechanism A: not the relevant statistic for an already-\n"
         "off-center, compact per-class blob.",
         fontsize=8.4, color="#6B7280")
savefig("fig26_detection_matrix_summary.png")

print("Figures 16 and 26 regenerated with the Section 5.6 radial-centroid baseline included.")
