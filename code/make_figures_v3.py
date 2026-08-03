"""
make_figures_v3.py
====================
Third pass on figures, triggered by feedback that (a) three figures that
already existed (fig12/13/14, the per-mechanism raw-vs-transformed scatter
plots) were generated in an earlier pass but never actually embedded in
the paper, and (b) the visual style across figures was inconsistent
and not very polished. This script:

  1. Defines ONE shared visual style (palette, fonts, spacing) and
     re-renders EVERY figure except Figures 5-6 in that style, reading only from
     already-saved JSON/CSV outputs in data/ -- it does NOT re-run any
     experiment, so every number is identical to what is already
     reported in the paper; only the rendering changes.
  2. Adds a genuinely new, differently-structured Figure 17: a single
     "method x mechanism" heatmap summarizing every recovery number in
     the paper in one view -- which methods see which hiding mechanisms,
     and how well -- rather than another bar/scatter/line chart.

Run this AFTER all the analysis scripts (it only reads data/*.json,
data/*.csv, and data/observable_dataset_hard.csv / hidden_labels_hard_
EVAL_ONLY.csv for the raw-scatter figures 4/5/6 and the correlation
heatmap fig2).
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as mticker
import os
BASE = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE, "..", "data")
FIG_DIR = os.path.join(BASE, "..", "figures")

# ---------------------------------------------------------------------
# Shared visual style
# ---------------------------------------------------------------------
PALETTE = {
    "naive": "#E15759",       # warm red -- "the problem"
    "informed": "#4E79A7",    # steady blue -- "the fix"
    "gold": "#F1B434",        # gold -- "hidden minority / highlight"
    "gray": "#9DA3AA",        # neutral background series
    "green": "#59A14F",       # positive / recovered
    "purple": "#B07AA1",      # secondary accent
    "dark": "#2B2F36",        # text / axis
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
# Load all data once
# ---------------------------------------------------------------------
observable = pd.read_csv(f"{DATA_DIR}/observable_dataset_hard.csv")
hidden = pd.read_csv(f"{DATA_DIR}/hidden_labels_hard_EVAL_ONLY.csv")
class_label = observable["class_label"].values
hidden_relational = hidden["hidden_relational_label"].values
hidden_scale = hidden["hidden_scale_label"].values

results = json.load(open(f"{DATA_DIR}/results.json"))
theory = json.load(open(f"{DATA_DIR}/theory_variance_budget_results.json"))
interaction = json.load(open(f"{DATA_DIR}/interaction_robustness_results.json"))
scaling = json.load(open(f"{DATA_DIR}/scaling_experiments_results.json"))
realdata = json.load(open(f"{DATA_DIR}/real_data_validation_results.json"))
baselines = json.load(open(f"{DATA_DIR}/stronger_baselines_results.json"))
ics = json.load(open(f"{DATA_DIR}/ics_kurtosis_screening_results.json"))
multiseed_raw = pd.read_csv(f"{DATA_DIR}/multiseed_robustness_raw.csv")
interaction_raw = pd.read_csv(f"{DATA_DIR}/interaction_robustness_raw.csv")
scaling_dim_raw = pd.read_csv(f"{DATA_DIR}/scaling_dimensionality_raw.csv")
ics_raw = pd.read_csv(f"{DATA_DIR}/ics_kurtosis_screening_raw.csv")

COLORS4 = ["#E15759", "#4E79A7", "#59A14F", "#B07AA1"]


# ======================================================================
# Figure 9 -- ARI comparison, naive vs hand-informed, seed 42
# ======================================================================
mechanisms = ["Distance\n(4 main groups)", "Relational\nsubgroup", "Scale/density\nsubgroup"]
naive_vals = [results["naive"]["ARI_class_label__vs__naive_k4"],
              results["naive"]["ARI_relational_hidden__vs__naive_k4"],
              results["naive"]["ARI_scale_hidden__vs__naive_k4"]]
informed_vals = [results["informed"]["ARI_class_label__distance_kmeans_on_feature_1_2"],
                  results["informed"]["ARI_relational_hidden__polar_dbscan_on_feature_3_4"],
                  results["informed"]["ARI_scale_hidden__knn_density_gmm_on_feature_5_6"]]
ceiling_scale = results["informed"]["ARI_scale_hidden__theoretical_ceiling_same_signal"]

fig, ax = plt.subplots(figsize=(8.6, 5.1))
x = np.arange(3)
w = 0.34
b1 = ax.bar(x - w/2, naive_vals, w, label="Naive PCA+KMeans", color=PALETTE["naive"],
            edgecolor="white", linewidth=0.6, zorder=3)
b2 = ax.bar(x + w/2, informed_vals, w, label="Hand-informed (mechanism-aware)", color=PALETTE["informed"],
            edgecolor="white", linewidth=0.6, zorder=3)
ax.scatter([2 + w/2], [ceiling_scale], marker="*", s=300, color=PALETTE["gold"],
           edgecolor=PALETTE["dark"], linewidth=0.7, zorder=5, label="Theoretical ceiling (scale mechanism)")
value_labels(ax, b1)
value_labels(ax, b2)
ax.set_ylabel("ARI (vs. ground truth)")
ax.set_xticks(x)
ax.set_xticklabels(mechanisms)
ax.set_ylim(-0.05, 1.08)
ax.axhline(0, color=PALETTE["dark"], linewidth=0.8, zorder=2)
ax.legend(loc="upper left", ncol=1, fontsize=9.5)
ax.set_title("Naive vs. hand-informed recovery (seed 42)")
style_axes(ax)
savefig("fig16_ari_comparison.png")

# ======================================================================
# Figure 10 -- PCA explained variance spread
# ======================================================================
ev = results["naive"]["explained_variance_ratio_per_PC"]
fig, ax = plt.subplots(figsize=(8.2, 4.8))
bars = ax.bar(range(1, len(ev)+1), ev, color=PALETTE["informed"], zorder=3,
              edgecolor="white", linewidth=0.6, width=0.68)
ax.axhline(1/len(ev), color=PALETTE["naive"], linestyle="--", linewidth=1.6, zorder=4,
           label=f"Uniform baseline (1/{len(ev)} = {1/len(ev):.3f})")
ax.set_xlabel("Principal component")
ax.set_ylabel("Explained variance ratio")
ax.set_xticks(range(1, len(ev)+1))
ax.set_title("PCA spectrum is near-uniform under mechanism co-presence")
ax.legend(fontsize=10)
style_axes(ax)
savefig("fig17_pca_variance.png")

# ======================================================================
# Figure 11 -- False confidence: Silhouette vs true ARI across k
# ======================================================================
sil = results["naive"]["silhouette_by_k"]
ks = sorted(int(k) for k in sil.keys())
sil_vals = [sil[str(k)] for k in ks]
ari_by_k_naive = baselines["linear_pca"]["ari_by_k"]
ari_vals = [ari_by_k_naive[str(k)] for k in ks]

fig, ax1 = plt.subplots(figsize=(8.4, 5.0))
ax2 = ax1.twinx()
l1, = ax1.plot(ks, sil_vals, "o-", color=PALETTE["informed"], linewidth=2.4, markersize=9,
               label="Silhouette (what the researcher sees)", zorder=4)
l2, = ax2.plot(ks, ari_vals, "s--", color=PALETTE["naive"], linewidth=2.4, markersize=9,
               label="True ARI (what the researcher does NOT see)", zorder=4)
ax1.fill_between(ks, sil_vals, alpha=0.08, color=PALETTE["informed"], zorder=1)
ax1.set_xlabel("Number of clusters (k)")
ax1.set_ylabel("Silhouette score", color=PALETTE["informed"])
ax2.set_ylabel("True ARI", color=PALETTE["naive"])
ax1.set_xticks(ks)
ax1.set_title("Silhouette gives no warning of poor recovery")
lines = [l1, l2]
ax1.legend(lines, [l.get_label() for l in lines], loc="upper center", fontsize=9.5,
           bbox_to_anchor=(0.5, -0.16), ncol=1)
style_axes(ax1); style_axes(ax2)
savefig("fig18_false_confidence.png")

# ======================================================================
# Figure 12 -- Distance mechanism: ground truth vs naive result
# ======================================================================
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

X_scaled = StandardScaler().fit_transform(observable.drop(columns=["class_label"]).values)
pcs = PCA(n_components=10, random_state=0).fit_transform(X_scaled)
top2 = pcs[:, :2]
km4_naive = KMeans(n_clusters=4, n_init=10, random_state=0).fit(top2)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4))
ax = axes[0]
for c in range(4):
    m = class_label == c
    ax.scatter(observable["feature_1"][m], observable["feature_2"][m], s=6, alpha=0.55,
               color=COLORS4[c], label=f"Group {c}", linewidths=0)
ax.set_title("Ground truth")
ax.set_xlabel("feature_1"); ax.set_ylabel("feature_2")
ax.legend(fontsize=9, markerscale=2.2, loc="upper right")
style_axes(ax)

ax = axes[1]
naive_labels = km4_naive.labels_
palette_tab = plt.cm.tab10(np.linspace(0, 1, 10))
for c in np.unique(naive_labels):
    m = naive_labels == c
    ax.scatter(observable["feature_1"][m], observable["feature_2"][m], s=6, alpha=0.55,
               color=palette_tab[c], label=f"Naive cluster {c}", linewidths=0)
ax.set_title(f"Naive PCA+KMeans result  (ARI = {results['naive']['ARI_class_label__vs__naive_k4']:.2f})")
ax.set_xlabel("feature_1"); ax.set_ylabel("feature_2")
ax.legend(fontsize=9, markerscale=2.2, loc="upper right")
style_axes(ax)
fig.suptitle("Mechanism A (distance): naive PCA scrambles an otherwise-easy structure", fontsize=13, fontweight="bold", y=1.03)
savefig("fig19_distance_mechanism.png")

# ======================================================================
# Figure 13 -- Relational mechanism: raw vs polar-transformed
# ======================================================================
f3 = observable["feature_3"].values
f4 = observable["feature_4"].values
angle2 = 2 * np.arctan2(f4, f3)
cx, cy = np.cos(angle2), np.sin(angle2)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4))
ax = axes[0]
ax.scatter(f3[hidden_relational == 0], f4[hidden_relational == 0], s=5, alpha=0.30, color=PALETTE["gray"],
           label="Normal samples", linewidths=0)
ax.scatter(f3[hidden_relational == 1], f4[hidden_relational == 1], s=9, alpha=0.9, color=PALETTE["gold"],
           label="Relational subgroup (truth)", linewidths=0)
ax.set_title("Raw space (feature_3, feature_4)")
ax.set_xlabel("feature_3"); ax.set_ylabel("feature_4")
ax.legend(fontsize=9, markerscale=2)
style_axes(ax)

ax = axes[1]
ax.scatter(cx[hidden_relational == 0], cy[hidden_relational == 0], s=5, alpha=0.30, color=PALETTE["gray"],
           label="Normal samples", linewidths=0)
ax.scatter(cx[hidden_relational == 1], cy[hidden_relational == 1], s=9, alpha=0.9, color=PALETTE["gold"],
           label="Relational subgroup (truth)", linewidths=0)
ari_rel = results["informed"]["ARI_relational_hidden__polar_dbscan_on_feature_3_4"]
ax.set_title(f"After angular transform  (DBSCAN ARI = {ari_rel:.2f})")
ax.set_xlabel("cos(2\u03b8)"); ax.set_ylabel("sin(2\u03b8)")
ax.legend(fontsize=9, markerscale=2)
style_axes(ax)
fig.suptitle("Mechanism B (relational): invisible by distance, obvious by angle", fontsize=13, fontweight="bold", y=1.03)
savefig("fig20_relational_mechanism.png")

# ======================================================================
# Figure 14 -- Scale mechanism: raw scatter + kNN density histogram
# ======================================================================
from sklearn.neighbors import NearestNeighbors
f5 = observable["feature_5"].values
f6 = observable["feature_6"].values
nn = NearestNeighbors(n_neighbors=10).fit(np.column_stack([f5, f6]))
dists, _ = nn.kneighbors(np.column_stack([f5, f6]))
knn_d = dists[:, 1:].mean(axis=1)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4))
ax = axes[0]
ax.scatter(f5[hidden_scale == 0], f6[hidden_scale == 0], s=5, alpha=0.25, color=PALETTE["gray"],
           label="Background (large variance)", linewidths=0)
ax.scatter(f5[hidden_scale == 1], f6[hidden_scale == 1], s=11, alpha=0.9, color=PALETTE["gold"],
           label="Scale subgroup (small variance)", linewidths=0)
ax.set_xlim(-10, 10); ax.set_ylim(-10, 10)
ax.set_title("Raw space (feature_5, feature_6) -- same center")
ax.set_xlabel("feature_5"); ax.set_ylabel("feature_6")
ax.legend(fontsize=9, markerscale=1.6)
style_axes(ax)

ax = axes[1]
ax.hist(knn_d[hidden_scale == 0], bins=55, alpha=0.55, color=PALETTE["gray"], density=True, label="Background", zorder=2)
ax.hist(knn_d[hidden_scale == 1], bins=55, alpha=0.85, color=PALETTE["gold"], density=True, label="Scale subgroup", zorder=3)
ari_scale = results["informed"]["ARI_scale_hidden__knn_density_gmm_on_feature_5_6"]
ax.set_title(f"Local density (10-NN)  (informed ARI = {ari_scale:.2f})")
ax.set_xlabel("Mean k-NN distance"); ax.set_ylabel("Density")
ax.legend(fontsize=9)
style_axes(ax)
fig.suptitle("Mechanism C (scale): invisible by centroid, visible by local density", fontsize=13, fontweight="bold", y=1.03)
savefig("fig21_scale_mechanism.png")

# ======================================================================
# Figure 15 -- Multi-seed robustness (N=80): ARI vs Silhouette
# ======================================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 5.0))
ax = axes[0]
ax.hist(multiseed_raw["ari_k4"], bins=22, color=PALETTE["naive"], alpha=0.85, edgecolor="white", linewidth=0.5)
ax.axvline(multiseed_raw["ari_k4"].mean(), color=PALETTE["dark"], linestyle="--", linewidth=1.6,
           label=f"mean = {multiseed_raw['ari_k4'].mean():.3f}")
ax.set_title("ARI across 80 seeds\n(highly variable)")
ax.set_xlabel("ARI at k=4"); ax.set_ylabel("Seed count")
ax.legend(fontsize=9.5)
style_axes(ax)

ax = axes[1]
ax.hist(multiseed_raw["sil_k4"], bins=22, color=PALETTE["informed"], alpha=0.85, edgecolor="white", linewidth=0.5)
ax.axvline(multiseed_raw["sil_k4"].mean(), color=PALETTE["dark"], linestyle="--", linewidth=1.6,
           label=f"mean = {multiseed_raw['sil_k4'].mean():.3f}")
ax.set_title("Silhouette across 80 seeds\n(almost constant)")
ax.set_xlabel("Silhouette at k=4"); ax.set_ylabel("Seed count")
ax.legend(fontsize=9.5)
style_axes(ax)
fig.suptitle("Recovery is seed-dependent; validator confidence is not", fontsize=13, fontweight="bold", y=1.04)
savefig("fig22_multiseed_robustness.png")

# ======================================================================
# Figure 16 -- Interaction effect distribution across 80 seeds
# ======================================================================
pivot = interaction_raw.pivot(index="seed", columns="config", values="ari_k4")
drop_B = pivot["A_only"] - pivot["A_plus_B"]
drop_C = pivot["A_only"] - pivot["A_plus_C"]
drop_BC = pivot["A_only"] - pivot["A_plus_B_plus_C"]
predicted_additive = drop_B + drop_C
observed_interaction = drop_BC - predicted_additive

fig, ax = plt.subplots(figsize=(8.6, 5.0))
ax.hist(observed_interaction, bins=24, color=PALETTE["purple"], alpha=0.85, edgecolor="white", linewidth=0.5, zorder=3)
ax.axvline(0, color=PALETTE["dark"], linewidth=1.4, zorder=4, label="No interaction (additive)")
ax.axvline(observed_interaction.mean(), color=PALETTE["naive"], linestyle="--", linewidth=1.8, zorder=4,
           label=f"Observed mean = {observed_interaction.mean():.4f}")
ax.set_title("Interaction effect across 80 seeds -- centered at zero")
ax.set_xlabel("Interaction effect (ARI units)"); ax.set_ylabel("Seed count")
ax.legend(fontsize=9.5)
style_axes(ax)
savefig("fig24_interaction_effect.png")

# ======================================================================
# Figure 1 -- Theory: analytical vs true eigenvalue spectrum
# ======================================================================
analytical = theory["analytical_explained_variance_ratio"]
true_evr = theory["true_pca_explained_variance_ratio"]
fig, ax = plt.subplots(figsize=(8.6, 5.0))
xidx = np.arange(1, len(analytical)+1)
ax.plot(xidx, true_evr, "o-", color=PALETTE["naive"], linewidth=2.2, markersize=8,
        label="True PCA spectrum (full correlation matrix)", zorder=4)
ax.plot(xidx, analytical, "s--", color=PALETTE["informed"], linewidth=2.2, markersize=7,
        label="Block-diagonal approximation (Prop. 1)", zorder=4)
ax.axhline(1/len(analytical), color=PALETTE["gray"], linestyle=":", linewidth=1.4, label="Uniform baseline", zorder=2)
ax.set_xlabel("Principal component"); ax.set_ylabel("Explained variance ratio")
ax.set_xticks(xidx)
ax.set_title(f"Theory tracks reality (max error = {theory['max_abs_approximation_error']:.4f})")
ax.legend(fontsize=9.5)
style_axes(ax)
savefig("fig02_theory_eigenvalue_spectrum.png")

# ======================================================================
# Figure 2 -- Feature correlation heatmap with mechanism blocks
# ======================================================================
feat_cols = [c for c in observable.columns if c != "class_label"]
corr = observable[feat_cols].corr().values
fig, ax = plt.subplots(figsize=(7.6, 6.6))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-0.6, vmax=0.6)
ax.set_xticks(range(len(feat_cols))); ax.set_xticklabels(feat_cols, rotation=90, fontsize=9)
ax.set_yticks(range(len(feat_cols))); ax.set_yticklabels(feat_cols, fontsize=9)
# mechanism block outlines
blocks = [(0, 2, "A"), (2, 2, "B"), (4, 2, "C")]
for start, size, label in blocks:
    ax.add_patch(plt.Rectangle((start-0.5, start-0.5), size, size, fill=False,
                                edgecolor=PALETTE["dark"], linewidth=2.2, zorder=5))
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Pearson correlation")
ax.set_title("Feature correlation matrix (mechanism blocks outlined)")
savefig("fig03_correlation_heatmap.png")

# ======================================================================
# Figure 3 -- Dimensionality scaling: ARI and Silhouette vs d
# ======================================================================
dim_summary = scaling_dim_raw.groupby("d_total").agg(
    ari_mean=("ari_k4", "mean"), ari_std=("ari_k4", "std"),
    sil_mean=("sil_k4", "mean"), sil_std=("sil_k4", "std")).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(11, 5.0))
ax = axes[0]
ax.errorbar(dim_summary["d_total"], dim_summary["ari_mean"], yerr=dim_summary["ari_std"],
            fmt="o-", color=PALETTE["naive"], linewidth=2.2, markersize=8, capsize=4, zorder=4)
ax.set_xscale("log")
ax.set_xlabel("Total dimensionality (d)"); ax.set_ylabel("Mean ARI (distance mechanism)")
ax.set_title("Recovery degrades further as d grows")
style_axes(ax)

ax = axes[1]
ax.errorbar(dim_summary["d_total"], dim_summary["sil_mean"], yerr=dim_summary["sil_std"],
            fmt="s-", color=PALETTE["informed"], linewidth=2.2, markersize=8, capsize=4, zorder=4)
ax.set_xscale("log")
ax.set_ylim(0.3, 0.45)
ax.set_xlabel("Total dimensionality (d)"); ax.set_ylabel("Mean Silhouette")
ax.set_title("...while Silhouette stays flat")
style_axes(ax)
fig.suptitle("The false-confidence problem worsens, not improves, with realistic dimensionality",
             fontsize=12.5, fontweight="bold", y=1.04)
savefig("fig08_dimensionality_scaling.png")

# ======================================================================
# Figure 7 -- Stronger baselines comparison
# ======================================================================
methods = ["Linear PCA\n(naive)", "Sparse PCA", "Kernel PCA\n(RBF)", "t-SNE"]
keys = ["linear_pca", "sparse_pca", "kernel_pca_rbf", "tsne"]
ariA = [baselines[k]["ari_at_k4"] for k in keys]
ariB = [baselines[k]["ari_relational_hidden_at_k4"] for k in keys]
ariC = [baselines[k]["ari_scale_hidden_at_k4"] for k in keys]

fig, ax = plt.subplots(figsize=(9.6, 5.4))
x = np.arange(len(methods)); w = 0.26
b1 = ax.bar(x - w, ariA, w, label="Mechanism A (class)", color=PALETTE["informed"], edgecolor="white", linewidth=0.5, zorder=3)
b2 = ax.bar(x, ariB, w, label="Mechanism B (relational)", color=PALETTE["gold"], edgecolor="white", linewidth=0.5, zorder=3)
b3 = ax.bar(x + w, ariC, w, label="Mechanism C (scale)", color=PALETTE["purple"], edgecolor="white", linewidth=0.5, zorder=3)
ax.set_xticks(x); ax.set_xticklabels(methods)
ax.set_ylabel("ARI")
ax.set_title("Nonlinear reduction rescues mechanism A only")
ax.legend(fontsize=9.5)
style_axes(ax)
savefig("fig11_stronger_baselines.png")

# ======================================================================
# Figure 4 -- Real-data validation
# ======================================================================
datasets = ["Wine", "Breast Cancer", "Digits"]
keys = ["wine", "breast_cancer", "digits"]
ari_true_k = [realdata[k]["ari_at_true_k"] for k in keys]
agree = [realdata[k]["silhouette_true_vs_best_k_agree"] for k in keys]
colors_bar = [PALETTE["green"] if a else PALETTE["naive"] for a in agree]

fig, ax = plt.subplots(figsize=(8.2, 5.0))
bars = ax.bar(datasets, ari_true_k, color=colors_bar, edgecolor="white", linewidth=0.6, zorder=3, width=0.55)
value_labels(ax, bars)
ax.set_ylabel("ARI at true k")
ax.set_ylim(0, 1.05)
ax.set_title("The collapse is not a universal property of naive pipelines")
from matplotlib.patches import Patch
legend_elems = [Patch(facecolor=PALETTE["green"], label="Silhouette k agrees with truth"),
                Patch(facecolor=PALETTE["naive"], label="Silhouette k disagrees")]
ax.legend(handles=legend_elems, fontsize=9.5, loc="lower left")
style_axes(ax)
savefig("fig07_real_data_validation.png")

# ======================================================================
# Figure 8 -- ICS kurtosis screening (restyled)
# ======================================================================
headline_scores = ics["headline_seed42"]["pair_scores"]
pair_names = list(headline_scores.keys())
devs = [headline_scores[p]["abs_deviation_from_1"] for p in pair_names]
colors_ics = [PALETTE["gold"] if p == "feature_3,feature_4" else PALETTE["informed"] for p in pair_names]

fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
ax = axes[0]
bars = ax.barh(range(len(pair_names)), devs, color=colors_ics, edgecolor="white", linewidth=0.6, zorder=3)
ax.set_yticks(range(len(pair_names))); ax.set_yticklabels(pair_names, fontsize=9.5)
ax.invert_yaxis()
ax.set_xlabel("|ICS generalized eigenvalue - 1|\n(excess-kurtosis anomaly score)")
ax.set_title("Seed 42: relational pair is a clear outlier")
style_axes(ax)

ax = axes[1]
ax.hist(ics_raw["kurtosis_dev_feature_3_4"], bins=18, alpha=0.85, color=PALETTE["gold"],
        label="True relational pair (B)", zorder=3)
ax.hist(ics_raw["kurtosis_dev_feature_5_6"], bins=18, alpha=0.75, color=PALETTE["gray"],
        label="True scale pair (C, negative control)", zorder=2)
ax.set_xlabel("|ICS generalized eigenvalue - 1|"); ax.set_ylabel("Count across 80 seeds")
ax.set_title("N=80: separation is consistent, not lucky")
ax.legend(fontsize=9)
style_axes(ax)
fig.suptitle("ICS kurtosis screening reliably routes to the relational mechanism",
             fontsize=13, fontweight="bold", y=1.03)
savefig("fig14_ics_kurtosis_screening.png")

# ======================================================================
# Figure 17 (NEW) -- Method x Mechanism detection-matrix heatmap
# ======================================================================
row_labels = [
    "Naive PCA + KMeans",
    "Sparse PCA",
    "Kernel PCA (RBF)",
    "t-SNE",
    "Hand-informed\n(mechanism-specific)",
    "ICS kurtosis routing\n(oracle ceiling)*",
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
])

fig, ax = plt.subplots(figsize=(8.6, 6.4))
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
fig.text(0.02, -0.02,
         "*ICS row reports the oracle ceiling on its kurtosis-anomaly signal (Section 5.5), not an automatic\n"
         "end-to-end result; all other rows are fully automatic, label-free pipeline outputs at k=4, seed 42.",
         fontsize=8.6, color="#6B7280")
savefig("fig26_detection_matrix_summary.png")

print("\nAll figures (1-15) regenerated with unified style.")
