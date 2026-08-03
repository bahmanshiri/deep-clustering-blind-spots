import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
BASE = os.path.dirname(os.path.abspath(__file__))

plt.rcParams["font.size"] = 12
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["font.family"] = "DejaVu Sans"

DATA_DIR = os.path.join(BASE, "..", "data")
FIG_DIR = os.path.join(BASE, "..", "figures")

COLORS4 = ["#e15759", "#4e79a7", "#59a14f", "#af7aa1"]
GOLD = "#f1c232"
GRAY = "#999999"

# ---------- Fig 9: theory -- analytical vs true PCA eigenvalue spectrum ----------
with open(f"{DATA_DIR}/theory_variance_budget_results.json") as f:
    T = json.load(f)

fig, ax = plt.subplots(figsize=(7, 4.5))
x = np.arange(1, 11)
ax.plot(x, T["true_pca_explained_variance_ratio"], "o-", color="#e15759",
        label="True PCA (full 10x10 correlation matrix)", linewidth=2)
ax.plot(x, T["analytical_explained_variance_ratio"], "s--", color="#4e79a7",
        label="Block-diagonal analytical approximation\n(Proposition 1)", linewidth=2)
ax.axhline(0.10, color=GRAY, linestyle=":", label="Uniform baseline (1/d = 0.10)")
ax.set_xlabel("Principal component")
ax.set_ylabel("Explained variance ratio")
ax.set_title(f"Theory vs. data: near-uniform spectrum\n(max approx. error = {T['max_abs_approximation_error']:.4f})")
ax.set_xticks(x)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig02_theory_eigenvalue_spectrum.png", dpi=170)
plt.close()

# ---------- Fig 10: mechanism cross-correlation heatmap (block-diagonality) ----------
observable = pd.read_csv(f"{DATA_DIR}/observable_dataset_hard.csv")
feature_cols = [c for c in observable.columns if c != "class_label"]
Xs = (observable[feature_cols].values - observable[feature_cols].values.mean(0)) / observable[feature_cols].values.std(0)
R = np.corrcoef(Xs, rowvar=False)

fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(R, cmap="RdBu_r", vmin=-0.6, vmax=0.6)
ax.set_xticks(range(10)); ax.set_yticks(range(10))
ax.set_xticklabels(feature_cols, rotation=90, fontsize=8)
ax.set_yticklabels(feature_cols, fontsize=8)
# outline the mechanism blocks
for (lo, hi, label) in [(-0.5, 1.5, "A"), (1.5, 3.5, "B"), (3.5, 5.5, "C"), (5.5, 9.5, "D (noise)")]:
    ax.add_patch(plt.Rectangle((lo, lo), hi - lo, hi - lo, fill=False, edgecolor="black", linewidth=1.8))
ax.set_title("Feature correlation matrix\n(near-zero off-block terms = approx. block-diagonal)")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig03_correlation_heatmap.png", dpi=170)
plt.close()

# ---------- Fig 11: dimensionality + noise-scale sweeps ----------
with open(f"{DATA_DIR}/scaling_experiments_results.json") as f:
    S = json.load(f)
dim_rows = S["dimensionality_sweep"]["by_d"]
d_vals = [r["d_total"] for r in dim_rows]
ari_m = [r["ari_mean"] for r in dim_rows]
ari_s = [r["ari_std"] for r in dim_rows]
sil_m = [r["sil_mean"] for r in dim_rows]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
axes[0].errorbar(d_vals, ari_m, yerr=ari_s, marker="o", color="#e15759", capsize=4, label="ARI (mean ± SD)")
axes[0].set_xlabel("Total feature dimensionality (d)")
axes[0].set_ylabel("ARI (true class recovery)")
axes[0].set_title("Recovery degrades further as d grows")
axes[0].set_xscale("log"); axes[0].set_xticks(d_vals); axes[0].set_xticklabels(d_vals)

axes[1].plot(d_vals, sil_m, marker="s", color=GOLD, linewidth=2, label="Silhouette (mean)")
axes[1].set_xlabel("Total feature dimensionality (d)")
axes[1].set_ylabel("Silhouette score")
axes[1].set_title("...while Silhouette stays flat & confident")
axes[1].set_xscale("log"); axes[1].set_xticks(d_vals); axes[1].set_xticklabels(d_vals)
axes[1].set_ylim(0, 0.6)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig08_dimensionality_scaling.png", dpi=170)
plt.close()

# ---------- Fig 12: stronger baselines comparison ----------
with open(f"{DATA_DIR}/stronger_baselines_results.json") as f:
    B = json.load(f)
methods = ["linear_pca", "sparse_pca", "kernel_pca_rbf", "tsne"]
labels = ["Linear PCA\n(naive)", "Sparse PCA", "Kernel PCA\n(RBF)", "t-SNE"]
ari_class = [B[m]["ari_at_k4"] for m in methods]
ari_rel = [B[m]["ari_relational_hidden_at_k4"] for m in methods]
ari_scale = [B[m]["ari_scale_hidden_at_k4"] for m in methods]

fig, ax = plt.subplots(figsize=(8.5, 4.8))
xpos = np.arange(len(methods))
w = 0.26
ax.bar(xpos - w, ari_class, width=w, label="Mechanism A (distance / class)", color="#e15759")
ax.bar(xpos, ari_rel, width=w, label="Mechanism B (relational, hidden)", color="#4e79a7")
ax.bar(xpos + w, ari_scale, width=w, label="Mechanism C (scale, hidden)", color="#59a14f")
ax.set_xticks(xpos); ax.set_xticklabels(labels)
ax.set_ylabel("ARI (at k=4, no ground-truth used)")
ax.set_title("Nonlinear reduction rescues mechanism A,\nbut B and C stay invisible to every generic front end")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig11_stronger_baselines.png", dpi=170)
plt.close()

# ---------- Fig 13: real-data validation summary ----------
with open(f"{DATA_DIR}/real_data_validation_results.json") as f:
    RD = json.load(f)
names = list(RD.keys())
disp = {"wine": "Wine\n(k=3)", "breast_cancer": "Breast Cancer\n(k=2)", "digits": "Digits\n(k=10)"}
ari_true = [RD[n]["ari_at_true_k"] for n in names]
ari_best = [RD[n]["ari_at_silhouette_best_k"] for n in names]
agree = [RD[n]["silhouette_true_vs_best_k_agree"] for n in names]

fig, ax = plt.subplots(figsize=(7.5, 4.3))
xpos = np.arange(len(names))
ax.bar(xpos - 0.17, ari_true, width=0.34, label="ARI @ true k", color="#4e79a7")
ax.bar(xpos + 0.17, ari_best, width=0.34, label="ARI @ Silhouette-preferred k", color=GOLD)
for i, a in enumerate(agree):
    ax.annotate("k agree" if a else "k DISAGREE", (xpos[i], max(ari_true[i], ari_best[i]) + 0.03),
                ha="center", fontsize=8, color=("#59a14f" if a else "#e15759"))
ax.set_xticks(xpos); ax.set_xticklabels([disp[n] for n in names])
ax.set_ylabel("ARI vs. true label")
ax.set_ylim(0, 1.05)
ax.set_title("Naive pipeline on unrelated real datasets:\nfailure mode is NOT universal (secondary check)")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig07_real_data_validation.png", dpi=170)
plt.close()

print("Saved figures 1, 2, 3, 4, 7 to", FIG_DIR)
import os
for f in sorted(os.listdir(FIG_DIR)):
    print(" ", f)
