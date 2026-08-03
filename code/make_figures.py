import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
BASE = os.path.dirname(os.path.abspath(__file__))

plt.rcParams["font.size"] = 12
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["font.family"] = "DejaVu Sans"

FIG_DIR = os.path.join(BASE, "..", "figures")
DATA_DIR = os.path.join(BASE, "..", "data")

observable = pd.read_csv(f"{DATA_DIR}/observable_dataset_hard.csv")
hidden = pd.read_csv(f"{DATA_DIR}/hidden_labels_hard_EVAL_ONLY.csv")
class_label = observable["class_label"].values
hidden_relational = hidden["hidden_relational_label"].values
hidden_scale = hidden["hidden_scale_label"].values
with open(f"{DATA_DIR}/results.json") as f:
    R = json.load(f)

COLORS4 = ["#e15759", "#4e79a7", "#59a14f", "#af7aa1"]
GOLD = "#f1c232"
GRAY = "#999999"

# Fig 1: ARI comparison
mechanisms = ["Distance structure\n(4 main groups)", "Relational subgroup", "Scale/density subgroup"]
naive_vals = [
    R["naive"]["ARI_class_label__vs__naive_k4"],
    R["naive"]["ARI_relational_hidden__vs__naive_k4"],
    R["naive"]["ARI_scale_hidden__vs__naive_k4"],
]
informed_vals = [
    R["informed"]["ARI_class_label__distance_kmeans_on_feature_1_2"],
    R["informed"]["ARI_relational_hidden__polar_dbscan_on_feature_3_4"],
    R["informed"]["ARI_scale_hidden__knn_density_gmm_on_feature_5_6"],
]
ceiling_scale = R["informed"]["ARI_scale_hidden__theoretical_ceiling_same_signal"]

fig, ax = plt.subplots(figsize=(9, 5.5))
x = np.arange(len(mechanisms))
w = 0.35
b1 = ax.bar(x - w/2, naive_vals, w, label="Naive pipeline (standard PCA+KMeans)", color="#e15759")
b2 = ax.bar(x + w/2, informed_vals, w, label="Informed pipeline (mechanism-aware)", color="#4e79a7")
ax.scatter([2 + w/2], [ceiling_scale], marker="*", s=260, color="#f1c232",
           zorder=5, label="Theoretical ceiling (scale mechanism)", edgecolor="black", linewidth=0.6)
ax.set_ylabel("ARI (vs. ground truth)")
ax.set_xticks(x)
ax.set_xticklabels(mechanisms)
ax.set_ylim(-0.05, 1.05)
ax.axhline(0, color="black", linewidth=0.8)
for bars in (b1, b2):
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}", (bar.get_x()+bar.get_width()/2, h),
                    textcoords="offset points", xytext=(0,4), ha="center", fontsize=10)
ax.legend(loc="upper right", fontsize=10)
ax.set_title("Naive vs. Informed pipeline: ARI across all 3 hiding mechanisms")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig16_ari_comparison.png", dpi=170)
plt.close()

# Fig 2: PCA explained variance
ev = R["naive"]["explained_variance_ratio_per_PC"]
fig, ax = plt.subplots(figsize=(8.5, 5))
ax.bar(range(1, len(ev)+1), ev, color="#4e79a7")
ax.axhline(1/len(ev), color="#e15759", linestyle="--", linewidth=1.5,
           label=f"Uniform baseline (1/{len(ev)} = {1/len(ev):.3f})")
ax.set_xlabel("Principal Component")
ax.set_ylabel("Explained variance ratio")
ax.set_xticks(range(1, len(ev)+1))
ax.set_title("PCA variance spread — a warning sign of multiple independent structures")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig17_pca_variance.png", dpi=170)
plt.close()

# Fig 3: false confidence (silhouette vs true ARI)
sil = R["naive"]["silhouette_by_k"]
ks = sorted(int(k) for k in sil.keys())
sil_vals = [sil[str(k)] for k in ks]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(observable.drop(columns=["class_label"]).values)
pca = PCA(n_components=10, random_state=0)
pcs = pca.fit_transform(X_scaled)
top2 = pcs[:, :2]
ari_class_by_k = []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(top2)
    ari_class_by_k.append(adjusted_rand_score(class_label, km.labels_))

fig, ax1 = plt.subplots(figsize=(8.5, 5))
ax2 = ax1.twinx()
l1, = ax1.plot(ks, sil_vals, "o-", color="#4e79a7", linewidth=2, markersize=8,
               label="Silhouette (internal metric — what the researcher sees)")
l2, = ax2.plot(ks, ari_class_by_k, "s--", color="#e15759", linewidth=2, markersize=8,
               label="True ARI vs. 4 main groups (what the researcher does NOT see)")
ax1.set_xlabel("Number of clusters (k)")
ax1.set_ylabel("Silhouette Score", color="#4e79a7")
ax2.set_ylabel("True ARI", color="#e15759")
ax1.set_xticks(ks)
ax1.set_title("False confidence: internal metric looks fine, ground truth is not")
lines = [l1, l2]
ax1.legend(lines, [l.get_label() for l in lines], loc="upper center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig18_false_confidence.png", dpi=170)
plt.close()

# Fig 4: distance mechanism ground truth vs naive result
km4_naive = KMeans(n_clusters=4, n_init=10, random_state=0).fit(top2)
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
ax = axes[0]
for c in range(4):
    m = class_label == c
    ax.scatter(observable["feature_1"][m], observable["feature_2"][m],
               s=6, alpha=0.5, color=COLORS4[c], label=f"Group {c}")
ax.set_title("Ground truth (feature_1 vs feature_2)")
ax.set_xlabel("feature_1"); ax.set_ylabel("feature_2")
ax.legend(fontsize=9, markerscale=2)

ax = axes[1]
naive_labels = km4_naive.labels_
palette = plt.cm.tab10(np.linspace(0, 1, 10))
for c in np.unique(naive_labels):
    m = naive_labels == c
    ax.scatter(observable["feature_1"][m], observable["feature_2"][m],
               s=6, alpha=0.5, color=palette[c], label=f"Naive cluster {c}")
ax.set_title(f"Naive pipeline result (ARI={R['naive']['ARI_class_label__vs__naive_k4']:.2f})")
ax.set_xlabel("feature_1"); ax.set_ylabel("feature_2")
ax.legend(fontsize=9, markerscale=2)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig19_distance_mechanism.png", dpi=170)
plt.close()

# Fig 5: relational mechanism raw vs polar
f3 = observable["feature_3"].values
f4 = observable["feature_4"].values
angle2 = 2*np.arctan2(f4, f3)
cx, cy = np.cos(angle2), np.sin(angle2)

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
ax = axes[0]
ax.scatter(f3[hidden_relational==0], f4[hidden_relational==0], s=5, alpha=0.35, color=GRAY, label="Normal samples")
ax.scatter(f3[hidden_relational==1], f4[hidden_relational==1], s=8, alpha=0.8, color=GOLD, label="Relational subgroup (truth)")
ax.set_title("Raw space (feature_3, feature_4)")
ax.set_xlabel("feature_3"); ax.set_ylabel("feature_4")
ax.legend(fontsize=9, markerscale=2)

ax = axes[1]
ax.scatter(cx[hidden_relational==0], cy[hidden_relational==0], s=5, alpha=0.35, color=GRAY, label="Normal samples")
ax.scatter(cx[hidden_relational==1], cy[hidden_relational==1], s=8, alpha=0.8, color=GOLD, label="Relational subgroup (truth)")
ax.set_title(f"After angular transform (DBSCAN ARI={R['informed']['ARI_relational_hidden__polar_dbscan_on_feature_3_4']:.2f})")
ax.set_xlabel("cos(2\u03b8)"); ax.set_ylabel("sin(2\u03b8)")
ax.legend(fontsize=9, markerscale=2)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig20_relational_mechanism.png", dpi=170)
plt.close()

# Fig 6: scale mechanism raw scatter + knn distance histogram
f5 = observable["feature_5"].values
f6 = observable["feature_6"].values
nn = NearestNeighbors(n_neighbors=10).fit(np.column_stack([f5,f6]))
dists,_ = nn.kneighbors(np.column_stack([f5,f6]))
knn_d = dists[:,1:].mean(axis=1)

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
ax = axes[0]
ax.scatter(f5[hidden_scale==0], f6[hidden_scale==0], s=5, alpha=0.3, color=GRAY, label="Background cloud (large variance)")
ax.scatter(f5[hidden_scale==1], f6[hidden_scale==1], s=10, alpha=0.9, color=GOLD, label="Scale subgroup (small variance)")
ax.set_xlim(-10,10); ax.set_ylim(-10,10)
ax.set_title("Raw space (feature_5, feature_6) — same center, different scale")
ax.set_xlabel("feature_5"); ax.set_ylabel("feature_6")
ax.legend(fontsize=9, markerscale=1.5)

ax = axes[1]
ax.hist(knn_d[hidden_scale==0], bins=60, alpha=0.55, color=GRAY, density=True, label="Background")
ax.hist(knn_d[hidden_scale==1], bins=60, alpha=0.75, color=GOLD, density=True, label="Scale subgroup")
ax.set_title(f"Mean distance to 10 nearest neighbors — informed ARI={R['informed']['ARI_scale_hidden__knn_density_gmm_on_feature_5_6']:.2f}")
ax.set_xlabel("Mean k-NN distance"); ax.set_ylabel("Density")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig21_scale_mechanism.png", dpi=170)
plt.close()

print("All figures saved to", FIG_DIR)
import os
for f in sorted(os.listdir(FIG_DIR)):
    print(" -", f)
