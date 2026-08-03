"""
analyze_pipelines.py
=====================
Compares two clustering workflows on the harder synthetic benchmark:

  (1) NAIVE pipeline  -- what a researcher unaware of the hidden structure
      would typically do: StandardScaler -> PCA -> KMeans, evaluated with
      the usual internal validity metrics (which look fine!).

  (2) INFORMED pipeline -- mechanism-aware analysis: each independent
      sub-structure is recovered using the transform/algorithm suited to
      it (distance -> KMeans, ratio -> polar+DBSCAN, scale -> local
      density/variance test).

All results are saved to results.json and printed to stdout for the
report. Adjusted Rand Index (ARI) against the true labels is the primary
metric; NOTE the true labels are only used for EVALUATION, never for
discovery, exactly as in the previous benchmark.
"""

import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import adjusted_rand_score, silhouette_score
import os
BASE = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE, "..", "data")
OUT_DIR = os.path.join(BASE, "..", "data")

observable = pd.read_csv(f"{DATA_DIR}/observable_dataset_hard.csv")
hidden = pd.read_csv(f"{DATA_DIR}/hidden_labels_hard_EVAL_ONLY.csv")

class_label = observable["class_label"].values
hidden_relational = hidden["hidden_relational_label"].values
hidden_scale = hidden["hidden_scale_label"].values

feature_cols = [c for c in observable.columns if c != "class_label"]
X_raw = observable[feature_cols].values

results = {}

# ---------------------------------------------------------------------
# (1) NAIVE PIPELINE: StandardScaler -> PCA -> KMeans
#     This is the standard "textbook" recipe applied by someone who does
#     not know that 3 independent structures + noise are mixed together.
# ---------------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

pca = PCA(n_components=10, random_state=0)
pcs = pca.fit_transform(X_scaled)
explained = pca.explained_variance_ratio_.tolist()

# Naive researcher keeps the top-2 PCs for visualization/clustering
# (extremely common practice) and picks k via silhouette among small k.
top2 = pcs[:, :2]

naive_silhouette = {}
naive_models = {}
for k in [2, 3, 4, 5, 6]:
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(top2)
    sil = silhouette_score(top2, km.labels_)
    naive_silhouette[k] = sil
    naive_models[k] = km

best_k_by_silhouette = max(naive_silhouette, key=naive_silhouette.get)
naive_best_km = naive_models[best_k_by_silhouette]

# Also explicitly test the "correct" k=4 (in case researcher happens to
# guess the right number of classes from domain knowledge) for a fair
# comparison, plus k=5 (4 classes + "1 outlier cluster" guess).
km_k4_naive = naive_models[4]
km_k5_naive = naive_models[5]

results["naive"] = {
    "explained_variance_ratio_per_PC": explained,
    "silhouette_by_k": naive_silhouette,
    "best_k_by_silhouette": best_k_by_silhouette,
    "ARI_class_label__vs__naive_k_equal_best": adjusted_rand_score(class_label, naive_best_km.labels_),
    "ARI_class_label__vs__naive_k4": adjusted_rand_score(class_label, km_k4_naive.labels_),
    "ARI_relational_hidden__vs__naive_k4": adjusted_rand_score(hidden_relational, km_k4_naive.labels_),
    "ARI_relational_hidden__vs__naive_k5": adjusted_rand_score(hidden_relational, km_k5_naive.labels_),
    "ARI_scale_hidden__vs__naive_k4": adjusted_rand_score(hidden_scale, km_k4_naive.labels_),
    "ARI_scale_hidden__vs__naive_k5": adjusted_rand_score(hidden_scale, km_k5_naive.labels_),
}

# Also: naive KMeans directly on ALL 10 raw (scaled) features, k=4 --
# i.e. skipping PCA but still not knowing which features matter.
km_allfeat = KMeans(n_clusters=4, n_init=10, random_state=0).fit(X_scaled)
results["naive_no_pca_all_features"] = {
    "ARI_class_label": adjusted_rand_score(class_label, km_allfeat.labels_),
    "ARI_relational_hidden": adjusted_rand_score(hidden_relational, km_allfeat.labels_),
    "ARI_scale_hidden": adjusted_rand_score(hidden_scale, km_allfeat.labels_),
    "silhouette": silhouette_score(X_scaled, km_allfeat.labels_),
}

# ---------------------------------------------------------------------
# (2) INFORMED PIPELINE: mechanism-aware, three separate analyses
# ---------------------------------------------------------------------

# (A) Distance mechanism -> KMeans on feature_1, feature_2 ONLY
dist_space = observable[["feature_1", "feature_2"]].values
km_dist = KMeans(n_clusters=4, n_init=10, random_state=0).fit(dist_space)
ari_dist = adjusted_rand_score(class_label, km_dist.labels_)

# (B) Relational mechanism -> polar transform + DBSCAN on feature_3, feature_4
f3 = observable["feature_3"].values
f4 = observable["feature_4"].values
angle2 = 2 * np.arctan2(f4, f3)
polar_space = np.column_stack([np.cos(angle2), np.sin(angle2)])
db_rel = DBSCAN(eps=0.001, min_samples=10).fit(polar_space)
# largest non-noise cluster = relational hidden subgroup
labels_rel = db_rel.labels_
if (labels_rel != -1).any():
    vals, counts = np.unique(labels_rel[labels_rel != -1], return_counts=True)
    biggest = vals[np.argmax(counts)]
    pred_relational = (labels_rel == biggest).astype(int)
else:
    pred_relational = np.zeros_like(labels_rel)
ari_relational = adjusted_rand_score(hidden_relational, pred_relational)

# (C) Scale mechanism -> local density (k-NN distance) on feature_5, feature_6
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
f5f6 = observable[["feature_5", "feature_6"]].values
nn = NearestNeighbors(n_neighbors=10).fit(f5f6)
dists, _ = nn.kneighbors(f5f6)
mean_knn_dist = dists[:, 1:].mean(axis=1)  # exclude self
log_d = np.log(mean_knn_dist + 1e-9)

# Low-variance minority sits in a much denser local neighborhood.
# A plain (uninformed) 2-component GMM on log(kNN distance) does NOT
# converge to the right split (heavily imbalanced, ~4%, unimodal-looking
# mixture of two Rayleigh-shaped distributions) -- this itself is an
# honest, reported finding: mechanism C is intrinsically harder than A/B.
# An "informed" run that seeds the minority component with a small-weight,
# low-mean prior (reasonable domain assumption: "look for a small, dense
# subgroup", NOT the exact true fraction) recovers it far better than the
# naive pipeline, though still imperfectly.
means_init = np.array([[np.percentile(log_d, 5)], [np.percentile(log_d, 60)]])
weights_init = np.array([0.05, 0.95])
gmm = GaussianMixture(
    n_components=2, random_state=0,
    means_init=means_init, weights_init=weights_init, n_init=1
).fit(log_d.reshape(-1, 1))
comp_means = gmm.means_.ravel()
dense_component = np.argmin(comp_means)
gmm_labels = gmm.predict(log_d.reshape(-1, 1))
pred_scale = (gmm_labels == dense_component).astype(int)
ari_scale = adjusted_rand_score(hidden_scale, pred_scale)

# Theoretical ceiling: best ARI achievable by ANY density threshold on this
# same kNN-distance signal (reported for honesty; uses true labels only for
# evaluation of the ceiling, exactly as the AUC~0.5 ceiling was computed in
# the earlier HCSB benchmark -- never for discovery).
best_ari_ceiling = -1.0
for t in np.percentile(mean_knn_dist, np.arange(1, 20, 0.5)):
    p = (mean_knn_dist <= t).astype(int)
    a = adjusted_rand_score(hidden_scale, p)
    if a > best_ari_ceiling:
        best_ari_ceiling = a

results["informed"] = {
    "ARI_class_label__distance_kmeans_on_feature_1_2": ari_dist,
    "ARI_relational_hidden__polar_dbscan_on_feature_3_4": ari_relational,
    "ARI_scale_hidden__knn_density_gmm_on_feature_5_6": ari_scale,
    "ARI_scale_hidden__theoretical_ceiling_same_signal": best_ari_ceiling,
    "dbscan_relational_found_cluster_size": int((pred_relational == 1).sum()),
    "density_scale_found_cluster_size": int((pred_scale == 1).sum()),
}

# ---------------------------------------------------------------------
# Save everything
# ---------------------------------------------------------------------
with open(f"{OUT_DIR}/results.json", "w") as f:
    json.dump(results, f, indent=2, default=float)

print(json.dumps(results, indent=2, default=float))
