"""
stronger_baselines.py
=======================
Tests whether more sophisticated (but still off-the-shelf, no
ground-truth-informed) dimensionality-reduction front ends rescue the
naive pipeline on the full 3-mechanism benchmark:

  - Linear PCA (baseline, reused from analyze_pipelines.py)
  - Sparse PCA (encourages sparse loadings; in principle could isolate a
    mechanism to a subset of components if it aligns with few features)
  - Kernel PCA (RBF kernel; can capture some nonlinear structure, e.g.
    the angular/ratio geometry of mechanism B)
  - t-SNE (nonlinear manifold embedding, widely used as a "PCA
    replacement" for clustering-adjacent visualization)

UMAP is NOT included here (not available in this iteration). We report
this omission explicitly
rather than silently skip it.

All methods reduce to 2 dimensions (matching the naive pipeline's
top-2-PC convention) and then follow with the identical KMeans(k=4) +
Silhouette selection over k in {2..6}, so the only thing that changes is
the reduction step -- an apples-to-apples comparison.
"""

import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, SparsePCA, KernelPCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
import os
BASE = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE, "..", "data")

observable = pd.read_csv(f"{DATA_DIR}/observable_dataset_hard.csv")
hidden = pd.read_csv(f"{DATA_DIR}/hidden_labels_hard_EVAL_ONLY.csv")
class_label = observable["class_label"].values
hidden_relational = hidden["hidden_relational_label"].values
hidden_scale = hidden["hidden_scale_label"].values
feature_cols = [c for c in observable.columns if c != "class_label"]
X = observable[feature_cols].values
Xs = StandardScaler().fit_transform(X)


def cluster_and_score(embedding, k_grid=(2, 3, 4, 5, 6)):
    sil_by_k, ari_by_k = {}, {}
    models = {}
    for k in k_grid:
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(embedding)
        sil_by_k[k] = float(silhouette_score(embedding, km.labels_))
        ari_by_k[k] = float(adjusted_rand_score(class_label, km.labels_))
        models[k] = km
    best_k = max(sil_by_k, key=sil_by_k.get)
    return {
        "silhouette_by_k": sil_by_k,
        "ari_by_k": ari_by_k,
        "best_k_by_silhouette": int(best_k),
        "ari_at_silhouette_best_k": ari_by_k[best_k],
        "ari_at_k4": ari_by_k[4],
        "ari_relational_hidden_at_k4": float(adjusted_rand_score(hidden_relational, models[4].labels_)),
        "ari_scale_hidden_at_k4": float(adjusted_rand_score(hidden_scale, models[4].labels_)),
    }


results = {}

# 1. Linear PCA (reference, matches analyze_pipelines.py)
pca_emb = PCA(n_components=2, random_state=0).fit_transform(Xs)
results["linear_pca"] = cluster_and_score(pca_emb)

# 2. Sparse PCA
spca = SparsePCA(n_components=2, alpha=1.0, random_state=0, max_iter=200)
spca_emb = spca.fit_transform(Xs)
results["sparse_pca"] = cluster_and_score(spca_emb)
results["sparse_pca"]["component_sparsity_nonzero_loadings"] = [
    int(np.sum(np.abs(spca.components_[i]) > 1e-6)) for i in range(2)
]

# 3. Kernel PCA (RBF)
kpca = KernelPCA(n_components=2, kernel="rbf", gamma=None, random_state=0)
kpca_emb = kpca.fit_transform(Xs)
results["kernel_pca_rbf"] = cluster_and_score(kpca_emb)

# 4. t-SNE
tsne_emb = TSNE(n_components=2, random_state=0, perplexity=30, init="pca").fit_transform(Xs)
results["tsne"] = cluster_and_score(tsne_emb)

results["_note_umap"] = (
    "umap-learn is not installed in this sandboxed, network-disabled "
    "environment and could not be added; readers wishing to check UMAP "
    "specifically should run pip install umap-learn and re-execute this "
    "script, which will pick it up automatically if the import is added."
)

with open(f"{DATA_DIR}/stronger_baselines_results.json", "w") as f:
    json.dump(results, f, indent=2, default=float)

for name, r in results.items():
    if name.startswith("_"):
        continue
    print(f"{name:15s} best_k={r['best_k_by_silhouette']}  ARI@best_k={r['ari_at_silhouette_best_k']:.3f}  "
          f"ARI@k4={r['ari_at_k4']:.3f}  ARI_rel@k4={r['ari_relational_hidden_at_k4']:.3f}  "
          f"ARI_scale@k4={r['ari_scale_hidden_at_k4']:.3f}")
