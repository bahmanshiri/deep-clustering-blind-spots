"""
real_data_validation.py
=========================
Tests whether the qualitative phenomenon (naive PCA+KMeans+Silhouette
certifying a partition that is a poor match to a real, meaningful label)
shows up outside the synthetic benchmark.

SCOPE LIMITATION: we do not use downloaded UCI/Kaggle/banking datasets
here. We use three real,
well-known tabular datasets that ship LOCALLY with scikit-learn (no
network call): Wine, Breast Cancer Wisconsin (Diagnostic), and Digits.
These are not domain-matched to banking/clinical subgroup discovery, and
none of them is constructed to contain three independent hiding
mechanisms by design -- so this is a WEAKER, secondary check ("does the
qualitative failure mode appear at all on unrelated real data"), not a
domain-matched replication of the synthetic benchmark. We report
whatever the pipeline actually produces, including cases where the naive
pipeline does fine, since selectively reporting only confirming datasets
would misrepresent generalizability.
"""

import json
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine, load_breast_cancer, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score, normalized_mutual_info_score
import os
BASE = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE, "..", "data")


def naive_pipeline(X, y, k_grid, top_n_pcs=2):
    Xs = StandardScaler().fit_transform(X)
    n_comp = min(Xs.shape[1], Xs.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=0)
    pcs = pca.fit_transform(Xs)
    topN = pcs[:, :top_n_pcs]

    sil_by_k, ari_by_k, nmi_by_k = {}, {}, {}
    models = {}
    for k in k_grid:
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(topN)
        sil_by_k[k] = float(silhouette_score(topN, km.labels_))
        ari_by_k[k] = float(adjusted_rand_score(y, km.labels_))
        nmi_by_k[k] = float(normalized_mutual_info_score(y, km.labels_))
        models[k] = km

    best_k = max(sil_by_k, key=sil_by_k.get)
    return {
        "explained_variance_ratio_top5": pca.explained_variance_ratio_[:5].tolist(),
        "silhouette_by_k": sil_by_k,
        "ari_by_k": ari_by_k,
        "nmi_by_k": nmi_by_k,
        "best_k_by_silhouette": int(best_k),
        "true_n_classes": int(len(np.unique(y))),
        "ari_at_silhouette_best_k": ari_by_k[best_k],
        "ari_at_true_k": ari_by_k.get(len(np.unique(y)), None),
        "silhouette_true_vs_best_k_agree": bool(best_k == len(np.unique(y))),
    }


def run_dataset(name, X, y, k_grid):
    print(f"\n=== {name} (n={X.shape[0]}, d={X.shape[1]}, classes={len(np.unique(y))}) ===")
    r = naive_pipeline(X, y, k_grid)
    print(json.dumps(r, indent=2))
    return r


def main():
    results = {}

    wine = load_wine()
    results["wine"] = run_dataset("Wine", wine.data, wine.target, k_grid=[2, 3, 4, 5])

    bc = load_breast_cancer()
    results["breast_cancer"] = run_dataset("Breast Cancer Wisconsin", bc.data, bc.target, k_grid=[2, 3, 4])

    digits = load_digits()
    # digits is high-d (64) and 10 classes -- realistic "many noisy-ish pixel
    # features" case, closer in spirit (high-d, unclear which dims matter)
    results["digits"] = run_dataset("Digits", digits.data, digits.target, k_grid=[2, 5, 8, 10, 12])

    with open(f"{DATA_DIR}/real_data_validation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n\n=== SUMMARY ===")
    for name, r in results.items():
        print(f"{name:20s} true_k={r['true_n_classes']:2d}  "
              f"silhouette_best_k={r['best_k_by_silhouette']:2d}  "
              f"agree={r['silhouette_true_vs_best_k_agree']}  "
              f"ARI@best_k={r['ari_at_silhouette_best_k']:.3f}  "
              f"ARI@true_k={r['ari_at_true_k']:.3f}")


if __name__ == "__main__":
    main()
