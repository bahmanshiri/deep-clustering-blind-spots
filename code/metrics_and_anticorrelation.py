"""
metrics_and_anticorrelation.py
================================
(1) Adds NMI and Purity alongside ARI for the seed=42 headline result, to
    check the finding is not an ARI-specific artifact.
(2) Directly tests whether Silhouette is anti-correlated with true
    recovery (ARI) -- across k within a seed, and across seeds at fixed
    k=4 -- rather than just asserting it qualitatively.
"""

import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score, silhouette_score,
    normalized_mutual_info_score,
)
from scipy.stats import spearmanr, pearsonr
import os
BASE = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE, "..", "data")


def purity_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    total = 0
    for c in np.unique(y_pred):
        mask = y_pred == c
        if mask.sum() == 0:
            continue
        vals, counts = np.unique(y_true[mask], return_counts=True)
        total += counts.max()
    return total / len(y_true)


observable = pd.read_csv(f"{DATA_DIR}/observable_dataset_hard.csv")
class_label = observable["class_label"].values
feature_cols = [c for c in observable.columns if c != "class_label"]
X = observable[feature_cols].values
Xs = StandardScaler().fit_transform(X)
pcs = PCA(n_components=10, random_state=0).fit_transform(Xs)
top2 = pcs[:, :2]

per_k = {}
for k in [2, 3, 4, 5, 6]:
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(top2)
    per_k[k] = {
        "silhouette": float(silhouette_score(top2, km.labels_)),
        "ARI": float(adjusted_rand_score(class_label, km.labels_)),
        "NMI": float(normalized_mutual_info_score(class_label, km.labels_)),
        "Purity": float(purity_score(class_label, km.labels_)),
    }

sil_vals = [per_k[k]["silhouette"] for k in per_k]
ari_vals = [per_k[k]["ARI"] for k in per_k]
# only 5 points (k=2..6) -- correlation across k within one seed is a weak
# test on its own; we combine it with the across-seed version below.
rho_within_seed, p_within_seed = spearmanr(sil_vals, ari_vals)

# --- across-seed correlation at fixed k=4, using the existing 20-seed
#     multi-seed robustness run (same seeds/results as the main paper) ---
msr = pd.read_csv(f"{DATA_DIR}/multiseed_robustness_raw.csv")
rho_across_seed, p_across_seed = spearmanr(msr["sil_k4"], msr["ari_k4"])
pearson_r_across_seed, pearson_p_across_seed = pearsonr(msr["sil_k4"], msr["ari_k4"])

result = {
    "seed42_metrics_by_k": per_k,
    "within_seed_k_sweep_spearman_sil_vs_ari": {
        "rho": float(rho_within_seed), "p": float(p_within_seed), "n_k_values": len(sil_vals),
        "caveat": "n=5 (k=2..6) is too small for a meaningful p-value on its own; "
                  "reported mainly for completeness alongside the across-seed test below.",
    },
    "across_seed_k4_fixed_spearman_sil_vs_ari": {
        "rho": float(rho_across_seed), "p": float(p_across_seed), "n_seeds": len(msr),
    },
    "across_seed_k4_fixed_pearson_sil_vs_ari": {
        "r": float(pearson_r_across_seed), "p": float(pearson_p_across_seed), "n_seeds": len(msr),
    },
    "interpretation": (
        "A near-zero or negative correlation between Silhouette and ARI "
        "across seeds at the fixed, Silhouette-preferred k=4 would support "
        "the claim that Silhouette provides no signal about (or actively "
        "misleads about) true recovery quality in this regime, beyond the "
        "single-seed anecdote."
    ),
}

with open(f"{DATA_DIR}/metrics_and_anticorrelation_results.json", "w") as f:
    json.dump(result, f, indent=2, default=float)

print(json.dumps(result, indent=2, default=float))
