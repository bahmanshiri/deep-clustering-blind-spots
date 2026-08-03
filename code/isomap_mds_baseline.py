"""
Extends Section 5.3's stronger-baselines comparison with the two front ends
named in the newly-found related paper (arXiv:2604.22099, "Assessing the
impact of dimensionality reduction on clustering performance") that this
paper had not yet tested: Isomap and (classical) MDS.

Protocol matches Section 5.3 exactly: StandardScaler -> DR to 2 components ->
K-Means(k=4) -> ARI scored against all three ground-truth mechanisms
(class_label = A, hidden_relational_label = B, hidden_scale_label = C),
seed 42, same dataset file used throughout the paper.

MDS (classical/SMACOF, sklearn) is O(n^2) per iteration and impractical at
n=8000 within reasonable runtime; consistent with how the paper itself
subsampled Spectral Clustering earlier, MDS is run on a 2,000-point random
subsample (disclosed, not hidden) and scored only on that subsample's
labels. Isomap is run on the full n=8,000 like every other method in the
table.
"""
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import Isomap, MDS
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
import os
BASE = os.path.dirname(os.path.abspath(__file__))

np.random.seed(42)

X = pd.read_csv(os.path.join(BASE, "..", "data", "observable_dataset_hard.csv"))
y_A = X.pop('class_label').values
eval_labels = pd.read_csv(os.path.join(BASE, "..", "data", "hidden_labels_hard_EVAL_ONLY.csv"))
y_B = eval_labels['hidden_relational_label'].values
y_C = eval_labels['hidden_scale_label'].values
Xs = StandardScaler().fit_transform(X.values)

def run_pipeline(Z, y_A_, y_B_, y_C_, label):
    km = KMeans(n_clusters=4, n_init=10, random_state=42).fit(Z)
    labels = km.labels_
    ari_A = adjusted_rand_score(y_A_, labels)
    ari_B = adjusted_rand_score(y_B_, labels)
    ari_C = adjusted_rand_score(y_C_, labels)
    sil = silhouette_score(Z, labels, sample_size=2000, random_state=42)
    print(f"{label}: ARI-A={ari_A:.3f}  ARI-B={ari_B:.3f}  ARI-C={ari_C:.3f}  Silhouette={sil:.3f}")
    return ari_A, ari_B, ari_C, sil

print("="*70)
print("Isomap (n=8,000, full dataset, matching every other method in Table 1)")
print("="*70)
t0 = time.time()
iso = Isomap(n_neighbors=15, n_components=2)
Z_iso = iso.fit_transform(Xs)
print(f"  fit_transform took {time.time()-t0:.1f}s")
run_pipeline(Z_iso, y_A, y_B, y_C, "Isomap")

print("\n" + "="*70)
print("MDS (2,000-point random subsample -- classical MDS is O(n^2)/iter,")
print("impractical at n=8,000; disclosed deviation from the other rows)")
print("="*70)
sub_idx = np.random.RandomState(42).choice(len(Xs), size=2000, replace=False)
Xs_sub = Xs[sub_idx]
y_A_sub, y_B_sub, y_C_sub = y_A[sub_idx], y_B[sub_idx], y_C[sub_idx]
t0 = time.time()
mds = MDS(n_components=2, random_state=42, n_init=1, normalized_stress='auto')
Z_mds = mds.fit_transform(Xs_sub)
print(f"  fit_transform took {time.time()-t0:.1f}s (n=2000 subsample)")
run_pipeline(Z_mds, y_A_sub, y_B_sub, y_C_sub, "MDS (n=2000 subsample)")

print("\n" + "="*70)
print("For reference, Section 5.3's existing table (seed 42, full n=8,000):")
print("="*70)
print("Linear PCA (naive)  : ARI-A=0.115  ARI-B=0.237  ARI-C=0.001")
print("Sparse PCA          : ARI-A=0.130  ARI-B=0.239  ARI-C=0.001")
print("Kernel PCA (RBF)    : ARI-A=0.945  ARI-B=0.000  ARI-C=0.000")
print("t-SNE                : ARI-A=0.732  ARI-B=0.004  ARI-C=0.000")
