"""
baselines_comparison.py
========================
Priority 3 (Action Plan / Methodology / Code Guide docs): the reviewer's
point is that showing "naive K-Means fails to recover a non-convex /
linear / density-defined subgroup" is not, by itself, informative --
K-Means' bias toward convex, similarly-sized clusters is well known.
What is informative is whether OTHER off-the-shelf methods specifically
designed for non-spherical structure -- Spectral Clustering and DBSCAN --
ALSO fail when applied the same way a non-expert would apply them: as a
drop-in replacement for the clustering step in the naive
StandardScaler -> PCA(top-2) -> cluster front end, with NO oracle
knowledge of which 2 raw features carry which mechanism (that oracle
feature selection is what the paper's own "hand-informed pipeline"
already uses, and is reported separately in results.json / the paper's
Section 5.2 table -- it is not what this script tests).

Method
------
For the synthetic benchmark (all N=8000 points, no subsampling needed --
both methods run in ~1-2s at this size) and three real datasets bundled
with scikit-learn (Wine, Breast Cancer, Digits -- no network calls), we:

  1. Build the naive representation: StandardScaler -> PCA, keep the
     top-2 components (identical convention to analyze_pipelines.py /
     real_data_validation.py elsewhere in this package).
  2. Run KMeans (the paper's naive-pipeline reference), Spectral
     Clustering (nearest-neighbors affinity), and DBSCAN on that same
     2D representation.
  3. Grid-search each method's hyperparameters (n_clusters x n_neighbors
     for Spectral; eps x min_samples for DBSCAN, with eps candidates
     drawn from quantiles of the empirical k-distance distribution
     rather than an arbitrary fixed range) and report the BEST ARI/NMI
     found per method per target.

HONESTY NOTE: grid-searching hyperparameters against ground-truth ARI is
an oracle/best-case comparison for the baselines (they get to "see" the
labels to pick their own settings), exactly as this package's existing
oracle-ceiling numbers already do (e.g. results.json's
"ARI_scale_hidden__theoretical_ceiling_same_signal"). This is the
correct way to ask "can this method see the structure AT ALL under its
best setting", which is what the reviewer's Priority 3 is asking; it is
NOT a claim that Spectral/DBSCAN would reach these numbers automatically
in a fully label-free deployment. That distinction is kept explicit in
the printed output and the saved JSON.

Runtime: under a minute total, CPU only, only numpy/pandas/scikit-learn
(all already in requirements.txt).
"""

import json
import numpy as np
import pandas as pd
from itertools import product
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.datasets import load_wine, load_breast_cancer, load_digits
import os
BASE = os.path.dirname(os.path.abspath(__file__))

SEED = 0


# ---------------------------------------------------------------------
# Representation
# ---------------------------------------------------------------------
def naive_pca_top2(X):
    Xs = StandardScaler().fit_transform(X)
    n_comp = min(Xs.shape[1], Xs.shape[0] - 1, 2)
    pcs = PCA(n_components=max(n_comp, 2) if Xs.shape[1] >= 2 else n_comp,
              random_state=SEED).fit_transform(Xs)
    return pcs[:, :2]


# ---------------------------------------------------------------------
# Clustering wrappers
# ---------------------------------------------------------------------
def run_kmeans(X, params):
    return KMeans(n_clusters=params["n_clusters"], n_init=10, random_state=SEED).fit_predict(X)


def run_spectral(X, params):
    return SpectralClustering(n_clusters=params["n_clusters"], affinity="nearest_neighbors",
                               n_neighbors=params["n_neighbors"], random_state=SEED,
                               assign_labels="kmeans").fit_predict(X)


def run_dbscan(X, params):
    return DBSCAN(eps=params["eps"], min_samples=params["min_samples"]).fit_predict(X)


def dbscan_eps_candidates(X, min_samples, n_candidates=8):
    """Quantiles of the distance to each point's min_samples-th nearest
    neighbor -- the standard k-distance-plot heuristic for choosing a
    DBSCAN eps range, rather than an arbitrary fixed grid."""
    nn = NearestNeighbors(n_neighbors=min_samples).fit(X)
    dists, _ = nn.kneighbors(X)
    kth = np.sort(dists[:, -1])
    qs = np.linspace(0.10, 0.95, n_candidates)
    return sorted(set(np.round(np.quantile(kth, qs), 4).tolist()))


def grid_search_best(build_fn, param_grid, X, y_true):
    best = None
    for combo in product(*param_grid.values()):
        params = dict(zip(param_grid.keys(), combo))
        try:
            labels = build_fn(X, params)
        except Exception:
            continue
        n_found = len(set(labels)) - (1 if -1 in labels else 0)
        if n_found < 2:
            continue
        ari = float(adjusted_rand_score(y_true, labels))
        nmi = float(normalized_mutual_info_score(y_true, labels))
        row = {"ARI": ari, "NMI": nmi, "params": params, "n_clusters_found": int(n_found)}
        if best is None or ari > best["ARI"]:
            best = row
    return best


def evaluate_target(X_rep, y_true, k_options, n_neighbors_options, min_samples_options):
    results = {}

    best_km = grid_search_best(run_kmeans, {"n_clusters": k_options}, X_rep, y_true)
    results["KMeans (naive reference)"] = best_km

    best_sp = grid_search_best(
        run_spectral, {"n_clusters": k_options, "n_neighbors": n_neighbors_options}, X_rep, y_true)
    results["Spectral Clustering (grid-searched)"] = best_sp

    eps_by_ms = {ms: dbscan_eps_candidates(X_rep, ms) for ms in min_samples_options}
    best_db = None
    for ms, eps_candidates in eps_by_ms.items():
        row = grid_search_best(run_dbscan, {"eps": eps_candidates, "min_samples": [ms]}, X_rep, y_true)
        if row is not None and (best_db is None or row["ARI"] > best_db["ARI"]):
            best_db = row
    results["DBSCAN (grid-searched)"] = best_db

    return results


# ---------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------
def synthetic_targets():
    observable = pd.read_csv(os.path.join(BASE, "..", "data", "observable_dataset_hard.csv"))
    hidden = pd.read_csv(os.path.join(BASE, "..", "data", "hidden_labels_hard_EVAL_ONLY.csv"))
    feature_cols = [c for c in observable.columns if c != "class_label"]
    X = observable[feature_cols].values
    return X, {
        "mechanism_A_distance (class_label, k*=4)": (observable["class_label"].values, [3, 4, 5]),
        "mechanism_B_relational (hidden subgroup, k*=2)": (hidden["hidden_relational_label"].values, [2, 3]),
        "mechanism_C_scale (hidden subgroup, k*=2)": (hidden["hidden_scale_label"].values, [2, 3]),
    }


def real_datasets():
    out = {}
    wine = load_wine()
    out["Wine"] = (wine.data, wine.target, [2, 3, 4])
    bc = load_breast_cancer()
    out["Breast Cancer"] = (bc.data, bc.target, [2, 3])
    digits = load_digits()
    out["Digits"] = (digits.data, digits.target, [8, 9, 10, 11, 12])
    return out


def main():
    all_rows = []
    interpretation_notes = []

    # --- Synthetic benchmark ---
    X_syn, targets = synthetic_targets()
    X_rep = naive_pca_top2(X_syn)
    for target_name, (y, k_options) in targets.items():
        res = evaluate_target(X_rep, y, k_options, n_neighbors_options=[10, 20],
                               min_samples_options=[5, 10, 20])
        for method_name, row in res.items():
            all_rows.append({
                "dataset": "Synthetic benchmark (N=8000)", "target": target_name,
                "method": method_name,
                "ARI": None if row is None else round(row["ARI"], 4),
                "NMI": None if row is None else round(row["NMI"], 4),
                "n_clusters_found": None if row is None else row["n_clusters_found"],
                "best_params": None if row is None else json.dumps(row["params"]),
            })
        print(f"\n[Synthetic] {target_name}")
        for method_name, row in res.items():
            if row is None:
                print(f"  {method_name:38s}  no valid clustering found in grid")
            else:
                print(f"  {method_name:38s}  ARI={row['ARI']:.4f}  NMI={row['NMI']:.4f}  params={row['params']}")

    # --- Real datasets ---
    for name, (X, y, k_options) in real_datasets().items():
        X_rep = naive_pca_top2(X)
        res = evaluate_target(X_rep, y, k_options, n_neighbors_options=[10, 20],
                               min_samples_options=[5, 10, 20])
        for method_name, row in res.items():
            all_rows.append({
                "dataset": name, "target": "true class label",
                "method": method_name,
                "ARI": None if row is None else round(row["ARI"], 4),
                "NMI": None if row is None else round(row["NMI"], 4),
                "n_clusters_found": None if row is None else row["n_clusters_found"],
                "best_params": None if row is None else json.dumps(row["params"]),
            })
        print(f"\n[{name}] true class label")
        for method_name, row in res.items():
            if row is None:
                print(f"  {method_name:38s}  no valid clustering found in grid")
            else:
                print(f"  {method_name:38s}  ARI={row['ARI']:.4f}  NMI={row['NMI']:.4f}  params={row['params']}")

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(BASE, "..", "data", "baselines_comparison_results.csv"), index=False)

    # --- Compare against the paper's own oracle-informed pipeline numbers ---
    try:
        informed = json.load(open(os.path.join(BASE, "..", "data", "results.json")))["informed"]
    except Exception:
        informed = None

    summary = {
        "config": {"seed": SEED, "representation": "StandardScaler -> PCA -> top-2 components",
                    "hyperparameter_selection": "grid search against ground-truth ARI (oracle/best-case for baselines, see HONESTY NOTE in script)"},
        "rows": all_rows,
        "paper_hand_informed_pipeline_ARI_for_reference": informed,
    }
    with open(os.path.join(BASE, "..", "data", "baselines_comparison_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # --- Figure: grouped bar chart, synthetic benchmark only (the paper's core claim) ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    syn_df = df[df["dataset"] == "Synthetic benchmark (N=8000)"].copy()
    targets_order = list(targets.keys())
    methods_order = ["KMeans (naive reference)", "Spectral Clustering (grid-searched)", "DBSCAN (grid-searched)"]
    x = np.arange(len(targets_order))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 4.8))
    colors = ["#7f8c8d", "#2b6cb0", "#c0392b"]
    for i, method in enumerate(methods_order):
        vals = []
        for t in targets_order:
            match = syn_df[(syn_df["target"] == t) & (syn_df["method"] == method)]
            vals.append(match["ARI"].values[0] if len(match) and match["ARI"].values[0] is not None else 0.0)
        ax.bar(x + (i - 1) * width, vals, width, label=method, color=colors[i])
    if informed is not None:
        ref_vals = [
            informed.get("ARI_class_label__distance_kmeans_on_feature_1_2"),
            informed.get("ARI_relational_hidden__polar_dbscan_on_feature_3_4"),
            informed.get("ARI_scale_hidden__knn_density_gmm_on_feature_5_6"),
        ]
        ax.scatter(x, ref_vals, marker="*", s=180, color="black", zorder=5,
                   label="Hand-informed pipeline (oracle feature selection)")
    ax.set_xticks(x)
    ax.set_xticklabels([t.split(" (")[0].replace("mechanism_", "") for t in targets_order], fontsize=9)
    ax.set_ylabel("ARI (best over grid search)")
    ax.set_title("Priority 3: do off-the-shelf non-spherical methods\nsee what naive K-Means misses, blindly?")
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(BASE, "..", "figures", "fig12_baselines_comparison.png"), dpi=200)

    print("\nSaved: data/baselines_comparison_results.csv, "
          "data/baselines_comparison_results.json, "
          "figures/fig12_baselines_comparison.png")


if __name__ == "__main__":
    main()
