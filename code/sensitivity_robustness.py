"""
sensitivity_robustness.py
==========================
Priority 6 (Action Plan / Methodology / Code Guide docs): the reviewer's
point about interaction_robustness_study.py / interaction_robustness_
multiconfig.py is that testing for a super-additive INTERACTION between
mechanisms that are constructed to be population-independent by design
(dataset_lib.py draws each from its own independent RNG stream) has a
foregone-conclusion flavor: of course independently-drawn mechanisms
don't interact. A more informative question is a STRESS test: deliberately
break that independence by a controlled amount and find the point at
which detection collapses -- a breakdown-point / sensitivity analysis,
not an interaction test.

Method
------
For each cross-mechanism correlation level c in CORR_LEVELS and each of
N_SEEDS seeds, we:

  1. Generate the standard benchmark (dataset_lib.generate), unmodified.
  2. Contaminate all 10 raw features with a SHARED per-sample latent
     factor g ~ N(0,1) via
         X_c[:, j] = sqrt(1 - c^2) * X_raw[:, j] + c * std_j * g
     which preserves each column's own marginal variance (std_j) exactly
     at every c (since g is independent of X_raw with unit variance) and
     induces an expected pairwise correlation of approximately c between
     EVERY pair of columns, not only cross-block pairs. This is a
     stronger stress test than pure cross-block-only contamination (it
     also corrupts within-block structure, e.g. mechanism B's
     radius/ratio relationship), which is a conservative choice: if
     detection survives this, it survives the narrower cross-block-only
     case Theorem 1 actually assumes. c=0 exactly reproduces the
     unmodified benchmark (verified below).
  3. Re-run the SAME hand-informed detectors used for the paper's
     headline numbers (identical code to analyze_pipelines.py's
     "informed" pipeline: KMeans on feature_1/2 for mechanism A,
     polar-transform + DBSCAN on feature_3/4 for mechanism B, kNN-density
     + informed-prior GMM on feature_5/6 for mechanism C) and record ARI
     against each mechanism's true/hidden label.
  4. Also record eps_hat and ||E_hat||_2 (same quantities as
     sample_complexity_simulation.py / theory_variance_budget.py) as a
     grounding check that c is actually moving the quantity Theorem 1's
     bound is stated in terms of.

We then report, per mechanism, the smallest tested c at which mean ARI
first drops below 50% of its c=0 value ("breakdown point"), without
presupposing where that point is.

HONESTY NOTE: reported exactly as measured, including if a mechanism's
detector is already too weak at c=0 to have a meaningful "50%-of-
baseline" breakdown point (this is flagged explicitly per mechanism
rather than silently omitted).

Runtime: ~1-2 min, CPU only, only numpy/pandas/scikit-learn/matplotlib.
"""

import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
import os
BASE = os.path.dirname(os.path.abspath(__file__))
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

import dataset_lib

SEED_BASE = 42
N_SEEDS = 5
CORR_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
N_SIGNAL_BLOCKS = 3
BLOCK_SIZE = 2


def contaminate(rng, X_raw, c):
    """Inject a shared latent factor at correlation level c; preserves
    each column's marginal variance exactly."""
    n, d = X_raw.shape
    if c == 0.0:
        return X_raw.copy()
    stds = X_raw.std(axis=0, ddof=0)
    g = rng.standard_normal(n)
    X_c = np.sqrt(1 - c ** 2) * X_raw + c * np.outer(g, stds)
    return X_c


def cross_block_mask(d, n_signal_blocks=N_SIGNAL_BLOCKS, block_size=BLOCK_SIZE):
    mask = np.ones((d, d), dtype=bool)
    np.fill_diagonal(mask, False)
    for b in range(n_signal_blocks):
        i0, i1 = b * block_size, (b + 1) * block_size
        mask[i0:i1, i0:i1] = False
    return mask


def theory_quantities(X_c):
    R = np.corrcoef(X_c, rowvar=False)
    d = X_c.shape[1]
    mask = cross_block_mask(d)
    eps_hat = float(np.max(np.abs(R[mask])))
    E = np.where(mask, R, 0.0)
    E2_hat = float(np.max(np.abs(np.linalg.eigvalsh(E))))
    return eps_hat, E2_hat


def informed_pipeline_aris(X_c, class_label, hidden_relational, hidden_scale):
    # (A) distance mechanism: feature_1, feature_2 (columns 0, 1)
    dist_space = X_c[:, 0:2]
    km_dist = KMeans(n_clusters=4, n_init=10, random_state=0).fit(dist_space)
    ari_a = adjusted_rand_score(class_label, km_dist.labels_)

    # (B) relational mechanism: feature_3, feature_4 (columns 2, 3), polar + DBSCAN
    f3, f4 = X_c[:, 2], X_c[:, 3]
    angle2 = 2 * np.arctan2(f4, f3)
    polar_space = np.column_stack([np.cos(angle2), np.sin(angle2)])
    db_rel = DBSCAN(eps=0.001, min_samples=10).fit(polar_space)
    labels_rel = db_rel.labels_
    if (labels_rel != -1).any():
        vals, counts = np.unique(labels_rel[labels_rel != -1], return_counts=True)
        biggest = vals[np.argmax(counts)]
        pred_relational = (labels_rel == biggest).astype(int)
    else:
        pred_relational = np.zeros_like(labels_rel)
    ari_b = adjusted_rand_score(hidden_relational, pred_relational)

    # (C) scale mechanism: feature_5, feature_6 (columns 4, 5), kNN-density + informed GMM
    f5f6 = X_c[:, 4:6]
    nn = NearestNeighbors(n_neighbors=10).fit(f5f6)
    dists, _ = nn.kneighbors(f5f6)
    mean_knn_dist = dists[:, 1:].mean(axis=1)
    log_d = np.log(mean_knn_dist + 1e-9)
    means_init = np.array([[np.percentile(log_d, 5)], [np.percentile(log_d, 60)]])
    weights_init = np.array([0.05, 0.95])
    try:
        gmm = GaussianMixture(n_components=2, random_state=0, means_init=means_init,
                               weights_init=weights_init, n_init=1).fit(log_d.reshape(-1, 1))
        comp_means = gmm.means_.ravel()
        dense_component = np.argmin(comp_means)
        gmm_labels = gmm.predict(log_d.reshape(-1, 1))
        pred_scale = (gmm_labels == dense_component).astype(int)
        ari_c = adjusted_rand_score(hidden_scale, pred_scale)
    except Exception:
        ari_c = float("nan")

    return ari_a, ari_b, ari_c


def main():
    rows = []
    for c in CORR_LEVELS:
        for s in range(N_SEEDS):
            seed = SEED_BASE + s
            observable, hidden = dataset_lib.generate(seed)
            feature_cols = [col for col in observable.columns if col != "class_label"]
            X_raw = observable[feature_cols].values
            class_label = observable["class_label"].values
            hidden_relational = hidden["hidden_relational_label"]
            hidden_scale = hidden["hidden_scale_label"]

            rng = np.random.default_rng(10_000 + seed * 97 + int(c * 1000))
            X_c = contaminate(rng, X_raw, c)

            eps_hat, E2_hat = theory_quantities(X_c)
            ari_a, ari_b, ari_c = informed_pipeline_aris(X_c, class_label, hidden_relational, hidden_scale)

            rows.append({"c": c, "seed": seed, "eps_hat": eps_hat, "E2_hat": E2_hat,
                         "ARI_A_distance": ari_a, "ARI_B_relational": ari_b, "ARI_C_scale": ari_c})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(BASE, "..", "data", "sensitivity_robustness_raw.csv"), index=False)

    agg = df.groupby("c").agg(
        eps_hat_mean=("eps_hat", "mean"),
        E2_hat_mean=("E2_hat", "mean"),
        ARI_A_mean=("ARI_A_distance", "mean"), ARI_A_std=("ARI_A_distance", "std"),
        ARI_B_mean=("ARI_B_relational", "mean"), ARI_B_std=("ARI_B_relational", "std"),
        ARI_C_mean=("ARI_C_scale", "mean"), ARI_C_std=("ARI_C_scale", "std"),
    ).reset_index()

    print(agg.to_string(index=False))

    # --- Breakdown points: first c where mean ARI < 50% of the c=0 value ---
    breakdown = {}
    for mech, col in [("A_distance", "ARI_A_mean"), ("B_relational", "ARI_B_mean"), ("C_scale", "ARI_C_mean")]:
        baseline = agg.loc[agg["c"] == 0.0, col].values[0]
        if baseline is None or np.isnan(baseline) or baseline <= 0.01:
            breakdown[mech] = f"not meaningful -- c=0 baseline ARI already near zero ({baseline:.4f})"
            continue
        threshold = 0.5 * baseline
        below = agg[agg[col] < threshold]
        if len(below) == 0:
            breakdown[mech] = f"not reached within tested range (c up to {max(CORR_LEVELS)}); baseline={baseline:.4f}"
        else:
            c_star = float(below["c"].min())
            breakdown[mech] = f"c = {c_star:.2f} (baseline ARI={baseline:.4f}, threshold={threshold:.4f})"

    print("\nBreakdown points (first c with mean ARI < 50% of c=0 baseline):")
    for mech, msg in breakdown.items():
        print(f"  mechanism {mech}: {msg}")

    results = {
        "config": {"seed_base": SEED_BASE, "n_seeds": N_SEEDS, "corr_levels": CORR_LEVELS,
                    "contamination_model": "shared per-sample latent factor g~N(0,1), "
                                            "X_c = sqrt(1-c^2)*X_raw + c*std_j*g (preserves marginal variance)"},
        "aggregated": agg.to_dict(orient="records"),
        "breakdown_points": breakdown,
    }
    with open(os.path.join(BASE, "..", "data", "sensitivity_robustness_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)

    # --- Figure ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for col, std_col, label, color in [
        ("ARI_A_mean", "ARI_A_std", "Mechanism A (distance)", "#2b6cb0"),
        ("ARI_B_mean", "ARI_B_std", "Mechanism B (relational)", "#27ae60"),
        ("ARI_C_mean", "ARI_C_std", "Mechanism C (scale)", "#c0392b"),
    ]:
        ax.errorbar(agg["c"], agg[col], yerr=agg[std_col], marker="o", capsize=3,
                    label=label, color=color)
    ax.set_xlabel("injected cross-mechanism correlation $c$")
    ax.set_ylabel("ARI (hand-informed detector, mean $\\pm$ std over %d seeds)" % N_SEEDS)
    ax.set_title("Priority 6: sensitivity / breakdown-point analysis\n(replaces the i.i.d.-mechanism TOST interaction test)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(BASE, "..", "figures", "fig25_sensitivity_breakdown.png"), dpi=200)

    print("\nSaved: data/sensitivity_robustness_raw.csv, "
          "data/sensitivity_robustness_results.json, "
          "figures/fig25_sensitivity_breakdown.png")


if __name__ == "__main__":
    main()
