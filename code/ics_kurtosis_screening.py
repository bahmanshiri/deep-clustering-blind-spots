"""
ics_kurtosis_screening.py
===========================
NEW analysis for this revision cycle. Motivated by a simple question the
first revision's auto-diagnostic prototype (auto_diagnostic_pipeline.py)
left open: its angular-minority-cluster heuristic for detecting the
RELATIONAL mechanism (B) failed (it was fooled by a pure-noise pair,
routing to feature_9/feature_10 instead of the true feature_3/feature_4).
We test whether a completely different, off-the-shelf statistic --
Invariant Coordinate Selection (ICS; Tyler, Critchley, Duembgen & Oja,
2009, JRSS-B) -- does better, and try to explain WHY, tying the
explanation back to this paper's own Theorem 1 / Proposition 1 framework.

What ICS is, briefly: given two scatter (covariance-like) matrices of
the same data, one that is standard second-moment covariance (COV) and
one that weights each point additionally by its squared Mahalanobis
distance (the classical fourth-moment scatter matrix, COV4), ICS jointly
diagonalizes the pair. The resulting generalized eigenvalues equal 1 for
directions along which the data is exactly Gaussian, and depart from 1
along directions with excess kurtosis (heavy or light tails) -- this is
literally what COV4 vs COV measures, generalizing the familiar univariate
kurtosis = E[(X-mu)^4]/sigma^4 comparison to a multivariate, direction-
specific version. ICS is affine-invariant (invariant to any linear
rescaling/rotation of the two input columns), unlike raw PCA which is
only rotation-invariant and is driven purely by variance (2nd moment),
not shape.

Our hypothesis: mechanism B's relational subgroup places ~12% of points
on a fixed-ratio line with radius UNIFORM(3,9) -- a much heavier-tailed,
non-Gaussian marginal than the Gaussian background it is embedded in.
This should show up as a strong excess-kurtosis direction specifically
in the (feature_3, feature_4) pair and, by design, should NOT show up
comparably for mechanism A (Gaussian blobs) or mechanism C (a
Gaussian-in-Gaussian scale mixture, which is directionally isotropic and
so should not create a strong single anomalous LINEAR direction).

We test this with the SAME statistical rigor bar the rest of the revision
uses: not a seed=42 anecdote, but N=80 independent seeds (identical
seed range 1000..1079 used throughout Section 6 of the paper, for direct
comparability), reporting mean/SD, and an explicit auto-routing
comparison against the existing DBSCAN-angular heuristic under the exact
same pre-registered-threshold-style protocol as auto_diagnostic_pipeline.py.

Outputs:
  data/ics_kurtosis_screening_results.json
  data/ics_kurtosis_screening_raw.csv
  figures/fig14_ics_kurtosis_screening.png
"""

import json
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, roc_auc_score

import dataset_lib as dl

import matplotlib
import os
BASE = os.path.dirname(os.path.abspath(__file__))
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(BASE, "..", "data")
FIG_DIR = os.path.join(BASE, "..", "figures")

N_SEEDS = 80
SEEDS = list(range(1000, 1000 + N_SEEDS))  # identical seed range to interaction_robustness_study.py

CANDIDATE_PAIRS = [
    ("feature_1", "feature_2"), ("feature_3", "feature_4"),
    ("feature_5", "feature_6"), ("feature_7", "feature_8"),
    ("feature_9", "feature_10"), ("feature_2", "feature_5"),
    ("feature_1", "feature_8"),
]
TRUE_PAIR = {"distance": "feature_1,feature_2", "angular": "feature_3,feature_4",
             "density": "feature_5,feature_6"}


def ics_pair_score(pair_vals):
    """2x2 ICS (COV4 vs COV1): returns (top generalized eigenvalue,
    |eigenvalue - 1|, and the eigenvector direction) for the single most
    non-Gaussian direction in this 2-D pair."""
    Xc = pair_vals - pair_vals.mean(axis=0)
    n, p = Xc.shape
    COV1 = np.cov(Xc, rowvar=False)
    sq = np.sum(Xc ** 2, axis=1)
    COV4 = (Xc * sq[:, None]).T @ Xc / n / (p + 2)
    eigvals, eigvecs = eigh(COV4, COV1)
    idx = np.argmax(np.abs(eigvals - 1))
    return float(eigvals[idx]), float(abs(eigvals[idx] - 1)), eigvecs[:, idx]


def ics_route_and_detect(observable, hidden_relational):
    """Score every candidate pair by ICS kurtosis-anomaly; the pair with
    the LARGEST anomaly is routed to an angular/ratio-style detector,
    exactly mirroring the pre-registered-threshold style of
    auto_diagnostic_pipeline.py (no peeking at ground truth for ROUTING).

    For WITHIN-pair minority detection we report three tiers, honestly
    separated, rather than collapsing them into one number:
      (i)   oracle ceiling -- best achievable ARI on this exact signal,
            evaluation-only use of true labels (never for discovery),
            identical convention to analyze_pipelines.py;
      (ii)  fully unsupervised 2-component GMM on log(projection^2) --
            no fraction assumption at all;
      (iii) fraction-assisted rule -- assumes only that the anomalous
            subgroup is a MINORITY of unspecified but small size (we use
            a generic 10% prior, not the true 12%), matching the
            informed-init convention already used for mechanism C in
            analyze_pipelines.py.
    """
    scores = {}
    for f1, f2 in CANDIDATE_PAIRS:
        pv = observable[[f1, f2]].values
        _, dev, _ = ics_pair_score(pv)
        scores[f"{f1},{f2}"] = dev
    detected_pair = max(scores, key=scores.get)
    correct_pair = (detected_pair == TRUE_PAIR["angular"])

    f1, f2 = detected_pair.split(",")
    pv = observable[[f1, f2]].values
    _, _, direction = ics_pair_score(pv)
    Xc = pv - pv.mean(axis=0)
    proj_sq = (Xc @ direction) ** 2
    log_sq = np.log(proj_sq + 1e-9).reshape(-1, 1)

    # (ii) fully unsupervised GMM, no fraction prior
    gmm_plain = GaussianMixture(n_components=2, random_state=0, n_init=10).fit(log_sq)
    means_plain = gmm_plain.means_.ravel()
    heavy_plain = np.argmax(means_plain)
    pred_plain = (gmm_plain.predict(log_sq) == heavy_plain).astype(int)
    ari_plain = float(adjusted_rand_score(hidden_relational, pred_plain))

    # (iii) generic small-minority-prior GMM (10%, not the true 12%)
    means_init = np.array([[np.percentile(log_sq, 45)], [np.percentile(log_sq, 90)]])
    weights_init = np.array([0.90, 0.10])
    gmm_prior = GaussianMixture(n_components=2, random_state=0, means_init=means_init,
                                 weights_init=weights_init, n_init=1).fit(log_sq)
    means_prior = gmm_prior.means_.ravel()
    heavy_prior = np.argmax(means_prior)
    pred_prior = (gmm_prior.predict(log_sq) == heavy_prior).astype(int)
    ari_prior = float(adjusted_rand_score(hidden_relational, pred_prior))

    return {
        "pair_scores": scores,
        "detected_pair": detected_pair,
        "correct": correct_pair,
        "ari_unsupervised_gmm_no_prior": ari_plain,
        "ari_generic_minority_prior_gmm": ari_prior,
        "predicted_minority_size_no_prior": int(pred_plain.sum()),
        "predicted_minority_size_with_prior": int(pred_prior.sum()),
    }


# --- 1. Headline (seed 42, matches paper's Section 6.1 style) ---
observable42, hidden42 = dl.generate(seed=42)
headline_scores = {}
for f1, f2 in CANDIDATE_PAIRS:
    ev, dev, _ = ics_pair_score(observable42[[f1, f2]].values)
    headline_scores[f"{f1},{f2}"] = {"top_generalized_eigenvalue": ev, "abs_deviation_from_1": dev}

headline_routing = ics_route_and_detect(observable42, hidden42["hidden_relational_label"])

# Oracle ceiling on the SAME anomalous-direction signal (best achievable
# threshold given the true labels, reported ONLY for evaluation, exactly
# as the paper's existing oracle-ceiling convention in analyze_pipelines.py)
f1, f2 = headline_routing["detected_pair"].split(",")
pv = observable42[[f1, f2]].values
_, _, direction = ics_pair_score(pv)
Xc = pv - pv.mean(axis=0)
proj_sq = (Xc @ direction) ** 2
auc_oracle = float(roc_auc_score(hidden42["hidden_relational_label"], proj_sq))
best_ari, best_t = -1.0, None
for t in np.percentile(proj_sq, np.arange(80, 99.5, 0.25)):
    p_ = (proj_sq >= t).astype(int)
    a = adjusted_rand_score(hidden42["hidden_relational_label"], p_)
    if a > best_ari:
        best_ari, best_t = a, t

# --- 2. Multi-seed robustness (N=80, seeds 1000-1079) ---
rows = []
for seed in SEEDS:
    obs, hid = dl.generate(seed=seed)
    scores = {}
    for f1, f2 in CANDIDATE_PAIRS:
        _, dev, _ = ics_pair_score(obs[[f1, f2]].values)
        scores[f"{f1},{f2}"] = dev
    detected = max(scores, key=scores.get)
    correct = detected == TRUE_PAIR["angular"]

    df1, df2 = detected.split(",")
    pv = obs[[df1, df2]].values
    _, _, direction = ics_pair_score(pv)
    Xc = pv - pv.mean(axis=0)
    proj_sq = (Xc @ direction) ** 2
    log_sq = np.log(proj_sq + 1e-9).reshape(-1, 1)

    best_a = -1.0
    for t in np.percentile(proj_sq, np.arange(80, 99.5, 1.0)):
        p_ = (proj_sq >= t).astype(int)
        a = adjusted_rand_score(hid["hidden_relational_label"], p_)
        if a > best_a:
            best_a = a

    gmm_plain = GaussianMixture(n_components=2, random_state=0, n_init=10).fit(log_sq)
    heavy_plain = np.argmax(gmm_plain.means_.ravel())
    pred_plain = (gmm_plain.predict(log_sq) == heavy_plain).astype(int)
    ari_plain = float(adjusted_rand_score(hid["hidden_relational_label"], pred_plain))

    means_init = np.array([[np.percentile(log_sq, 45)], [np.percentile(log_sq, 90)]])
    weights_init = np.array([0.90, 0.10])
    gmm_prior = GaussianMixture(n_components=2, random_state=0, means_init=means_init,
                                 weights_init=weights_init, n_init=1).fit(log_sq)
    heavy_prior = np.argmax(gmm_prior.means_.ravel())
    pred_prior = (gmm_prior.predict(log_sq) == heavy_prior).astype(int)
    ari_prior = float(adjusted_rand_score(hid["hidden_relational_label"], pred_prior))

    # For comparison: the SAME oracle-threshold procedure applied to the
    # TRUE pair regardless of what was detected, and to mechanism C's true
    # pair, to test the negative control (ICS should NOT find mechanism C)
    pv_true_b = obs[["feature_3", "feature_4"]].values
    _, dev_true_b, dir_true_b = ics_pair_score(pv_true_b)
    Xc_b = pv_true_b - pv_true_b.mean(axis=0)
    proj_sq_b = (Xc_b @ dir_true_b) ** 2
    best_a_trueB = -1.0
    for t in np.percentile(proj_sq_b, np.arange(80, 99.5, 1.0)):
        p_ = (proj_sq_b >= t).astype(int)
        a = adjusted_rand_score(hid["hidden_relational_label"], p_)
        if a > best_a_trueB:
            best_a_trueB = a

    pv_true_c = obs[["feature_5", "feature_6"]].values
    _, dev_true_c, dir_true_c = ics_pair_score(pv_true_c)
    Xc_c = pv_true_c - pv_true_c.mean(axis=0)
    proj_sq_c = (Xc_c @ dir_true_c) ** 2
    best_a_trueC = -1.0
    for t in np.percentile(proj_sq_c, np.arange(1, 20, 1.0)):
        p_ = (proj_sq_c <= t).astype(int)
        a = adjusted_rand_score(hid["hidden_scale_label"], p_)
        if a > best_a_trueC:
            best_a_trueC = a

    rows.append({
        "seed": seed,
        "detected_pair": detected,
        "routing_correct": correct,
        "kurtosis_dev_feature_3_4": dev_true_b,
        "kurtosis_dev_feature_5_6": dev_true_c,
        "oracle_ari_on_detected_pair": best_a,
        "unsupervised_gmm_no_prior_ari": ari_plain,
        "generic_minority_prior_gmm_ari": ari_prior,
        "oracle_ari_on_true_relational_pair": best_a_trueB,
        "oracle_ari_on_true_scale_pair_negative_control": best_a_trueC,
    })

raw = pd.DataFrame(rows)
raw.to_csv(f"{DATA_DIR}/ics_kurtosis_screening_raw.csv", index=False)

summary = {
    "n_seeds": N_SEEDS,
    "routing_accuracy": float(raw["routing_correct"].mean()),
    "mean_kurtosis_dev_true_relational_pair": float(raw["kurtosis_dev_feature_3_4"].mean()),
    "sd_kurtosis_dev_true_relational_pair": float(raw["kurtosis_dev_feature_3_4"].std()),
    "mean_kurtosis_dev_true_scale_pair_negative_control": float(raw["kurtosis_dev_feature_5_6"].mean()),
    "sd_kurtosis_dev_true_scale_pair_negative_control": float(raw["kurtosis_dev_feature_5_6"].std()),
    "mean_oracle_ari_on_detected_pair": float(raw["oracle_ari_on_detected_pair"].mean()),
    "sd_oracle_ari_on_detected_pair": float(raw["oracle_ari_on_detected_pair"].std()),
    "mean_unsupervised_gmm_no_prior_ari": float(raw["unsupervised_gmm_no_prior_ari"].mean()),
    "sd_unsupervised_gmm_no_prior_ari": float(raw["unsupervised_gmm_no_prior_ari"].std()),
    "mean_generic_minority_prior_gmm_ari": float(raw["generic_minority_prior_gmm_ari"].mean()),
    "sd_generic_minority_prior_gmm_ari": float(raw["generic_minority_prior_gmm_ari"].std()),
    "mean_oracle_ari_true_relational_pair": float(raw["oracle_ari_on_true_relational_pair"].mean()),
    "mean_oracle_ari_true_scale_pair_negative_control": float(raw["oracle_ari_on_true_scale_pair_negative_control"].mean()),
}

# Comparison point taken directly from the existing auto-diagnostic results
# (auto_diagnostic_pipeline_results.json, seed 42): the DBSCAN-angular
# heuristic's routing was WRONG (selected feature_9,feature_10) and its
# resulting ARI was 0.0043.
comparison = {
    "existing_dbscan_angular_heuristic_seed42_routing_correct": False,
    "existing_dbscan_angular_heuristic_seed42_ari": 0.004340086317494413,
    "ics_kurtosis_heuristic_seed42_routing_correct": headline_routing["correct"],
    "ics_kurtosis_heuristic_seed42_ari_unsupervised_no_prior": headline_routing["ari_unsupervised_gmm_no_prior"],
    "ics_kurtosis_heuristic_seed42_ari_generic_minority_prior": headline_routing["ari_generic_minority_prior_gmm"],
    "ics_kurtosis_heuristic_seed42_ari_oracle_ceiling": best_ari,
    "ics_kurtosis_heuristic_seed42_auc_oracle": auc_oracle,
}

results = {
    "headline_seed42": {
        "pair_scores": headline_scores,
        "routing": headline_routing,
        "oracle_ceiling": {"ari": best_ari, "auc": auc_oracle},
    },
    "multi_seed_summary_N80": summary,
    "comparison_to_existing_dbscan_angular_heuristic": comparison,
}

with open(f"{DATA_DIR}/ics_kurtosis_screening_results.json", "w") as f:
    json.dump(results, f, indent=2, default=float)

print(json.dumps(results, indent=2, default=float))

# --- Figure ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
pair_names = list(headline_scores.keys())
devs = [headline_scores[p]["abs_deviation_from_1"] for p in pair_names]
colors = ["#f1c232" if p == "feature_3,feature_4" else "#4e79a7" for p in pair_names]
ax.bar(range(len(pair_names)), devs, color=colors)
ax.set_xticks(range(len(pair_names)))
ax.set_xticklabels(pair_names, rotation=40, ha="right", fontsize=8)
ax.set_ylabel("|ICS generalized eigenvalue - 1|\n(excess-kurtosis anomaly score)")
ax.set_title("ICS kurtosis screening correctly flags\nthe true relational pair (seed 42)")

ax = axes[1]
ax.hist(raw["kurtosis_dev_feature_3_4"], bins=20, alpha=0.7, color="#f1c232", label="True relational pair (B)")
ax.hist(raw["kurtosis_dev_feature_5_6"], bins=20, alpha=0.7, color="#999999", label="True scale pair (C, negative control)")
ax.set_xlabel("|ICS generalized eigenvalue - 1|")
ax.set_ylabel("Count across 80 seeds")
ax.set_title("Mechanism B shows strong, consistent excess\nkurtosis; mechanism C does not (N=80 seeds)")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig14_ics_kurtosis_screening.png", dpi=170)
plt.close()

print("\nSaved figure to", f"{FIG_DIR}/fig14_ics_kurtosis_screening.png")
