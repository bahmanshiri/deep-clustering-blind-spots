"""
ICS_Eigengap_HDBSCAN.py
========================
Reviewer-response Priority 5: replace the GMM final-clustering step used
in ics_kurtosis_screening.py's within-pair minority detection with (a) an
eigengap-based automatic thresholding rule for ROUTING confidence, and
(b) HDBSCAN for the final WITHIN-pair clustering, so that no step in the
pipeline assumes a minority-fraction prior or a fixed number of mixture
components.

Two GMM-based steps existed in ics_kurtosis_screening.py that this script
replaces:

  1. Routing confidence was implicit (argmax over 7 candidate-pair
     kurtosis-deviation scores, no notion of "how much better than the
     runner-up"). Here this is made explicit and automatic via an
     eigengap rule: sort the 7 deviation scores descending and take the
     largest gap between consecutive scores. If that gap occurs at
     position 1 (i.e. between the top score and everything else), the
     routing is called CONFIDENT; otherwise it is flagged AMBIGUOUS. This
     mirrors the standard eigengap heuristic used to pick the number of
     clusters in spectral clustering (look for the largest gap in a
     sorted spectrum), applied here to a spectrum of pair-level anomaly
     scores instead of Laplacian eigenvalues.

  2. Within-pair minority detection previously used two GMM variants,
     one of them seeded with a hand-picked "generic 10%" minority-weight
     prior (ics_kurtosis_screening.py, tier iii). That prior is a manual
     assumption. Here it is replaced entirely by HDBSCAN applied to the
     standardized 2-D (detected pair) feature space. HDBSCAN is used
     because it (a) does not take a target number of clusters, (b)
     explicitly models a "noise"/background class rather than forcing
     every point into a component, and (c) its main sensitivity
     parameter (min_cluster_size) is set here from a fixed, generic
     percentage of n (not from the known 12% true minority fraction),
     analogous in spirit to the old prior but without needing to assume
     the *count* of clusters and without using the ground-truth fraction.

Ground truth (hidden_relational_label / hidden_scale_label) is used only
for scoring (ARI, oracle-threshold ceiling), exactly as in every other
script in this package -- never for routing or for fitting HDBSCAN.

Designed to run standalone (no other files from this package required)
on Google Colab:

    !pip install hdbscan
    # upload/paste this file, then:
    !python ICS_Eigengap_HDBSCAN.py

Outputs (written next to this script if code/../data and code/../figures
do not exist, otherwise into the package's data/ and figures/ dirs):
  data/ics_eigengap_hdbscan_results.json
  data/ics_eigengap_hdbscan_raw.csv
  figures/fig27_ics_eigengap_hdbscan.png
"""

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.metrics import adjusted_rand_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    import hdbscan
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "hdbscan"])
    import hdbscan

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Paths: write into the package's data/figures dirs if this script is
# run from inside code/ (as in the rest of this package); otherwise
# fall back to ./data and ./figures next to the script, so the file
# also runs unmodified when pasted standalone into Colab.
# ---------------------------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
_pkg_data = os.path.join(BASE, "..", "data")
_pkg_fig = os.path.join(BASE, "..", "figures")
DATA_DIR = _pkg_data if os.path.isdir(_pkg_data) else os.path.join(BASE, "data")
FIG_DIR = _pkg_fig if os.path.isdir(_pkg_fig) else os.path.join(BASE, "figures")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# Dataset generator, inlined from dataset_lib.py (identical logic, same
# defaults/RNG-call order) so this script has no import dependency on
# the rest of the package.
# ---------------------------------------------------------------------
N_PER_CLASS = 2000
RELATIONAL_FRACTION = 0.12
SCALE_FRACTION = 0.04
RATIO = 2.3
DIST_CENTERS = [(4, 4), (-4, 4), (4, -4), (-4, -4)]
DIST_STD = 1.6


def _make_distance_classes(rng, n_per_class, centers, std):
    feats, labels = [], []
    for i, c in enumerate(centers):
        pts = rng.normal(loc=c, scale=std, size=(n_per_class, 2))
        feats.append(pts)
        labels.append(np.full(n_per_class, i))
    return np.vstack(feats), np.concatenate(labels)


def _inject_relational_subgroup(rng, n_total, fraction, ratio, noise_std=0.08):
    is_hidden = np.zeros(n_total, dtype=int)
    n_hidden = int(n_total * fraction)
    idx = rng.choice(n_total, size=n_hidden, replace=False)
    is_hidden[idx] = 1

    feature_3 = np.empty(n_total)
    feature_4 = np.empty(n_total)

    n_bg = n_total - n_hidden
    feature_3[is_hidden == 0] = rng.normal(0, 3.0, size=n_bg)
    feature_4[is_hidden == 0] = rng.normal(0, 3.0, size=n_bg)

    radius = rng.uniform(3, 9, size=n_hidden)
    sign = rng.choice([-1, 1], size=n_hidden)
    base = sign * radius
    feature_3[is_hidden == 1] = base + rng.normal(0, noise_std, size=n_hidden)
    feature_4[is_hidden == 1] = ratio * base + rng.normal(0, noise_std, size=n_hidden)

    return feature_3, feature_4, is_hidden


def _inject_scale_subgroup(rng, n_total, fraction, host_std=3.0, minority_std=0.15):
    is_hidden = np.zeros(n_total, dtype=int)
    n_hidden = int(n_total * fraction)
    idx = rng.choice(n_total, size=n_hidden, replace=False)
    is_hidden[idx] = 1

    feature_5 = np.empty(n_total)
    feature_6 = np.empty(n_total)

    n_bg = n_total - n_hidden
    feature_5[is_hidden == 0] = rng.normal(0, host_std, size=n_bg)
    feature_6[is_hidden == 0] = rng.normal(0, host_std, size=n_bg)

    feature_5[is_hidden == 1] = rng.normal(0, minority_std, size=n_hidden)
    feature_6[is_hidden == 1] = rng.normal(0, minority_std, size=n_hidden)

    return feature_5, feature_6, is_hidden


def generate(seed, n_per_class=N_PER_CLASS):
    """Identical to dataset_lib.generate(seed) with all mechanisms on
    and default strengths -- reproduces the paper's headline benchmark."""
    rng = np.random.default_rng(seed)

    ab, class_label = _make_distance_classes(rng, n_per_class, DIST_CENTERS, DIST_STD)
    feature_1, feature_2 = ab[:, 0], ab[:, 1]
    n = len(class_label)
    perm = rng.permutation(n)
    feature_1, feature_2, class_label = feature_1[perm], feature_2[perm], class_label[perm]

    feature_3, feature_4, hidden_relational = _inject_relational_subgroup(
        rng, n, RELATIONAL_FRACTION, RATIO
    )
    feature_5, feature_6, hidden_scale = _inject_scale_subgroup(
        rng, n, SCALE_FRACTION
    )

    feature_7 = rng.normal(0, 2.0, size=n)
    feature_8 = rng.normal(0, 2.0, size=n)
    feature_9 = rng.normal(0, 2.0, size=n)
    feature_10 = rng.normal(0, 2.0, size=n)

    observable = pd.DataFrame({
        "feature_1": feature_1, "feature_2": feature_2,
        "feature_3": feature_3, "feature_4": feature_4,
        "feature_5": feature_5, "feature_6": feature_6,
        "feature_7": feature_7, "feature_8": feature_8,
        "feature_9": feature_9, "feature_10": feature_10,
        "class_label": class_label,
    })
    hidden = {
        "hidden_relational_label": hidden_relational,
        "hidden_scale_label": hidden_scale,
    }
    return observable, hidden


# ---------------------------------------------------------------------
# ICS pair scoring -- identical to ics_kurtosis_screening.py
# ---------------------------------------------------------------------
CANDIDATE_PAIRS = [
    ("feature_1", "feature_2"), ("feature_3", "feature_4"),
    ("feature_5", "feature_6"), ("feature_7", "feature_8"),
    ("feature_9", "feature_10"), ("feature_2", "feature_5"),
    ("feature_1", "feature_8"),
]
TRUE_PAIR = {"distance": "feature_1,feature_2", "angular": "feature_3,feature_4",
             "density": "feature_5,feature_6"}

N_SEEDS = 80
SEEDS = list(range(1000, 1000 + N_SEEDS))  # same seed range as the rest of Section 6


def ics_pair_score(pair_vals):
    """2x2 ICS (COV4 vs COV1): returns (top generalized eigenvalue,
    |eigenvalue - 1|, eigenvector direction) for the most non-Gaussian
    direction in this 2-D pair. Identical to ics_kurtosis_screening.py."""
    Xc = pair_vals - pair_vals.mean(axis=0)
    n, p = Xc.shape
    COV1 = np.cov(Xc, rowvar=False)
    sq = np.sum(Xc ** 2, axis=1)
    COV4 = (Xc * sq[:, None]).T @ Xc / n / (p + 2)
    eigvals, eigvecs = eigh(COV4, COV1)
    idx = np.argmax(np.abs(eigvals - 1))
    return float(eigvals[idx]), float(abs(eigvals[idx] - 1)), eigvecs[:, idx]


def eigengap_route(observable):
    """Score all candidate pairs, then apply an eigengap rule to the
    SORTED deviation-score spectrum to decide routing confidence
    automatically (no fixed threshold picked by hand).

    Returns: detected_pair, is_confident, gap_ratio, all pair scores.
    """
    scores = {}
    for f1, f2 in CANDIDATE_PAIRS:
        pv = observable[[f1, f2]].values
        _, dev, _ = ics_pair_score(pv)
        scores[f"{f1},{f2}"] = dev

    sorted_items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    sorted_vals = np.array([v for _, v in sorted_items])
    gaps = -np.diff(sorted_vals)  # gaps[i] = sorted_vals[i] - sorted_vals[i+1]
    largest_gap_idx = int(np.argmax(gaps))  # 0 means gap is between rank-1 and rank-2
    # confident routing == the largest gap in the whole spectrum occurs
    # immediately after the top-ranked pair (the eigengap-heuristic
    # analogue of "one dominant component")
    is_confident = (largest_gap_idx == 0)
    gap_ratio = float(gaps[0] / (sorted_vals[0] + 1e-12))

    detected_pair = sorted_items[0][0]
    return detected_pair, is_confident, gap_ratio, scores


# ---------------------------------------------------------------------
# HDBSCAN-based within-pair minority detection (replaces the GMM step)
# ---------------------------------------------------------------------
def hdbscan_minority_detect(pair_vals, direction, min_cluster_frac=0.03, min_samples=None):
    """Replace the GMM step. The GMM in ics_kurtosis_screening.py was
    fit on log(projection^2) along the ICS-detected non-Gaussian
    direction -- a 1-D transformed space, not raw 2-D coordinates -- so
    HDBSCAN is applied to the SAME transformed space here for a like-for-
    like comparison (clustering directly in raw 2-D coordinates was
    tried first and failed: the relational subgroup is elongated and
    density-overlapping with the background blob in raw feature space,
    so a density-based method needs the same non-Gaussianity-revealing
    transform ICS already computed, exactly as the GMM step did).

    No minority-fraction prior is used: min_cluster_size is set from a
    fixed generic percentage of n (3%), not from the known 12% true
    relational fraction, and HDBSCAN's own noise/outlier model (label
    -1) is left to decide how many points are background -- unlike the
    GMM step, which always forced a hard 2-way split of every point.
    """
    n = len(pair_vals)
    Xc = pair_vals - pair_vals.mean(axis=0)
    proj_sq = (Xc @ direction) ** 2
    log_sq = np.log(proj_sq + 1e-9).reshape(-1, 1)
    Xs = StandardScaler().fit_transform(log_sq)

    min_cluster_size = max(5, int(round(min_cluster_frac * n)))
    if min_samples is None:
        min_samples = max(5, min_cluster_size // 3)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(Xs)

    unique, counts = np.unique(labels[labels != -1], return_counts=True)
    if len(unique) == 0:
        # everything called noise -> no structure found, predict all-background
        pred_minority = np.zeros(n, dtype=int)
        n_clusters_found = 0
    else:
        # the "minority" is the cluster with the higher mean log(proj^2)
        # (heavier tail / larger anomaly score), matching the GMM step's
        # "heaviest-mean component = minority" convention; noise points
        # are left as background (0), not folded into the minority, since
        # HDBSCAN noise here is dominated by the diffuse Gaussian
        # background rather than the relational line
        cluster_means = [log_sq[labels == c].mean() for c in unique]
        minority_cluster = unique[np.argmax(cluster_means)]
        pred_minority = (labels == minority_cluster).astype(int)
        n_clusters_found = len(unique)

    return pred_minority, n_clusters_found, labels


# ---------------------------------------------------------------------
# 1. Headline run (seed 42), matching the paper's reporting convention
# ---------------------------------------------------------------------
observable42, hidden42 = generate(seed=42)
detected42, confident42, gap_ratio42, scores42 = eigengap_route(observable42)

f1, f2 = detected42.split(",")
pv42 = observable42[[f1, f2]].values
_, _, direction42 = ics_pair_score(pv42)
pred_minority42, n_clusters42, _ = hdbscan_minority_detect(pv42, direction42)
ari_hdbscan42 = float(adjusted_rand_score(hidden42["hidden_relational_label"], pred_minority42))

# Oracle ceiling on the same detected-pair ICS direction, identical
# convention to ics_kurtosis_screening.py / analyze_pipelines.py
Xc42 = pv42 - pv42.mean(axis=0)
proj_sq42 = (Xc42 @ direction42) ** 2
auc_oracle42 = float(roc_auc_score(hidden42["hidden_relational_label"], proj_sq42))
best_ari42, best_t42 = -1.0, None
for t in np.percentile(proj_sq42, np.arange(80, 99.5, 0.25)):
    p_ = (proj_sq42 >= t).astype(int)
    a = adjusted_rand_score(hidden42["hidden_relational_label"], p_)
    if a > best_ari42:
        best_ari42, best_t42 = a, t

headline = {
    "detected_pair": detected42,
    "routing_correct": detected42 == TRUE_PAIR["angular"],
    "routing_confident_eigengap": confident42,
    "eigengap_ratio": gap_ratio42,
    "pair_scores": scores42,
    "n_clusters_found_hdbscan": n_clusters42,
    "ari_hdbscan": ari_hdbscan42,
    "oracle_ceiling_ari": best_ari42,
    "oracle_ceiling_auc": auc_oracle42,
}

# ---------------------------------------------------------------------
# 2. Multi-seed robustness (N=80, seeds 1000-1079)
# ---------------------------------------------------------------------
rows = []
for seed in SEEDS:
    obs, hid = generate(seed=seed)
    detected, confident, gap_ratio, scores = eigengap_route(obs)
    routing_correct = detected == TRUE_PAIR["angular"]

    df1, df2 = detected.split(",")
    pv = obs[[df1, df2]].values
    _, _, direction = ics_pair_score(pv)
    pred_minority, n_clusters_found, _ = hdbscan_minority_detect(pv, direction)
    ari_hdbscan = float(adjusted_rand_score(hid["hidden_relational_label"], pred_minority))

    # Oracle ceiling on the same detected pair (evaluation-only)
    Xc = pv - pv.mean(axis=0)
    proj_sq = (Xc @ direction) ** 2
    best_a = -1.0
    for t in np.percentile(proj_sq, np.arange(80, 99.5, 1.0)):
        p_ = (proj_sq >= t).astype(int)
        a = adjusted_rand_score(hid["hidden_relational_label"], p_)
        if a > best_a:
            best_a = a

    # Negative control: same HDBSCAN procedure applied to the TRUE
    # scale pair (mechanism C) -- HDBSCAN should not confidently carve
    # out a minority matching hidden_scale_label unless there is
    # genuine density structure (mechanism C is a scale/density
    # mechanism, so unlike mechanism B this is not a pure negative
    # control -- reported for comparison, not as a strict null check).
    pv_true_c = obs[["feature_5", "feature_6"]].values
    _, _, direction_c = ics_pair_score(pv_true_c)
    pred_c, n_clusters_c, _ = hdbscan_minority_detect(pv_true_c, direction_c)
    ari_hdbscan_true_c = float(adjusted_rand_score(hid["hidden_scale_label"], pred_c))

    rows.append({
        "seed": seed,
        "detected_pair": detected,
        "routing_correct": routing_correct,
        "routing_confident_eigengap": confident,
        "eigengap_ratio": gap_ratio,
        "n_clusters_found_hdbscan": n_clusters_found,
        "ari_hdbscan_on_detected_pair": ari_hdbscan,
        "oracle_ari_on_detected_pair": best_a,
        "n_clusters_found_hdbscan_true_scale_pair": n_clusters_c,
        "ari_hdbscan_true_scale_pair": ari_hdbscan_true_c,
    })

raw = pd.DataFrame(rows)
raw.to_csv(os.path.join(DATA_DIR, "ics_eigengap_hdbscan_raw.csv"), index=False)

summary = {
    "n_seeds": N_SEEDS,
    "routing_accuracy": float(raw["routing_correct"].mean()),
    "routing_confident_fraction": float(raw["routing_confident_eigengap"].mean()),
    "mean_eigengap_ratio": float(raw["eigengap_ratio"].mean()),
    "mean_ari_hdbscan_on_detected_pair": float(raw["ari_hdbscan_on_detected_pair"].mean()),
    "sd_ari_hdbscan_on_detected_pair": float(raw["ari_hdbscan_on_detected_pair"].std()),
    "mean_oracle_ari_on_detected_pair": float(raw["oracle_ari_on_detected_pair"].mean()),
    "sd_oracle_ari_on_detected_pair": float(raw["oracle_ari_on_detected_pair"].std()),
    "mean_n_clusters_found_hdbscan": float(raw["n_clusters_found_hdbscan"].mean()),
    "mean_ari_hdbscan_true_scale_pair": float(raw["ari_hdbscan_true_scale_pair"].mean()),
}

results = {
    "description": (
        "Priority 5: eigengap-based automatic routing confidence + "
        "HDBSCAN within-pair clustering, replacing ics_kurtosis_screening.py's "
        "GMM-based (fixed-prior) final-clustering step. No minority-fraction "
        "prior and no fixed number-of-components assumption is used anywhere "
        "in this script; ground-truth labels are used only for ARI/AUC scoring."
    ),
    "headline_seed42": headline,
    "multi_seed_summary_N80": summary,
}

with open(os.path.join(DATA_DIR, "ics_eigengap_hdbscan_results.json"), "w") as f:
    json.dump(results, f, indent=2, default=float)

print(json.dumps(results, indent=2, default=float))

# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.hist(raw["ari_hdbscan_on_detected_pair"], bins=20, alpha=0.75, color="#59a14f",
        label="HDBSCAN (this script, N=80)")
ax.hist(raw["oracle_ari_on_detected_pair"], bins=20, alpha=0.45, color="#999999",
        label="Oracle ceiling (evaluation-only)")
ax.set_xlabel("ARI on detected pair vs. hidden_relational_label")
ax.set_ylabel("Count across 80 seeds")
ax.set_title("Eigengap-routed + HDBSCAN-clustered\nminority recovery (N=80 seeds)")
ax.legend(fontsize=9)

ax = axes[1]
ax.scatter(raw["eigengap_ratio"], raw["ari_hdbscan_on_detected_pair"],
           c=raw["routing_correct"].map({True: "#4e79a7", False: "#e15759"}), alpha=0.7)
ax.set_xlabel("Eigengap ratio (routing confidence)")
ax.set_ylabel("ARI (HDBSCAN, detected pair)")
ax.set_title("Routing confidence vs. downstream ARI\n(blue = correct routing, red = incorrect)")

plt.tight_layout()
fig_path = os.path.join(FIG_DIR, "fig27_ics_eigengap_hdbscan.png")
plt.savefig(fig_path, dpi=170)
plt.close()

print("\nSaved figure to", fig_path)
