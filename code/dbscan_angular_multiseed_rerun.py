"""
dbscan_angular_multiseed_rerun.py
===================================
Fixes a specific apples-to-oranges comparison: the
original auto_diagnostic_pipeline.py (DBSCAN-angular heuristic) was
evaluated on a SINGLE seed (42), while ics_kurtosis_screening.py (the
ICS-based heuristic it is compared against in Section 5.5) was evaluated
across N=80 seeds. This made ICS look more reliable partly because it
was simply tested more thoroughly, not necessarily because it IS more
reliable. This script re-runs the exact same DBSCAN-angular detection and
scoring procedure from auto_diagnostic_pipeline.py -- same thresholds
(DIST_THRESH=0.35, ANG_THRESH=0.05, DENS_THRESH=50.0), same candidate
pairs, same two-stage DBSCAN parameterization (eps=0.01/min_samples=15
for pair-scoring; eps=0.001/min_samples=10 for the final within-pair
membership call) -- across the identical N=80 seed range (1000..1079)
used everywhere else in Section 6 and by ics_kurtosis_screening.py, so
the two heuristics can finally be compared on equal footing.

We do NOT retune thresholds or change the method itself: this is a
like-for-like re-evaluation, not an improvement of the DBSCAN-angular
heuristic.
"""

import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

import dataset_lib as dl
import os
BASE = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE, "..", "data")

N_SEEDS = 80
SEEDS = list(range(1000, 1000 + N_SEEDS))  # identical range to ics_kurtosis_screening.py

CANDIDATE_PAIRS = [
    ("feature_1", "feature_2"), ("feature_3", "feature_4"),
    ("feature_5", "feature_6"), ("feature_7", "feature_8"),
    ("feature_9", "feature_10"), ("feature_2", "feature_5"),
    ("feature_1", "feature_8"),
]
TRUE_PAIR = {"distance": "feature_1,feature_2", "angular": "feature_3,feature_4",
             "density": "feature_5,feature_6"}
DIST_THRESH, ANG_THRESH, DENS_THRESH = 0.35, 0.05, 50.0


def distance_score(pair_vals):
    km = KMeans(n_clusters=4, n_init=5, random_state=0).fit(pair_vals)
    return float(silhouette_score(pair_vals, km.labels_))


def angular_score(pair_vals):
    x, y = pair_vals[:, 0], pair_vals[:, 1]
    angle2 = 2 * np.arctan2(y, x)
    polar = np.column_stack([np.cos(angle2), np.sin(angle2)])
    db = DBSCAN(eps=0.01, min_samples=15).fit(polar)
    labels = db.labels_
    if (labels != -1).any():
        vals, counts = np.unique(labels[labels != -1], return_counts=True)
        frac = counts.max() / len(labels)
        return float(frac) if frac < 0.40 else 0.0
    return 0.0


def density_bimodality_score(pair_vals):
    nn = NearestNeighbors(n_neighbors=10).fit(pair_vals)
    dists, _ = nn.kneighbors(pair_vals)
    log_d = np.log(dists[:, 1:].mean(axis=1) + 1e-9).reshape(-1, 1)
    gmm1 = GaussianMixture(n_components=1, random_state=0).fit(log_d)
    gmm2 = GaussianMixture(n_components=2, random_state=0, n_init=3).fit(log_d)
    return float(gmm1.bic(log_d) - gmm2.bic(log_d))


def run_one_seed(seed):
    observable, hidden = dl.generate(seed)
    class_label = observable["class_label"].values
    hidden_relational = hidden["hidden_relational_label"]
    hidden_scale = hidden["hidden_scale_label"]

    diag = {}
    for f1, f2 in CANDIDATE_PAIRS:
        pv = observable[[f1, f2]].values
        diag[f"{f1},{f2}"] = {
            "distance": distance_score(pv),
            "angular": angular_score(pv),
            "density": density_bimodality_score(pv),
        }

    best_dist_pair = max(diag, key=lambda k: diag[k]["distance"])
    best_ang_pair = max(diag, key=lambda k: diag[k]["angular"])
    best_dens_pair = max(diag, key=lambda k: diag[k]["density"])

    routing = {
        "distance": best_dist_pair if diag[best_dist_pair]["distance"] > DIST_THRESH else None,
        "angular": best_ang_pair if diag[best_ang_pair]["angular"] > ANG_THRESH else None,
        "density": best_dens_pair if diag[best_dens_pair]["density"] > DENS_THRESH else None,
    }
    correct = {
        "distance": routing["distance"] == TRUE_PAIR["distance"],
        "angular": routing["angular"] == TRUE_PAIR["angular"],
        "density": routing["density"] == TRUE_PAIR["density"],
    }

    ari = {"distance": np.nan, "angular": np.nan, "density": np.nan}

    if routing["distance"]:
        f1, f2 = routing["distance"].split(",")
        km = KMeans(n_clusters=4, n_init=10, random_state=0).fit(observable[[f1, f2]].values)
        ari["distance"] = adjusted_rand_score(class_label, km.labels_)

    if routing["angular"]:
        f1, f2 = routing["angular"].split(",")
        x, y = observable[f1].values, observable[f2].values
        angle2 = 2 * np.arctan2(y, x)
        polar = np.column_stack([np.cos(angle2), np.sin(angle2)])
        db = DBSCAN(eps=0.001, min_samples=10).fit(polar)
        labels = db.labels_
        if (labels != -1).any():
            vals, counts = np.unique(labels[labels != -1], return_counts=True)
            biggest = vals[np.argmax(counts)]
            pred = (labels == biggest).astype(int)
        else:
            pred = np.zeros_like(labels)
        ari["angular"] = adjusted_rand_score(hidden_relational, pred)

    if routing["density"]:
        f1, f2 = routing["density"].split(",")
        pv = observable[[f1, f2]].values
        nn = NearestNeighbors(n_neighbors=10).fit(pv)
        dists, _ = nn.kneighbors(pv)
        log_d = np.log(dists[:, 1:].mean(axis=1) + 1e-9).reshape(-1, 1)
        means_init = np.array([[np.percentile(log_d, 5)], [np.percentile(log_d, 60)]])
        weights_init = np.array([0.05, 0.95])
        gmm = GaussianMixture(n_components=2, random_state=0, means_init=means_init,
                               weights_init=weights_init, n_init=1).fit(log_d)
        comp_means = gmm.means_.ravel()
        dense_component = np.argmin(comp_means)
        labels = gmm.predict(log_d)
        pred = (labels == dense_component).astype(int)
        ari["density"] = adjusted_rand_score(hidden_scale, pred)

    return correct, ari


if __name__ == "__main__":
    import os
    import sys

    raw_path = f"{DATA_DIR}/dbscan_angular_multiseed_raw.csv"
    batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 8

    if os.path.exists(raw_path):
        done_df = pd.read_csv(raw_path)
        done_seeds = set(done_df["seed"].tolist())
    else:
        done_df = pd.DataFrame()
        done_seeds = set()

    remaining = [s for s in SEEDS if s not in done_seeds]
    batch = remaining[:batch_size]

    new_rows = []
    for s in batch:
        correct, ari = run_one_seed(s)
        new_rows.append({
            "seed": s,
            "distance_routing_correct": correct["distance"],
            "angular_routing_correct": correct["angular"],
            "density_routing_correct": correct["density"],
            "distance_ari": ari["distance"],
            "angular_ari": ari["angular"],
            "density_ari": ari["density"],
        })

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        combined = pd.concat([done_df, new_df], ignore_index=True) if len(done_df) else new_df
        combined.to_csv(raw_path, index=False)
    else:
        combined = done_df

    n_done = len(combined)
    print(f"Processed {len(new_rows)} new seeds this run; {n_done}/{N_SEEDS} total done.")

    if n_done >= N_SEEDS:
        df = combined
        summary = {
            "n_seeds": N_SEEDS,
            "seed_range": [SEEDS[0], SEEDS[-1]],
            "routing_accuracy": {
                "distance_pct_correct": float(df["distance_routing_correct"].mean() * 100),
                "angular_pct_correct": float(df["angular_routing_correct"].mean() * 100),
                "density_pct_correct": float(df["density_routing_correct"].mean() * 100),
            },
            "ari_when_routed": {
                "distance_mean": float(df["distance_ari"].mean(skipna=True)),
                "distance_sd": float(df["distance_ari"].std(skipna=True)),
                "angular_mean": float(df["angular_ari"].mean(skipna=True)),
                "angular_sd": float(df["angular_ari"].std(skipna=True)),
                "density_mean": float(df["density_ari"].mean(skipna=True)),
                "density_sd": float(df["density_ari"].std(skipna=True)),
            },
            "comparison_to_original_single_seed_42_report": {
                "original_seed42_angular_routing_correct": False,
                "original_seed42_angular_ari": 0.004340086317494413,
                "note": (
                    "The original single-seed report (seed=42) showed 0/1 correct "
                    "routing for the angular/relational pair. The N=80 re-run below "
                    "shows whether that was representative or a single unlucky/lucky draw."
                ),
            },
            "comparison_to_ics_kurtosis_screening_N80": {
                "ics_angular_routing_pct_correct": 100.0,
                "ics_oracle_ari_mean": 0.938,
                "note": (
                    "ICS's 100% routing accuracy and 0.938 oracle ARI (Section 5.5) "
                    "are now compared against a DBSCAN-angular baseline evaluated "
                    "with the SAME N=80 seeds and SAME candidate-pair protocol, "
                    "closing the apples-to-oranges gap in the original comparison."
                ),
            },
        }
        with open(f"{DATA_DIR}/dbscan_angular_multiseed_results.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary, indent=2))
