"""
ics_kurtosis_parameter_sweep.py
==================================
Closes a specific generalizability gap left open by ics_kurtosis_screening.py
and flagged in internal review: that script's headline result (ICS kurtosis
screening routes to the correct relational-mechanism pair in 80/80 seeds,
i.e. 100%) was, like every other result before parameter_space_sweep.py,
computed at a SINGLE fixed benchmark configuration (RELATIONAL_FRACTION=0.12,
RATIO=2.3). parameter_space_sweep.py already stress-tested Theorem 1's bound
and the naive pipeline's collapse across a 6x5 grid over
(relational_fraction, ratio) -- the two knobs that jointly set epsilon /
rho_B for mechanism B -- but did NOT re-run the ICS routing comparison on
that same grid. This script does exactly that, so ICS's positive result is
held to the identical generalization standard as the paper's other claims,
rather than being the one result exempted from the parameter-space sweep.

Design, matched 1:1 to parameter_space_sweep.py's Part 2 (2D grid) for
direct comparability:
  - SAME grid: relational_fraction in {0.02, 0.04, 0.08, 0.12, 0.20, 0.30}
               ratio              in {1.2, 1.6, 2.3, 3.0, 4.0}
               (6 x 5 = 30 cells)
  - SAME seeds-per-cell (N=15) and SAME reduced n_per_class=500 convention
    used throughout parameter_space_sweep.py for computational tractability
    across ~450 runs (disclosed there and here; does not affect the paper's
    headline N=80 numbers in Section 5.5, which use the original
    n_per_class=2000).
  - A fresh, non-overlapping seed range (5000-5014) so no run here reuses
    a seed already used by parameter_space_sweep.py's own grid (3000-3014).

For every cell we report: routing accuracy (fraction of seeds where ICS's
top-kurtosis-anomaly pair equals the true relational pair feature_3,feature_4),
the mean kurtosis-anomaly score on the true pair vs. the strongest competing
pair, and the oracle-threshold ARI on whichever pair was actually detected
(so a routing failure's cost is visible, not just its frequency) -- exactly
the same three-tier logic already used in ics_kurtosis_screening.py.

Outputs:
  data/ics_parameter_sweep_raw.csv
  data/ics_parameter_sweep_results.json
  figures/fig15_ics_parameter_sweep.png
"""

import json
import itertools
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.metrics import adjusted_rand_score

import dataset_lib as dl

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
BASE = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE, "..", "data")
FIG_DIR = os.path.join(BASE, "..", "figures")

CANDIDATE_PAIRS = [
    ("feature_1", "feature_2"), ("feature_3", "feature_4"),
    ("feature_5", "feature_6"), ("feature_7", "feature_8"),
    ("feature_9", "feature_10"), ("feature_2", "feature_5"),
    ("feature_1", "feature_8"),
]
TRUE_RELATIONAL_PAIR = "feature_3,feature_4"

# Same grid as parameter_space_sweep.py Part 2
GRID_FRACTIONS = [0.02, 0.04, 0.08, 0.12, 0.20, 0.30]
GRID_RATIOS = [1.2, 1.6, 2.3, 3.0, 4.0]
N_SEEDS_PER_CELL = 15
SEEDS = list(range(5000, 5000 + N_SEEDS_PER_CELL))  # disjoint from 3000-3014
N_PER_CLASS = 500  # matches parameter_space_sweep.py's tractability convention


def ics_pair_score(pair_vals):
    Xc = pair_vals - pair_vals.mean(axis=0)
    n, p = Xc.shape
    COV1 = np.cov(Xc, rowvar=False)
    sq = np.sum(Xc ** 2, axis=1)
    COV4 = (Xc * sq[:, None]).T @ Xc / n / (p + 2)
    eigvals, eigvecs = eigh(COV4, COV1)
    idx = np.argmax(np.abs(eigvals - 1))
    return float(abs(eigvals[idx] - 1)), eigvecs[:, idx]


def run_one(seed, relational_fraction, ratio):
    observable, hidden = dl.generate(
        seed, relational_fraction=relational_fraction, ratio=ratio,
        n_per_class=N_PER_CLASS,
    )
    hidden_relational = hidden["hidden_relational_label"]

    scores = {}
    for f1, f2 in CANDIDATE_PAIRS:
        dev, _ = ics_pair_score(observable[[f1, f2]].values)
        scores[f"{f1},{f2}"] = dev
    detected_pair = max(scores, key=scores.get)
    correct = detected_pair == TRUE_RELATIONAL_PAIR

    # Oracle-threshold ARI on whichever pair was actually routed to
    f1, f2 = detected_pair.split(",")
    _, direction = ics_pair_score(observable[[f1, f2]].values)
    Xc = observable[[f1, f2]].values - observable[[f1, f2]].values.mean(axis=0)
    proj_sq = (Xc @ direction) ** 2
    best_ari = -1.0
    for pct in np.arange(80, 99.5, 1.0):
        thr = np.percentile(proj_sq, pct)
        pred = (proj_sq >= thr).astype(int)
        a = adjusted_rand_score(hidden_relational, pred)
        if a > best_ari:
            best_ari = a

    strongest_competitor = max(
        (v for k, v in scores.items() if k != TRUE_RELATIONAL_PAIR), default=np.nan
    )

    return {
        "relational_fraction": relational_fraction,
        "ratio": ratio,
        "seed": seed,
        "detected_pair": detected_pair,
        "routing_correct": correct,
        "kurtosis_dev_true_pair": scores[TRUE_RELATIONAL_PAIR],
        "kurtosis_dev_strongest_competitor": strongest_competitor,
        "oracle_ari_on_detected_pair": best_ari,
    }


rows = []
for frac, rat in itertools.product(GRID_FRACTIONS, GRID_RATIOS):
    for seed in SEEDS:
        rows.append(run_one(seed, frac, rat))

df = pd.DataFrame(rows)
df.to_csv(f"{DATA_DIR}/ics_parameter_sweep_raw.csv", index=False)

cell_summary = (
    df.groupby(["relational_fraction", "ratio"])
    .agg(
        routing_accuracy=("routing_correct", "mean"),
        n_correct=("routing_correct", "sum"),
        n=("seed", "count"),
        mean_kurtosis_dev_true_pair=("kurtosis_dev_true_pair", "mean"),
        mean_kurtosis_dev_strongest_competitor=("kurtosis_dev_strongest_competitor", "mean"),
        mean_oracle_ari_on_detected_pair=("oracle_ari_on_detected_pair", "mean"),
    )
    .reset_index()
)

total_cells = len(GRID_FRACTIONS) * len(GRID_RATIOS)
total_runs = len(df)
overall_accuracy = float(df["routing_correct"].mean())
worst_cell = cell_summary.loc[cell_summary["routing_accuracy"].idxmin()]
n_cells_below_100 = int((cell_summary["routing_accuracy"] < 1.0).sum())
n_cells_at_100 = int((cell_summary["routing_accuracy"] >= 1.0).sum())

results = {
    "purpose": (
        "Re-runs ICS kurtosis-screening routing (ics_kurtosis_screening.py, "
        "Section 5.5) across the identical (relational_fraction x ratio) "
        "2D grid already used to stress-test Theorem 1 (parameter_space_sweep.py, "
        "Section 4.3), so ICS's positive routing result is held to the same "
        "generalization standard as the rest of the paper rather than being "
        "reported at a single configuration only."
    ),
    "grid": {
        "relational_fraction_levels": GRID_FRACTIONS,
        "ratio_levels": GRID_RATIOS,
        "seeds_per_cell": N_SEEDS_PER_CELL,
        "seed_range": [SEEDS[0], SEEDS[-1]],
        "n_per_class": N_PER_CLASS,
        "total_cells": total_cells,
        "total_runs": total_runs,
    },
    "overall_routing_accuracy": overall_accuracy,
    "n_cells_at_100pct_routing": n_cells_at_100,
    "n_cells_below_100pct_routing": n_cells_below_100,
    "worst_cell": {
        "relational_fraction": float(worst_cell["relational_fraction"]),
        "ratio": float(worst_cell["ratio"]),
        "routing_accuracy": float(worst_cell["routing_accuracy"]),
    },
    "original_paper_configuration_check": cell_summary[
        (cell_summary["relational_fraction"] == 0.12) & (cell_summary["ratio"] == 2.3)
    ].to_dict(orient="records"),
    "cell_summary": cell_summary.to_dict(orient="records"),
}

with open(f"{DATA_DIR}/ics_parameter_sweep_results.json", "w") as f:
    json.dump(results, f, indent=2, default=float)

print(json.dumps({
    "total_runs": total_runs,
    "overall_routing_accuracy": overall_accuracy,
    "n_cells_at_100pct_routing": n_cells_at_100,
    "n_cells_below_100pct_routing": n_cells_below_100,
    "worst_cell": results["worst_cell"],
}, indent=2))

# --- Figure: heatmap of routing accuracy across the grid ---
pivot = cell_summary.pivot(index="relational_fraction", columns="ratio", values="routing_accuracy")
fig, ax = plt.subplots(figsize=(6.5, 5))
im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index)
ax.set_xlabel("ratio")
ax.set_ylabel("relational_fraction")
ax.set_title(f"ICS routing accuracy across the Theorem-1 parameter grid\n"
             f"({N_SEEDS_PER_CELL} seeds/cell, {total_runs} runs total)")
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        val = pivot.values[i, j]
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                color="black" if val > 0.4 else "white", fontsize=9)
# mark the original paper's single configuration
orig_i = list(pivot.index).index(0.12)
orig_j = list(pivot.columns).index(2.3)
ax.add_patch(plt.Rectangle((orig_j - 0.5, orig_i - 0.5), 1, 1, fill=False,
                            edgecolor="blue", linewidth=3))
fig.colorbar(im, ax=ax, label="routing accuracy")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig15_ics_parameter_sweep.png", dpi=150)
print("Saved figure to fig15_ics_parameter_sweep.png (blue box = original paper's configuration)")
