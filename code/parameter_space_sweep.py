"""
parameter_space_sweep.py
==========================
Addresses the single most important generalizability gap in the paper's design:
every result in the paper (Theorem 1's numerical
verification, the N=80 multi-seed robustness check, the interaction
re-analysis, the baseline/ICS comparisons) was computed on ONE fixed
benchmark configuration (RELATIONAL_FRACTION=0.12, SCALE_FRACTION=0.04,
RATIO=2.3, DIST_STD=1.6). We do not know, from the original manuscript
alone, whether the collapse and the false-confidence pattern are
properties of this specific corner of parameter space or hold more
broadly.

This script does NOT bolt on an unrelated robustness experiment. It ties
the sweep directly to Theorem 1: the theorem's bound is driven entirely
by epsilon (the largest absolute cross-mechanism correlation), and
epsilon itself is a function of the same mechanism-strength knobs
(RELATIONAL_FRACTION, RATIO, SCALE_FRACTION, DIST_STD) that a careful reader
would want varied. So instead of an ad hoc parameter sweep, we:

  (1) One-at-a-time sensitivity: vary each of the 4 mechanism-strength
      parameters independently (holding the other 3 at the original
      paper's value), N=20 seeds per level, and report how naive-pipeline
      ARI (per mechanism), Silhouette, and epsilon respond.

  (2) A 2D grid over the two parameters that most directly control
      epsilon for mechanism B (RELATIONAL_FRACTION x RATIO), N=15 seeds
      per cell, and explicitly verify that Theorem 1's inequality chain
      (observed EVR error <= Weyl/Gershgorin bound <= (d-1)*epsilon/d)
      holds in EVERY cell, not just at the original paper's single point.

This turns "we tested one configuration" into "we verified the theorem's
predicted relationship between epsilon and the collapse holds across the
parameter space, and the collapse itself is not a knife-edge property of
one specific setting."
"""

import json
import itertools
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

import dataset_lib as dl
import os
BASE = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE, "..", "data")
D = 10  # total feature dimensionality (fixed throughout, as in the paper)


def run_one_config(seed, n_per_class=500, **kwargs):
    """Generate one dataset instance under given mechanism-strength
    kwargs, run the naive pipeline at k=4 (matching the paper's
    headline-comparison convention), and return ARI/Silhouette/epsilon.

    NOTE ON SAMPLE SIZE: this sweep uses n_per_class=500 (2,000 rows
    total) rather than the paper's headline N_PER_CLASS=2,000 (8,000
    rows total), purely for computational tractability across ~900 runs.
    This is a deliberate, disclosed choice for this diagnostic sweep
    only; none of the paper's main headline numbers (Sections 3, 6) are
    affected, and the qualitative pattern (which is what this sweep is
    testing for) is not expected to depend on sample size beyond the
    normal seed-to-seed noise already characterized in Section 6.2."""
    observable, hidden = dl.generate(seed, n_per_class=n_per_class, **kwargs)
    feature_cols = [c for c in observable.columns if c != "class_label"]
    X = observable[feature_cols].values
    class_label = observable["class_label"].values
    hidden_relational = hidden["hidden_relational_label"]
    hidden_scale = hidden["hidden_scale_label"]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = PCA(n_components=D, random_state=0)
    pcs = pca.fit_transform(Xs)
    top2 = pcs[:, :2]

    km = KMeans(n_clusters=4, n_init=3, random_state=0).fit(top2)
    sil = silhouette_score(top2, km.labels_)

    ari_A = adjusted_rand_score(class_label, km.labels_)
    ari_B = adjusted_rand_score(hidden_relational, km.labels_)
    ari_C = adjusted_rand_score(hidden_scale, km.labels_)

    # --- epsilon / within-block rho, exactly as in theory_variance_budget.py ---
    R = np.corrcoef(Xs, rowvar=False)
    blocks = {"A": [0, 1], "B": [2, 3], "C": [4, 5],
              "D1": [6], "D2": [7], "D3": [8], "D4": [9]}
    block_mask = np.zeros_like(R, dtype=bool)
    for idx in blocks.values():
        for i in idx:
            for j in idx:
                block_mask[i, j] = True
    E = R.copy()
    E[block_mask] = 0.0
    epsilon = float(np.max(np.abs(E)))
    rho_B = float(R[2, 3])

    R_B = R.copy()
    R_B[~block_mask] = 0.0
    eig_true = np.linalg.eigvalsh(R)[::-1]
    eig_RB = np.linalg.eigvalsh(R_B)[::-1]
    observed_evr_error = float(np.max(np.abs(eig_true - eig_RB))) / D
    theorem1_bound_evr = (D - 1) * epsilon / D
    theorem1_holds = bool(observed_evr_error <= theorem1_bound_evr + 1e-9)

    return {
        "ari_A": ari_A, "ari_B": ari_B, "ari_C": ari_C,
        "silhouette": sil, "epsilon": epsilon, "rho_B": rho_B,
        "observed_evr_error": observed_evr_error,
        "theorem1_bound_evr": theorem1_bound_evr,
        "theorem1_holds": theorem1_holds,
    }


# ===========================================================================
# PART 1: One-at-a-time sensitivity sweeps (N=20 seeds per level)
# ===========================================================================
N_SEEDS_1D = 20
SEEDS_1D = list(range(2000, 2000 + N_SEEDS_1D))

sweep_grids = {
    "relational_fraction": [0.02, 0.04, 0.08, 0.12, 0.20, 0.30],
    "ratio": [1.2, 1.6, 2.3, 3.0, 4.0],
    "scale_fraction": [0.01, 0.02, 0.04, 0.08, 0.12],
    "dist_std": [0.8, 1.6, 2.4, 3.2],
}

one_at_a_time_rows = []
for param_name, levels in sweep_grids.items():
    for level in levels:
        kwargs = {param_name: level}
        recs = [run_one_config(s, **kwargs) for s in SEEDS_1D]
        for s, r in zip(SEEDS_1D, recs):
            row = {"param": param_name, "value": level, "seed": s}
            row.update(r)
            one_at_a_time_rows.append(row)

df_1d = pd.DataFrame(one_at_a_time_rows)
df_1d.to_csv(f"{DATA_DIR}/parameter_sweep_1d_raw.csv", index=False)

summary_1d = (
    df_1d.groupby(["param", "value"])
    .agg(
        mean_ari_A=("ari_A", "mean"), sd_ari_A=("ari_A", "std"),
        mean_ari_B=("ari_B", "mean"), sd_ari_B=("ari_B", "std"),
        mean_ari_C=("ari_C", "mean"), sd_ari_C=("ari_C", "std"),
        mean_silhouette=("silhouette", "mean"), sd_silhouette=("silhouette", "std"),
        mean_epsilon=("epsilon", "mean"),
        mean_rho_B=("rho_B", "mean"),
        theorem1_violations=("theorem1_holds", lambda x: int((~x).sum())),
        n=("seed", "count"),
    )
    .reset_index()
)

# ===========================================================================
# PART 2: 2D grid over (relational_fraction x ratio) -- the two knobs that
# most directly set epsilon/rho_B for mechanism B, which Theorem 1 depends on
# ===========================================================================
N_SEEDS_2D = 15
SEEDS_2D = list(range(3000, 3000 + N_SEEDS_2D))

grid_fractions = [0.02, 0.04, 0.08, 0.12, 0.20, 0.30]
grid_ratios = [1.2, 1.6, 2.3, 3.0, 4.0]

grid_rows = []
for frac, rat in itertools.product(grid_fractions, grid_ratios):
    recs = [run_one_config(s, relational_fraction=frac, ratio=rat) for s in SEEDS_2D]
    for s, r in zip(SEEDS_2D, recs):
        row = {"relational_fraction": frac, "ratio": rat, "seed": s}
        row.update(r)
        grid_rows.append(row)

df_2d = pd.DataFrame(grid_rows)
df_2d.to_csv(f"{DATA_DIR}/parameter_sweep_2d_raw.csv", index=False)

summary_2d = (
    df_2d.groupby(["relational_fraction", "ratio"])
    .agg(
        mean_ari_B=("ari_B", "mean"), sd_ari_B=("ari_B", "std"),
        mean_silhouette=("silhouette", "mean"),
        mean_epsilon=("epsilon", "mean"),
        mean_rho_B=("rho_B", "mean"),
        mean_observed_evr_error=("observed_evr_error", "mean"),
        mean_theorem1_bound_evr=("theorem1_bound_evr", "mean"),
        theorem1_violations=("theorem1_holds", lambda x: int((~x).sum())),
        n=("seed", "count"),
    )
    .reset_index()
)

total_cells = len(grid_fractions) * len(grid_ratios)
total_runs_2d = len(df_2d)
total_violations_2d = int(total_runs_2d - df_2d["theorem1_holds"].sum())

result = {
    "purpose": (
        "Resolves the 'single benchmark configuration' generalizability gap "
        "by (1) an one-at-a-time sensitivity sweep over all 4 mechanism-strength "
        "parameters and (2) a 2D grid over the two parameters (relational_fraction, "
        "ratio) that jointly set epsilon for mechanism B, with Theorem 1's bound "
        "checked in every single cell."
    ),
    "one_at_a_time": {
        "seeds_per_level": N_SEEDS_1D,
        "grids": sweep_grids,
        "original_paper_values": {
            "relational_fraction": dl.RELATIONAL_FRACTION,
            "ratio": dl.RATIO,
            "scale_fraction": dl.SCALE_FRACTION,
            "dist_std": dl.DIST_STD,
        },
        "summary": summary_1d.to_dict(orient="records"),
    },
    "grid_2d": {
        "seeds_per_cell": N_SEEDS_2D,
        "relational_fraction_levels": grid_fractions,
        "ratio_levels": grid_ratios,
        "total_cells": total_cells,
        "total_runs": total_runs_2d,
        "total_theorem1_violations": total_violations_2d,
        "theorem1_holds_in_every_cell": bool(total_violations_2d == 0),
        "summary": summary_2d.to_dict(orient="records"),
    },
}

with open(f"{DATA_DIR}/parameter_sweep_results.json", "w") as f:
    json.dump(result, f, indent=2)

print(f"One-at-a-time: {len(df_1d)} runs across {len(one_at_a_time_rows)} rows")
print(f"2D grid: {total_runs_2d} runs across {total_cells} cells; "
      f"Theorem 1 violations: {total_violations_2d}")
print(json.dumps({"grid_2d_holds_everywhere": result["grid_2d"]["theorem1_holds_in_every_cell"]}, indent=2))
