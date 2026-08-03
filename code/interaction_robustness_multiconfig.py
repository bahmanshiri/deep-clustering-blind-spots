"""
interaction_robustness_multiconfig.py
======================================
Closes the single-configuration limitation flagged in Limitations/Section 6.3
of v5: the N=80 mechanism-interaction null result (interaction_robustness_study.py,
part B) was previously reported at ONE fixed benchmark configuration
(relational_fraction=0.12, ratio=2.3 -- the paper's defaults). This script
re-runs the identical ablation (same pipeline, same paired-Wilcoxon test
against the additive null) at three additional points already used
elsewhere in the paper's own parameter-space sweep (Section 4.3's
grid_fractions / grid_ratios), so the interaction finding is checked across
the same parameter space the collapse itself was already shown to
generalize over -- not a new, cherry-picked grid.

Configurations (relational_fraction, ratio):
  weak    : (0.04, 1.6)   -- weaker relational mechanism, closer to background
  default : (0.12, 2.3)   -- the paper's existing reported configuration (not
                             re-run here; pulled from the existing result file
                             for the summary table)
  strong  : (0.20, 3.0)   -- stronger, more separated relational mechanism
  extreme : (0.30, 4.0)   -- upper end of the existing 4.3 grid

N=80 seeds per configuration, disjoint seed streams per configuration to
avoid any shared-seed leakage across configs.

Outputs:
  data/interaction_robustness_multiconfig_results.json
  figures/fig23_interaction_multiconfig.png
"""

import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

import dataset_lib as dl

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
BASE = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE, "..", "data")
FIG_DIR = os.path.join(BASE, "..", "figures")

N_SEEDS = 80
SIL_SAMPLE_SIZE = 2000

CONFIGS = {
    "weak":    dict(relational_fraction=0.04, ratio=1.6, seed_offset=5000),
    "strong":  dict(relational_fraction=0.20, ratio=3.0, seed_offset=6000),
    "extreme": dict(relational_fraction=0.30, ratio=4.0, seed_offset=7000),
}

# Existing default-configuration result, loaded rather than re-run, so the
# comparison table has all four points without duplicating the original run.
with open(f"{DATA_DIR}/interaction_robustness_results.json") as f:
    _existing = json.load(f)["mechanism_interaction_ablation"]


def run_naive_pipeline(observable, k_grid=(4,)):
    feature_cols = [c for c in observable.columns if c != "class_label"]
    X = observable[feature_cols].values
    class_label = observable["class_label"].values

    X_scaled = StandardScaler().fit_transform(X)
    pcs = PCA(n_components=10, random_state=0).fit_transform(X_scaled)
    top2 = pcs[:, :2]

    sil_by_k, ari_by_k = {}, {}
    for k in k_grid:
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(top2)
        sil_by_k[k] = silhouette_score(top2, km.labels_, sample_size=SIL_SAMPLE_SIZE, random_state=0)
        ari_by_k[k] = adjusted_rand_score(class_label, km.labels_)

    return {"ari_k4": ari_by_k[4], "sil_k4": sil_by_k[4]}


def run_config(name, relational_fraction, ratio, seed_offset):
    seeds = list(range(seed_offset, seed_offset + N_SEEDS))
    sub_configs = {
        "A_only": dict(include_relational=False, include_scale=False),
        "A_plus_B": dict(include_relational=True, include_scale=False),
        "A_plus_C": dict(include_relational=False, include_scale=True),
        "A_plus_B_plus_C": dict(include_relational=True, include_scale=True),
    }
    rows = []
    for seed in seeds:
        for cfg_name, kwargs in sub_configs.items():
            observable, _ = dl.generate(
                seed, relational_fraction=relational_fraction, ratio=ratio, **kwargs
            )
            r = run_naive_pipeline(observable)
            rows.append({"seed": seed, "config": cfg_name, "ari_k4": r["ari_k4"]})
        print(f"  [{name}] seed={seed} done", flush=True)

    raw = pd.DataFrame(rows)
    pivot = raw.pivot(index="seed", columns="config", values="ari_k4")
    pivot = pivot[["A_only", "A_plus_B", "A_plus_C", "A_plus_B_plus_C"]]

    drop_B = pivot["A_only"] - pivot["A_plus_B"]
    drop_C = pivot["A_only"] - pivot["A_plus_C"]
    drop_BC_observed = pivot["A_only"] - pivot["A_plus_B_plus_C"]
    drop_BC_predicted_additive = drop_B + drop_C
    interaction_per_seed = drop_BC_observed - drop_BC_predicted_additive

    w_stat, w_p = stats.wilcoxon(drop_BC_observed, drop_BC_predicted_additive)

    return {
        "relational_fraction": relational_fraction,
        "ratio": ratio,
        "n_seeds": N_SEEDS,
        "mean_ARI_k4_by_config": pivot.mean().to_dict(),
        "mean_interaction_effect": float(interaction_per_seed.mean()),
        "std_interaction_effect": float(interaction_per_seed.std(ddof=1)),
        "interaction_effect_positive_fraction_of_seeds": float((interaction_per_seed > 0).mean()),
        "wilcoxon_pvalue": float(w_p),
    }


def main():
    results = {
        "default (existing, N=80, not re-run here)": {
            "relational_fraction": 0.12,
            "ratio": 2.3,
            "n_seeds": _existing["n_seeds"],
            "mean_ARI_k4_by_config": _existing["mean_ARI_k4_by_config"],
            "mean_interaction_effect": _existing["mean_interaction_effect"],
            "std_interaction_effect": _existing["std_interaction_effect"],
            "interaction_effect_positive_fraction_of_seeds": _existing[
                "interaction_effect_positive_fraction_of_seeds"
            ],
            "wilcoxon_pvalue": _existing["wilcoxon_observed_vs_additive_pvalue"],
        }
    }
    for name, cfg in CONFIGS.items():
        print(f"Running configuration: {name} (relational_fraction={cfg['relational_fraction']}, ratio={cfg['ratio']})")
        results[name] = run_config(name, cfg["relational_fraction"], cfg["ratio"], cfg["seed_offset"])

    summary = {
        "purpose": (
            "Re-runs the mechanism-interaction ablation (originally single-configuration, "
            "N=80) at 3 additional points from the existing Section 4.3 parameter grid, "
            "to test whether the reported null-interaction finding generalizes across "
            "mechanism strength or was specific to the default configuration."
        ),
        "configurations": results,
        "all_four_null_or_not": {
            name: (
                "interaction effect present"
                if abs(r["mean_interaction_effect"]) > 0.02 and r["wilcoxon_pvalue"] < 0.05
                else "null (equivalent to additive)"
            )
            for name, r in results.items()
        },
    }

    with open(f"{DATA_DIR}/interaction_robustness_multiconfig_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary["all_four_null_or_not"], indent=2))

    # Figure: interaction effect (mean +/- std) across configurations
    names = list(results.keys())
    means = [results[n]["mean_interaction_effect"] for n in names]
    stds = [results[n]["std_interaction_effect"] for n in names]
    labels = [n.split(" ")[0] for n in names]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.bar(labels, means, yerr=stds, capsize=6, color="#4e79a7", alpha=0.9)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Mean interaction effect (observed - additive prediction)")
    ax.set_title("Mechanism-interaction null result across 4 benchmark configurations\n(N=80 seeds each; error bars = 1 std)")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig23_interaction_multiconfig.png", dpi=170)
    plt.close()

    print("\nDone. Wrote:")
    print(f"  {DATA_DIR}/interaction_robustness_multiconfig_results.json")
    print(f"  {FIG_DIR}/fig23_interaction_multiconfig.png")


if __name__ == "__main__":
    main()
