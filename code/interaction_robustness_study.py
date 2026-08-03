"""
interaction_robustness_study.py
=================================
NEW analysis added on top of the original single-seed (seed=42) report.
It directly answers the two honesty gaps flagged in the original write-up:

  (1) "Results are reported on a single random seed (seed=42); the final
      report should average multiple runs with different seeds."
  (2) The report ASSERTS that combining independent hiding mechanisms is
      more damaging than any one mechanism alone, but never MEASURES the
      size of that interaction effect against the null hypothesis that
      damage is simply additive.

This script does both in one pass:

  A. Multi-seed robustness: regenerate the FULL 3-mechanism benchmark under
     N_SEEDS independent seeds and re-run the naive PCA+KMeans pipeline on
     each, reporting mean +/- std of ARI(class_label) at k=4, ARI at the
     silhouette-selected k, and the k=4 silhouette score itself. This tests
     whether the seed=42 numbers in results.json are representative or a
     lucky/unlucky draw.

  B. 2x2 mechanism-presence ablation ("interaction effect" study): for the
     SAME set of seeds, regenerate the dataset with mechanism B
     (relational) and mechanism C (scale) each independently switched on
     or off, holding mechanism A (distance) and the noise columns fixed in
     distribution. Four configurations:
         A only        (B=off, C=off)
         A + B         (B=on,  C=off)
         A + C         (B=off, C=on)
         A + B + C     (B=on,  C=on)   <- the full benchmark
     For each configuration/seed we run the identical naive PCA(10)->top-2
     ->KMeans(k=4) pipeline and record ARI(class_label). We then test
     whether the damage from combining B and C is additive:

         predicted_drop  = drop(A+B) + drop(A+C)     [null: additive]
         observed_drop   = drop(A+B+C)
         interaction     = observed_drop - predicted_drop

     interaction > 0  => super-additive ("the whole is worse than the sum
                          of its parts") -- this is the quantitative
                          evidence for the paper's central claim.
     interaction ~= 0 => damage is merely additive; the combination is not
                          mechanistically special, only cumulative.
     interaction < 0  => sub-additive (mechanisms partially "shield" A).

A paired Wilcoxon signed-rank test (paired by seed) is used because the
same seed's mechanism-A draw is shared across all four configurations
(dataset_lib.generate keeps the RNG call order identical regardless of
which mechanisms are toggled), so within-seed comparisons are valid
paired comparisons, not independent samples.

Outputs:
  data/interaction_robustness_results.json  -- full numeric results
  data/interaction_robustness_raw.csv       -- per-seed, per-config raw ARIs
  figures/fig22_multiseed_robustness.png
  figures/fig24_interaction_effect.png
"""

import json
import os
BASE = os.path.dirname(os.path.abspath(__file__))
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

DATA_DIR = os.path.join(BASE, "..", "data")
FIG_DIR = os.path.join(BASE, "..", "figures")

N_SEEDS = 80  # bumped from 20 -> 80 in the revision: a post-hoc power analysis
              # (effect_size_power_stats.py) showed the original N=20 run had
              # only ~26% power to detect the observed interaction effect size
              # at alpha=0.05, and that N~76 was needed for 80% power.
SEEDS = list(range(1000, 1000 + N_SEEDS))  # disjoint from the original seed=42
SIL_SAMPLE_SIZE = 2000  # subsample for silhouette_score (O(n^2) otherwise); fixed seed for reproducibility


def run_naive_pipeline(observable, k_grid=(2, 3, 4, 5, 6)):
    feature_cols = [c for c in observable.columns if c != "class_label"]
    X = observable[feature_cols].values
    class_label = observable["class_label"].values

    X_scaled = StandardScaler().fit_transform(X)
    pcs = PCA(n_components=10, random_state=0).fit_transform(X_scaled)
    top2 = pcs[:, :2]

    sil_by_k, ari_by_k, models = {}, {}, {}
    for k in k_grid:
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(top2)
        sil_by_k[k] = silhouette_score(top2, km.labels_, sample_size=SIL_SAMPLE_SIZE, random_state=0)
        ari_by_k[k] = adjusted_rand_score(class_label, km.labels_)
        models[k] = km

    best_k = max(sil_by_k, key=sil_by_k.get)
    return {
        "ari_k4": ari_by_k[4],
        "sil_k4": sil_by_k[4],
        "best_k_by_silhouette": best_k,
        "ari_at_best_k": ari_by_k[best_k],
    }


def part_a_multiseed_robustness():
    rows = []
    for seed in SEEDS:
        observable, _ = dl.generate(seed, include_relational=True, include_scale=True)
        r = run_naive_pipeline(observable)
        r["seed"] = seed
        rows.append(r)
        print(f"  seed={seed} ari_k4={r['ari_k4']:.3f} sil_k4={r['sil_k4']:.3f}", flush=True)
    df = pd.DataFrame(rows)

    summary = {
        "n_seeds": N_SEEDS,
        "ari_k4_mean": float(df["ari_k4"].mean()),
        "ari_k4_std": float(df["ari_k4"].std(ddof=1)),
        "ari_k4_min": float(df["ari_k4"].min()),
        "ari_k4_max": float(df["ari_k4"].max()),
        "sil_k4_mean": float(df["sil_k4"].mean()),
        "sil_k4_std": float(df["sil_k4"].std(ddof=1)),
        "fraction_seeds_where_k4_is_silhouette_optimum": float((df["best_k_by_silhouette"] == 4).mean()),
        "ari_at_best_k_mean": float(df["ari_at_best_k"].mean()),
        "ari_at_best_k_std": float(df["ari_at_best_k"].std(ddof=1)),
        "seed42_reference_ari_k4": 0.11469845191388345,
        "seed42_reference_sil_k4": 0.3822239450351252,
    }
    return df, summary


def part_b_interaction_ablation():
    configs = {
        "A_only": dict(include_relational=False, include_scale=False),
        "A_plus_B": dict(include_relational=True, include_scale=False),
        "A_plus_C": dict(include_relational=False, include_scale=True),
        "A_plus_B_plus_C": dict(include_relational=True, include_scale=True),
    }
    rows = []
    for seed in SEEDS:
        for cfg_name, kwargs in configs.items():
            observable, _ = dl.generate(seed, **kwargs)
            r = run_naive_pipeline(observable, k_grid=(4,))
            rows.append({"seed": seed, "config": cfg_name, "ari_k4": r["ari_k4"], "sil_k4": r["sil_k4"]})
            print(f"  seed={seed} config={cfg_name} ari_k4={r['ari_k4']:.3f}", flush=True)
    raw = pd.DataFrame(rows)

    pivot = raw.pivot(index="seed", columns="config", values="ari_k4")
    pivot = pivot[["A_only", "A_plus_B", "A_plus_C", "A_plus_B_plus_C"]]

    mean_ari = pivot.mean().to_dict()
    std_ari = pivot.std(ddof=1).to_dict()

    drop_B = pivot["A_only"] - pivot["A_plus_B"]
    drop_C = pivot["A_only"] - pivot["A_plus_C"]
    drop_BC_observed = pivot["A_only"] - pivot["A_plus_B_plus_C"]
    drop_BC_predicted_additive = drop_B + drop_C
    interaction_per_seed = drop_BC_observed - drop_BC_predicted_additive

    w_stat, w_p = stats.wilcoxon(drop_BC_observed, drop_BC_predicted_additive)

    summary = {
        "n_seeds": N_SEEDS,
        "mean_ARI_k4_by_config": mean_ari,
        "std_ARI_k4_by_config": std_ari,
        "mean_drop_from_A_only__A_plus_B": float(drop_B.mean()),
        "mean_drop_from_A_only__A_plus_C": float(drop_C.mean()),
        "mean_drop_from_A_only__A_plus_B_plus_C__observed": float(drop_BC_observed.mean()),
        "mean_drop__additive_null_prediction": float(drop_BC_predicted_additive.mean()),
        "mean_interaction_effect": float(interaction_per_seed.mean()),
        "std_interaction_effect": float(interaction_per_seed.std(ddof=1)),
        "interaction_effect_positive_fraction_of_seeds": float((interaction_per_seed > 0).mean()),
        "wilcoxon_observed_vs_additive_statistic": float(w_stat),
        "wilcoxon_observed_vs_additive_pvalue": float(w_p),
        "interpretation": (
            "positive mean_interaction_effect => combining mechanisms B and C damages "
            "recovery of mechanism A MORE than the sum of their individual damages "
            "(super-additive / synergistic collapse); interaction_effect_positive_fraction_of_seeds "
            "reports how consistent this is across independent seeds, not just seed=42."
        ),
    }
    return raw, pivot, summary


def make_figures(msr_df, msr_summary, ablation_raw, ablation_pivot, ablation_summary):
    # Fig 7: multi-seed robustness -- ARI(k=4) distribution across seeds vs seed=42 reference
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.hist(msr_df["ari_k4"], bins=12, color="#4e79a7", alpha=0.8, edgecolor="white")
    ax.axvline(msr_summary["seed42_reference_ari_k4"], color="#e15759", linestyle="--", linewidth=2,
                label=f"seed=42 (reported in main text): {msr_summary['seed42_reference_ari_k4']:.3f}")
    ax.axvline(msr_summary["ari_k4_mean"], color="#333333", linestyle="-", linewidth=2,
                label=f"{N_SEEDS}-seed mean: {msr_summary['ari_k4_mean']:.3f} \u00b1 {msr_summary['ari_k4_std']:.3f}")
    ax.set_xlabel("Naive pipeline ARI (class_label, k=4)")
    ax.set_ylabel("Number of seeds")
    ax.set_title(f"Collapse of the distance mechanism is not a seed=42 artifact (N={N_SEEDS} seeds)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig22_multiseed_robustness.png", dpi=170)
    plt.close()

    # Fig 8: interaction effect bar chart with error bars
    labels = ["A only\n(distance alone)", "A + B\n(+ relational)", "A + C\n(+ scale)",
              "A + B + C\n(full benchmark)"]
    keys = ["A_only", "A_plus_B", "A_plus_C", "A_plus_B_plus_C"]
    means = [ablation_summary["mean_ARI_k4_by_config"][k] for k in keys]
    stds = [ablation_summary["std_ARI_k4_by_config"][k] for k in keys]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    colors = ["#59a14f", "#f28e2b", "#f28e2b", "#e15759"]
    bars = ax.bar(labels, means, yerr=stds, capsize=6, color=colors, alpha=0.9)
    ax.set_ylabel("Naive pipeline ARI (class_label, k=4)")
    ax.set_title(f"Mechanism-presence ablation: combined damage exceeds the additive\n"
                 f"prediction by {ablation_summary['mean_interaction_effect']:.3f} on average "
                 f"(N={N_SEEDS} seeds, Wilcoxon p={ablation_summary['wilcoxon_observed_vs_additive_pvalue']:.2e})")
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, m + s + 0.015, f"{m:.3f}",
                 ha="center", fontsize=9)
    ax.set_ylim(0, max(means) + max(stds) + 0.15)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/fig24_interaction_effect.png", dpi=170)
    plt.close()


def main():
    print("Running Part A: multi-seed robustness ...")
    msr_df, msr_summary = part_a_multiseed_robustness()
    print(json.dumps(msr_summary, indent=2))

    print("\nRunning Part B: mechanism interaction / ablation study ...")
    ablation_raw, ablation_pivot, ablation_summary = part_b_interaction_ablation()
    print(json.dumps(ablation_summary, indent=2))

    make_figures(msr_df, msr_summary, ablation_raw, ablation_pivot, ablation_summary)

    combined = {"multiseed_robustness": msr_summary, "mechanism_interaction_ablation": ablation_summary}
    with open(f"{DATA_DIR}/interaction_robustness_results.json", "w") as f:
        json.dump(combined, f, indent=2, default=float)

    raw_out = ablation_raw.copy()
    raw_out.to_csv(f"{DATA_DIR}/interaction_robustness_raw.csv", index=False)
    msr_df.to_csv(f"{DATA_DIR}/multiseed_robustness_raw.csv", index=False)

    print("\nSaved: data/interaction_robustness_results.json")
    print("Saved: data/interaction_robustness_raw.csv")
    print("Saved: data/multiseed_robustness_raw.csv")
    print("Saved: figures/fig22_multiseed_robustness.png")
    print("Saved: figures/fig24_interaction_effect.png")


if __name__ == "__main__":
    main()
