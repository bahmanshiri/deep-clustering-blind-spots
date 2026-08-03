"""
scaling_experiments.py
=======================
Tests whether the collapse finding generalizes beyond d=10 features and
beyond the specific noise-column variance used in the main benchmark.

Two sweeps, both built on top of dataset_lib.generate() (mechanisms A/B/C
unchanged) with EXTRA noise columns appended:

  (1) Dimensionality sweep: d_total in {10, 20, 50, 100} by adding
      (d_total - 6) extra i.i.d. Gaussian noise columns (std=2.0, same as
      the original 4 noise columns) to the existing 6 structured columns.
  (2) Noise-scale sweep: fixed d_total=10 (original), but the 4 noise
      columns' std swept over {0.5, 1.0, 2.0, 4.0, 8.0} to test whether
      the collapse depends on noise being comparably scaled to signal.

For each condition we run the naive pipeline (StandardScaler -> PCA(all)
-> top-2 -> KMeans, k=4) across N_SEEDS seeds and report mean/SD ARI and
Silhouette, exactly as in interaction_robustness_study.py, so numbers are
directly comparable to the d=10 headline results.

HONEST CAVEAT: this reuses the SAME generative mechanisms; it tests
robustness to dimensionality/noise-scale, not to entirely different
mechanism families, categorical/missing data, or real tabular data (see
real_data_validation.py for that).
"""

import json
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
N_SEEDS = 15
SEEDS = list(range(2000, 2000 + N_SEEDS))
SIL_SAMPLE = 2000


def generate_with_extra_noise(seed, n_extra_noise=0, noise_std=2.0, base_noise_std=2.0):
    """Wrap dataset_lib.generate to append n_extra_noise columns (std=noise_std)
    and optionally override the std of the original 4 noise columns."""
    observable, hidden = dl.generate(seed, include_relational=True, include_scale=True)
    rng = np.random.default_rng(seed + 999_983)  # independent stream, does not
                                                  # perturb the RNG call order
                                                  # used by dl.generate itself
    n = len(observable)

    if base_noise_std != 2.0:
        for col in ["feature_7", "feature_8", "feature_9", "feature_10"]:
            observable[col] = rng.normal(0, base_noise_std, size=n)

    for i in range(n_extra_noise):
        observable[f"extra_noise_{i+1}"] = rng.normal(0, noise_std, size=n)

    return observable, hidden


def run_naive(observable, k=4):
    feature_cols = [c for c in observable.columns if c != "class_label"]
    X = observable[feature_cols].values
    class_label = observable["class_label"].values
    Xs = StandardScaler().fit_transform(X)
    n_comp = min(10, Xs.shape[1])
    pcs = PCA(n_components=n_comp, random_state=0).fit_transform(Xs)
    top2 = pcs[:, :2]
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(top2)
    sil = silhouette_score(top2, km.labels_, sample_size=min(SIL_SAMPLE, len(top2)), random_state=0)
    ari = adjusted_rand_score(class_label, km.labels_)
    evr = PCA(random_state=0).fit(Xs).explained_variance_ratio_
    return ari, sil, evr


def sweep_dimensionality():
    d_targets = [10, 20, 50, 100]
    rows = []
    for d in d_targets:
        n_extra = d - 6  # 6 structured cols (A,B,C) always present
        for seed in SEEDS:
            observable, _ = generate_with_extra_noise(seed, n_extra_noise=n_extra)
            ari, sil, evr = run_naive(observable)
            rows.append({"d_total": d, "seed": seed, "ari_k4": ari, "sil_k4": sil,
                         "top1_evr": float(evr[0]), "evr_uniformity_std": float(np.std(evr))})
        print(f"  d_total={d} done", flush=True)
    df = pd.DataFrame(rows)
    summary = df.groupby("d_total").agg(
        ari_mean=("ari_k4", "mean"), ari_std=("ari_k4", "std"),
        sil_mean=("sil_k4", "mean"), sil_std=("sil_k4", "std"),
        top1_evr_mean=("top1_evr", "mean"),
    ).reset_index()
    return df, summary


def sweep_noise_scale():
    noise_stds = [0.5, 1.0, 2.0, 4.0, 8.0]
    rows = []
    for ns in noise_stds:
        for seed in SEEDS:
            observable, _ = generate_with_extra_noise(seed, n_extra_noise=0, base_noise_std=ns)
            ari, sil, evr = run_naive(observable)
            rows.append({"noise_std": ns, "seed": seed, "ari_k4": ari, "sil_k4": sil,
                         "top1_evr": float(evr[0])})
        print(f"  noise_std={ns} done", flush=True)
    df = pd.DataFrame(rows)
    summary = df.groupby("noise_std").agg(
        ari_mean=("ari_k4", "mean"), ari_std=("ari_k4", "std"),
        sil_mean=("sil_k4", "mean"), sil_std=("sil_k4", "std"),
        top1_evr_mean=("top1_evr", "mean"),
    ).reset_index()
    return df, summary


def main():
    print("Sweep 1: dimensionality (d = 10,20,50,100) ...")
    dim_df, dim_summary = sweep_dimensionality()
    print(dim_summary)

    print("\nSweep 2: noise-column scale (std = 0.5..8.0, d=10) ...")
    noise_df, noise_summary = sweep_noise_scale()
    print(noise_summary)

    dim_df.to_csv(f"{DATA_DIR}/scaling_dimensionality_raw.csv", index=False)
    noise_df.to_csv(f"{DATA_DIR}/scaling_noise_raw.csv", index=False)

    out = {
        "dimensionality_sweep": {
            "n_seeds": N_SEEDS,
            "by_d": dim_summary.to_dict(orient="records"),
        },
        "noise_scale_sweep": {
            "n_seeds": N_SEEDS,
            "by_noise_std": noise_summary.to_dict(orient="records"),
        },
    }
    with open(f"{DATA_DIR}/scaling_experiments_results.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("\nSaved scaling_experiments_results.json, scaling_dimensionality_raw.csv, scaling_noise_raw.csv")


if __name__ == "__main__":
    main()
