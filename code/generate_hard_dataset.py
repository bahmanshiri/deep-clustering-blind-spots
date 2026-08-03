"""
generate_hard_dataset.py
=========================
Synthetic benchmark generator combining THREE independent, simultaneous
"hiding mechanisms" that make subgroups invisible to naive clustering
pipelines. All feature names are generic (feature_1 ... feature_10) and
domain-neutral by design.

Mechanisms embedded:
  (A) Distance-based structure  -> feature_1, feature_2
      4 main classes, Gaussian blobs (moderately overlapping -> harder
      than the previous benchmark).
  (B) Relational (ratio) hiding -> feature_3, feature_4
      A minority subgroup defined by a near-constant ratio
      feature_4 / feature_3 ~= RATIO, invisible to Euclidean distance.
  (C) Scale/variance hiding     -> feature_5, feature_6
      A minority subgroup with drastically smaller variance sitting
      at the SAME location (centroid) as a much more diffuse background
      cloud -> invisible to centroid-distance clustering, only visible
      to local density / variance-aware methods.
  (D) Pure noise                -> feature_7 ... feature_10
      No structure at all, included to raise dimensionality and to
      dilute PCA the way real tabular data does.

Reproducible with a fixed random seed. Produces two files:
  observable_dataset_hard.csv        (features + class_label only)
  hidden_labels_hard_EVAL_ONLY.csv   (ground-truth hidden-subgroup labels,
                                       kept separate on purpose so that
                                       discovery pipelines cannot cheat)
"""

import numpy as np
import pandas as pd
import os
BASE = os.path.dirname(os.path.abspath(__file__))

RNG_SEED = 42
N_TOTAL = 8000
N_PER_CLASS = 2000          # 4 classes x 2000 = 8000
RELATIONAL_FRACTION = 0.12  # ~12% relational hidden subgroup
SCALE_FRACTION = 0.04       # ~4% scale/variance hidden subgroup
RATIO = 2.3                 # feature_4 / feature_3 for relational subgroup

rng = np.random.default_rng(RNG_SEED)


def make_distance_classes(n_per_class, centers, std):
    """Mechanism A: 4 Gaussian blobs, moderately overlapping."""
    feats, labels = [], []
    for i, c in enumerate(centers):
        pts = rng.normal(loc=c, scale=std, size=(n_per_class, 2))
        feats.append(pts)
        labels.append(np.full(n_per_class, i))
    return np.vstack(feats), np.concatenate(labels)


def inject_relational_subgroup(n_total, fraction, ratio, noise_std=0.08):
    """Mechanism B: subset of rows lie on a ratio line feature_4≈ratio*feature_3,
    with radius (distance from origin) varying freely -> invisible to distance."""
    is_hidden = np.zeros(n_total, dtype=int)
    n_hidden = int(n_total * fraction)
    idx = rng.choice(n_total, size=n_hidden, replace=False)
    is_hidden[idx] = 1

    feature_3 = np.empty(n_total)
    feature_4 = np.empty(n_total)

    # background (non-hidden): independent, unstructured
    n_bg = n_total - n_hidden
    feature_3[is_hidden == 0] = rng.normal(0, 3.0, size=n_bg)
    feature_4[is_hidden == 0] = rng.normal(0, 3.0, size=n_bg)

    # hidden subgroup: on the ratio line, radius varies 3..9, random sign/direction
    radius = rng.uniform(3, 9, size=n_hidden)
    sign = rng.choice([-1, 1], size=n_hidden)
    base = sign * radius
    feature_3[is_hidden == 1] = base + rng.normal(0, noise_std, size=n_hidden)
    feature_4[is_hidden == 1] = ratio * base + rng.normal(0, noise_std, size=n_hidden)

    return feature_3, feature_4, is_hidden


def inject_scale_subgroup(n_total, fraction, host_std=3.0, minority_std=0.15):
    """Mechanism C: a tiny low-variance cloud sitting at the SAME centroid
    as a much more diffuse background cloud -> centroid-distance is useless,
    only local density/variance separates them."""
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


def main():
    # --- Mechanism A: 4 main classes (harder = more overlap than before) ---
    centers = [(4, 4), (-4, 4), (4, -4), (-4, -4)]
    dist_std = 1.6  # was 1.3 in the previous (easier) benchmark
    ab, class_label = make_distance_classes(N_PER_CLASS, centers, dist_std)
    feature_1, feature_2 = ab[:, 0], ab[:, 1]

    n = len(class_label)
    assert n == N_TOTAL

    # shuffle row order so class blocks aren't contiguous
    perm = rng.permutation(n)
    feature_1, feature_2, class_label = feature_1[perm], feature_2[perm], class_label[perm]

    # --- Mechanism B: relational hidden subgroup ---
    feature_3, feature_4, hidden_relational = inject_relational_subgroup(
        n, RELATIONAL_FRACTION, RATIO
    )

    # --- Mechanism C: scale/variance hidden subgroup ---
    feature_5, feature_6, hidden_scale = inject_scale_subgroup(
        n, SCALE_FRACTION
    )

    # --- Mechanism D: pure irrelevant noise (4 columns) ---
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

    hidden_eval_only = pd.DataFrame({
        "hidden_relational_label": hidden_relational,
        "hidden_scale_label": hidden_scale,
    })

    observable.to_csv(os.path.join(BASE, "..", "data", "observable_dataset_hard.csv"), index=False)
    hidden_eval_only.to_csv(os.path.join(BASE, "..", "data", "hidden_labels_hard_EVAL_ONLY.csv"), index=False)

    print("N rows:", n)
    print("class_label counts:\n", pd.Series(class_label).value_counts().sort_index())
    print("relational hidden count:", hidden_relational.sum(),
          f"({hidden_relational.mean()*100:.1f}%)")
    print("scale hidden count:", hidden_scale.sum(),
          f"({hidden_scale.mean()*100:.1f}%)")
    overlap = ((hidden_relational == 1) & (hidden_scale == 1)).sum()
    print("rows in BOTH hidden subgroups (overlap, by chance):", overlap)


if __name__ == "__main__":
    main()
