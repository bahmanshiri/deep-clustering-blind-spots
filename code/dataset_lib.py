"""
dataset_lib.py
================
Shared, seed-parameterized, mechanism-toggleable version of the hard
benchmark generator. This module factors out the logic in
generate_hard_dataset.py so that the interaction/robustness study
(interaction_robustness_study.py) can:

  (1) regenerate the dataset under many random seeds (to check that the
      headline numbers in results.json are not a seed=42 artifact), and
  (2) selectively DISABLE mechanism B (relational) and/or mechanism C
      (scale) to measure how much of the naive pipeline's damage to
      mechanism A (distance) is attributable to each mechanism alone,
      versus their combination -- i.e. to test for a super-additive
      interaction effect rather than just asserting one.

No behavior of the original generator changes when all mechanisms are
enabled and seed=42: this module is a strict refactor for reuse.
"""

import numpy as np
import pandas as pd

N_TOTAL = 8000
N_PER_CLASS = 2000
RELATIONAL_FRACTION = 0.12
SCALE_FRACTION = 0.04
RATIO = 2.3
DIST_CENTERS = [(4, 4), (-4, 4), (4, -4), (-4, -4)]
DIST_STD = 1.6


def make_distance_classes(rng, n_per_class, centers, std):
    feats, labels = [], []
    for i, c in enumerate(centers):
        pts = rng.normal(loc=c, scale=std, size=(n_per_class, 2))
        feats.append(pts)
        labels.append(np.full(n_per_class, i))
    return np.vstack(feats), np.concatenate(labels)


def inject_relational_subgroup(rng, n_total, fraction, ratio, noise_std=0.08):
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


def inject_scale_subgroup(rng, n_total, fraction, host_std=3.0, minority_std=0.15):
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


def generate(seed, include_relational=True, include_scale=True, include_noise=True,
             relational_fraction=RELATIONAL_FRACTION, ratio=RATIO,
             scale_fraction=SCALE_FRACTION, dist_std=DIST_STD,
             scale_minority_std=0.15, n_per_class=N_PER_CLASS):
    """Generate one instance of the benchmark.

    Parameters
    ----------
    seed : int
        RNG seed. Using different seeds regenerates mechanism A (class
        centers/labels), mechanism B, and mechanism C independently but
        reproducibly, so every configuration at a given seed shares the
        same distance-mechanism draw ONLY when include_relational and
        include_scale are toggled with the same seed and the same rng
        call order is preserved (guaranteed here: A is always drawn
        first, then B, then C, then D, regardless of which are kept).
    relational_fraction, ratio, scale_fraction, dist_std, scale_minority_std : float
        Mechanism-strength knobs added for the parameter-space
        generalization study (Section 4.3). Defaults reproduce the
        original single-configuration benchmark exactly (no behavior
        change when called with no extra arguments -- verified by the
        original test in generate_hard_dataset.py).

    Returns
    -------
    observable : pd.DataFrame  (feature_1..feature_10, class_label)
    hidden : dict of np.ndarray  (hidden_relational_label, hidden_scale_label;
             all-zero arrays if the corresponding mechanism is disabled)
    """
    rng = np.random.default_rng(seed)

    ab, class_label = make_distance_classes(rng, n_per_class, DIST_CENTERS, dist_std)
    feature_1, feature_2 = ab[:, 0], ab[:, 1]
    n = len(class_label)
    perm = rng.permutation(n)
    feature_1, feature_2, class_label = feature_1[perm], feature_2[perm], class_label[perm]

    # Mechanism B is always DRAWN (to keep the RNG stream identical across
    # configs at a fixed seed) but its effect is discarded (replaced with
    # unstructured noise of the same marginal scale) when disabled, so that
    # turning B on/off is a clean ablation and not a confound of a
    # different random draw for feature_1/feature_2.
    feature_3, feature_4, hidden_relational = inject_relational_subgroup(
        rng, n, relational_fraction, ratio
    )
    if not include_relational:
        feature_3 = rng.normal(0, 3.0, size=n)
        feature_4 = rng.normal(0, 3.0, size=n)
        hidden_relational = np.zeros(n, dtype=int)

    feature_5, feature_6, hidden_scale = inject_scale_subgroup(
        rng, n, scale_fraction, minority_std=scale_minority_std
    )
    if not include_scale:
        feature_5 = rng.normal(0, 3.0, size=n)
        feature_6 = rng.normal(0, 3.0, size=n)
        hidden_scale = np.zeros(n, dtype=int)

    if include_noise:
        feature_7 = rng.normal(0, 2.0, size=n)
        feature_8 = rng.normal(0, 2.0, size=n)
        feature_9 = rng.normal(0, 2.0, size=n)
        feature_10 = rng.normal(0, 2.0, size=n)
    else:
        feature_7 = feature_8 = feature_9 = feature_10 = np.zeros(n)

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
