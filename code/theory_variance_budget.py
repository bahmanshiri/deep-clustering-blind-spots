"""
theory_variance_budget.py
==========================
Formal(ish) structural account of WHY explained variance spreads
near-uniformly across all 10 PCs when 3 comparable-scale, mutually
independent mechanisms are mixed with noise -- and a direct empirical
test of that account against the actual data.

HONESTY NOTE: this is a structural / block-diagonal argument with
numerical verification, not a general asymptotic theorem for arbitrary
generative processes. We are explicit about that scope limitation in
the writeup; we do not claim a distribution-free proof.

Argument (Proposition 1, informal):
  Let the standardized (correlation) matrix R of the d=10 features be
  partitioned into 2x2 (or 1x1, for pure-noise columns) diagonal blocks
  R_A, R_B, R_C, R_D1..D4, one per mechanism, with all OFF-block entries
  equal to the population cross-mechanism correlation, which is exactly
  0 in expectation because the mechanisms are generated from
  independent random draws (Sections 3 of the paper / dataset_lib.py).
  If R is EXACTLY block-diagonal, its eigenvalues are exactly the union
  of the eigenvalues of the individual blocks (a standard linear-algebra
  fact for block-diagonal matrices). Each 2x2 mechanism block has
  eigenvalues (1+rho_i, 1-rho_i) where rho_i is the within-mechanism
  correlation; each pure-noise column contributes a unit eigenvalue.
  Because rho_A, rho_B, rho_C are all small in magnitude (the mechanisms
  are constructed to be distance-, angle-, and density-based rather than
  linearly correlated -- see below), all 10 eigenvalues cluster near 1,
  which is exactly the near-uniform explained-variance spread reported
  empirically (Section 5.1). This is a STRUCTURAL, not merely
  coincidental, explanation: it follows from (a) block-diagonality
  (cross-mechanism independence) and (b) low within-block linear
  correlation (rho_i approx 0), both of which are properties we can
  measure directly from the sample correlation matrix.

We verify this in three steps below:
  1. Compute the empirical sample correlation matrix and confirm it is
     approximately block-diagonal (report max |off-block correlation|).
  2. Compute rho_i (the (1,2) entry of each 2x2 diagonal block) and the
     analytical eigenvalues (1+rho_i, 1-rho_i) it predicts.
  3. Compare the analytical "block-diagonal-approximation" eigenvalue
     spectrum to the TRUE PCA eigenvalues of the full 10x10 correlation
     matrix (which does include the small off-block terms), reporting
     the L1 and max-abs approximation error.
"""

import json
import numpy as np
import pandas as pd
import os
BASE = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE, "..", "data")

observable = pd.read_csv(f"{DATA_DIR}/observable_dataset_hard.csv")
feature_cols = [c for c in observable.columns if c != "class_label"]
X = observable[feature_cols].values
Xs = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)

R = np.corrcoef(Xs, rowvar=False)  # 10x10 sample correlation matrix

blocks = {"A": [0, 1], "B": [2, 3], "C": [4, 5],
          "D1": [6], "D2": [7], "D3": [8], "D4": [9]}

# --- Step 1: block-diagonality check ---
block_mask = np.zeros_like(R, dtype=bool)
for idx in blocks.values():
    for i in idx:
        for j in idx:
            block_mask[i, j] = True
off_block_vals = R[~block_mask]
max_abs_off_block = float(np.max(np.abs(off_block_vals)))
mean_abs_off_block = float(np.mean(np.abs(off_block_vals)))

# --- Step 2: within-block correlations + analytical eigenvalues ---
analytical_eigs = []
within_block_rho = {}
for name, idx in blocks.items():
    if len(idx) == 2:
        rho = R[idx[0], idx[1]]
        within_block_rho[name] = float(rho)
        analytical_eigs.extend([1 + rho, 1 - rho])
    else:
        within_block_rho[name] = None
        analytical_eigs.append(1.0)
analytical_eigs = np.array(sorted(analytical_eigs, reverse=True))
analytical_evr = analytical_eigs / analytical_eigs.sum()

# --- Step 3: true PCA eigenvalues of the FULL correlation matrix ---
true_eigs = np.linalg.eigvalsh(R)[::-1]  # descending
true_evr = true_eigs / true_eigs.sum()

l1_error = float(np.sum(np.abs(true_evr - analytical_evr)))
max_error = float(np.max(np.abs(true_evr - analytical_evr)))

result = {
    "max_abs_off_block_correlation": max_abs_off_block,
    "mean_abs_off_block_correlation": mean_abs_off_block,
    "within_block_rho": within_block_rho,
    "analytical_eigenvalues_block_diagonal_approx": analytical_eigs.tolist(),
    "analytical_explained_variance_ratio": analytical_evr.tolist(),
    "true_pca_eigenvalues_full_matrix": true_eigs.tolist(),
    "true_pca_explained_variance_ratio": true_evr.tolist(),
    "l1_approximation_error": l1_error,
    "max_abs_approximation_error": max_error,
    "interpretation": (
        "If the block-diagonal-approximation EVR curve tracks the true EVR "
        "curve closely (small L1/max error) while ALSO being near-uniform "
        "(no eigenvalue >> 1), this supports the structural claim that "
        "near-uniform spread follows from (i) cross-mechanism independence "
        "(near-zero off-block correlation) and (ii) low within-mechanism "
        "linear correlation (each block's own rho is small), rather than "
        "being a coincidence of this particular seed."
    ),
}

# --- Theorem 1: a genuine, general, provable perturbation bound -----------
# (added to give an eigenvalue-perturbation-theory
# argument, not just a numerically-verified structural claim. This uses two
# standard, textbook results -- Weyl's inequality and the Gershgorin circle
# theorem -- applied to this block structure. It is a real theorem with a
# real proof, not a heuristic; see the paper's Theorem 1 for the full
# statement and proof. Here we verify the resulting bound holds on the data.)
E = R.copy()
E[block_mask] = 0.0          # E = cross-block entries only (zero elsewhere)
R_B = R.copy()
R_B[~block_mask] = 0.0       # block-diagonal matrix built from R's own within-block entries
assert np.allclose(R_B + E, R)

eig_true = np.linalg.eigvalsh(R)[::-1]
eig_RB = np.linalg.eigvalsh(R_B)[::-1]
weyl_exact_spectral_norm_E = float(np.max(np.abs(np.linalg.eigvalsh(E))))
gershgorin_bound = float(np.max(np.sum(np.abs(E), axis=1)))  # data-specific, still a valid bound
epsilon = float(np.max(np.abs(E)))                            # max |cross-block correlation|
d = R.shape[0]
loose_general_bound = (d - 1) * epsilon                       # general, distribution-free bound

max_actual_eig_diff = float(np.max(np.abs(eig_true - eig_RB)))

theorem_check = {
    "theorem_statement": (
        "Theorem 1 (cross-block eigenvalue perturbation bound). Let R be a "
        "dxd correlation matrix partitioned into blocks, R = R_B + E where "
        "R_B keeps only within-block entries of R and E keeps only "
        "cross-block entries. If every cross-block correlation has "
        "|R_ij| <= epsilon, then for every eigenvalue index i: "
        "|lambda_i(R) - lambda_i(R_B)| <= ||E||_2 <= (d-1)*epsilon. "
        "Proof: Weyl's inequality gives the first step (|lambda_i(R)-lambda_i(R_B)| "
        "<= ||R-R_B||_2 = ||E||_2 for symmetric R,R_B); the Gershgorin circle "
        "theorem applied to the zero-diagonal symmetric matrix E gives "
        "||E||_2 <= max_i sum_j|E_ij| <= (d-1)*epsilon since each row has "
        "at most d-1 off-diagonal entries each bounded by epsilon. Dividing "
        "by trace(R)=trace(R_B)=d gives the same bound, ~epsilon, for "
        "explained-variance-ratio differences, for ANY d -- this is the "
        "general, distribution-free statement the block-diagonal empirical "
        "check above (Proposition 1) is a numerically-verified special case of."
    ),
    "epsilon_max_abs_cross_block_correlation": epsilon,
    "d": int(d),
    "observed_max_eigenvalue_difference": max_actual_eig_diff,
    "weyl_exact_spectral_norm_bound": weyl_exact_spectral_norm_E,
    "gershgorin_bound_data_specific": gershgorin_bound,
    "loose_general_distribution_free_bound_d_minus_1_times_epsilon": loose_general_bound,
    "chain_holds": bool(
        max_actual_eig_diff <= weyl_exact_spectral_norm_E + 1e-9
        and weyl_exact_spectral_norm_E <= gershgorin_bound + 1e-9
        and gershgorin_bound <= loose_general_bound + 1e-9
    ),
    "same_quantities_in_explained_variance_ratio_units": {
        "observed_max_EVR_error": max_actual_eig_diff / d,
        "weyl_exact_bound": weyl_exact_spectral_norm_E / d,
        "loose_general_bound": loose_general_bound / d,
    },
}
result["theorem_1_verification"] = theorem_check

with open(f"{DATA_DIR}/theory_variance_budget_results.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
