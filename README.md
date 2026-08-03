# Multi-Mechanism Blind Spots — Reproducibility Package

This package contains the full paper plus all code, data, and figures
needed to regenerate every number and figure in it.

**Submission file:** `paper/BlindSpots_Neurocomputing_v21_XRefAudited.docx`.
This is a cross-reference-audited manuscript prepared for Neurocomputing
(Elsevier) and supersedes every earlier docx described elsewhere in this
README (`v19`/`v20`/`v21_FINAL`/`v21_REGENERATED`/`v22_FINAL`); only this
file is kept in `paper/`. Its figures are numbered 1-26, sequentially, in
the order they actually appear in the text: Figures 1-3 (Section 3,
theory), Figures 4-6 (Section 4.2, CIC-IDS2017 domain-matched
validation), Figure 7 (Section 4.3, Wine/Breast Cancer/Digits check),
Figures 8-10 (Section 4.4, dimensionality/parameter sweeps), Figure 11
(Section 5.2, Kernel PCA/t-SNE), Figure 12 (Section 5.2, blind Spectral
Clustering/DBSCAN comparison), Figure 13 (Section 5.2, DEC/IDEC
comparison), Figures 14-22 (Section 5.3/6, ICS routing and mechanism
panels), Figures 23-25 (Section 6.3, interaction/sensitivity), Figure 26
(Section 6.5, recovery-matrix summary). `figures/`, `code/generate_paper.js`,
and this README are all synchronized to this numbering (see "Audit
findings" below for what that synchronization pass did).

**Note on Figure 27:** `figures/fig27_ics_eigengap_hdbscan.png` (Priority
5, see the reviewer-response table below) was added in v23 and is now
integrated as new Section 6.5 ("Automatic Within-Pair Detection
Revisited: Eigengap-Based Routing and HDBSCAN") in
`paper/BlindSpots_Neurocomputing_v23_XRefAudited.docx`, placed after
Section 6.4 and before Section 7 (Discussion) so no existing figure or
section number changes. Table 8 and Table 12 were each given one new
row for this result, and Section 5.3's GMM-failure paragraph now
forward-references Section 6.5. `code/generate_paper.js` was updated
with matching content (as its own "Section 6.6", since its internal
numbering already lagged the docx's consolidated numbering before this
change — see "Known gaps" below).

## Audit findings, this pass (against `BlindSpots_Neurocomputing_v21_XRefAudited.docx`)

Every code/data/figure file in this package was cross-checked against
the final submission docx: every script name the paper's Appendix A
(Table 13) and body text cite by name is present in `code/`; every one
of the 26 figures embedded in the docx was identified by exact byte
content and matched against `figures/`; every data file Appendix A lists
is present in `data/`. The following issues were found and fixed:

- **Figures renumbered.** `figures/` used an older numbering scheme
  (`fig1`-`fig25` + `fig7b`) that didn't match the document's actual
  Figure 1-26 order. Every file is now named `fig01_...png`-`fig26_...png`
  to match the docx exactly, verified by comparing image bytes, not
  filenames.
- **Figure 13 recovered.** `figures/fig13_dec_idec_comparison.png` (the
  DEC/IDEC comparison) is now present, matches the image the docx
  embeds, and is correctly referenced by `code/generate_paper.js`.
- **`data/cicids2017/` populated from the raw archive.** All eight files
  Appendix A (Table 13) lists —
  `test_A_distance.csv`, `test_B_relational.csv`, `test_C_scale.csv`,
  `robustness_A_raw.csv`, `robustness_A_log.csv`, `robustness_B.csv`,
  `robustness_C.csv`, and `CICIDS2017_validation_REPORT.md` — were
  produced by actually running `code/cicids2017/01_extract_pool.py`
  through `05_visualize.py` against the raw CIC-IDS2017
  `MachineLearningCVE` archive (not included here for size — see
  `code/cicids2017/raw/PLACE_RAW_CSVS_HERE.txt`). The rerun reproduced
  the paper's Figure 4 caption (ARI = 0.47) and every mean±SD value in
  Tables 2-4 to 3 decimal places; full detail and the pre-registered
  candidate-screening rationale are in `CICIDS2017_validation_REPORT.md`.
- **`data/radial_centroid_baseline_results.json` regenerated.**
  `code/radial_centroid_baseline.py` now writes this file; its oracle
  ARI values (0.875 for mechanism B, 0.880 for mechanism C) match what
  `generate_paper.js`/the manuscript already cite.
- **`code/generate_paper.js` synchronized to the 26-figure structure.**
  All 26 `Img(...)` calls now reference the correct filename, figure
  number, and caption for their position in the document; the
  previously-absent Figure 13 (DEC/IDEC) block was added in the correct
  section and position; Section 4's subsections were reordered so
  figures appear in ascending order (1→26) when the script is run;
  stale in-prose "Figure N"/"Section X.Y" cross-references were
  corrected to match. Running `node generate_paper.js` produces a
  44-page docx with all 26 figures in the correct order (verified by
  converting to PDF and checking `pdftotext` output against the figure
  list above).
- **13 figure-generating Python scripts carried the same stale
  pre-renumbering filenames as `generate_paper.js`** (e.g.
  `interaction_robustness_study.py` wrote `fig15_multiseed_robustness.png`
  and `fig16_interaction_effect.png` — the *current* fig15/fig16 are ICS
  parameter-sweep and ARI comparison, unrelated figures; two other
  scripts baked the wrong figure number directly into an image's title
  text). Every `savefig(...)` call across
  `make_figures.py`, `make_figures_v2.py`, `make_figures_v3.py`,
  `make_figures_v4_radial_update.py`, `make_figure_parameter_sweep.py`,
  `ics_kurtosis_screening.py`, `ics_kurtosis_parameter_sweep.py`,
  `interaction_robustness_study.py`, `interaction_robustness_multiconfig.py`,
  `baselines_comparison.py`, `sample_complexity_simulation.py`,
  `sensitivity_robustness.py`, and `dec_idec_figure.py` now writes to the
  correct fig01-fig26 filename, and the two baked-in image titles were
  corrected. Every one of these scripts (aside from `deep_clustering_baseline.py`
  and `DEC_IDEC_Colab_Baseline.ipynb`, which need `torch`/a GPU this
  environment doesn't have) was re-run end to end; all reproduced the
  numbers already reported in the paper (breakdown
  points c=0.60/0.40/0.10, ICS routing 100% in all 30 cells, Theorem 1
  holding in all 30 grid cells, DEC/IDEC seed-fractions 55%/58.75%, etc.)
  and wrote their figures to the correct filenames. `make_figures.py`/
  `make_figures_v2.py` also had a missing `import os` (a leftover from
  before path-portability was fixed); added.
- No zero-byte or corrupted files remain anywhere in the package.

## Honesty summary (read this first)

This project's computational work is reported exactly as it turned out,
not adjusted after the fact to fit a cleaner narrative. Some of it
confirmed the central claim more strongly (e.g. the collapse gets
*worse*, not better, as dimensionality grows, and the CIC-IDS2017 rerun
in Section 4.2 shows naive ARI at or below chance for mechanisms B/C —
a stronger real-data collapse than the synthetic result). Some of it
**overturned** an earlier finding: a "super-additive mechanism
interaction" result observed in an N=20 pilot did **not** replicate at
N=80 with proper statistical power (Section 6.3 of the paper). Some of
it **narrowed** the paper's scope: nonlinear reduction (Kernel PCA,
t-SNE) largely rescues the distance mechanism, and the collapse does
not reproduce on 2 of 3 unrelated real datasets tested. A later addition
**partially
overturned** an earlier claim that the relational mechanism is
"invisible to every generic front end": Invariant Coordinate Selection
routes to the correct relational-mechanism pair with 100% accuracy
across 80 seeds (Section 5.5) — though it does NOT solve automatic
within-pair thresholding, which remains an honestly reported open
problem, and it does NOT rescue the density/scale mechanism (explicit
negative control, mean oracle ARI 0.467 vs. 0.938 for the relational
mechanism). A deep-clustering (autoencoder) baseline added later told
a similar story: a single seed=42 run looked like a clean partial
success on the relational mechanism (ARI=0.505), but an N=80 sweep
showed this was seed-dependent and bimodal (recovers mechanism A in
20% of seeds, mechanism B in 49%, both together in 0%, and mechanism C
in none) -- reported as a distribution, not the single flattering
number. A follow-up test of a substantially wider autoencoder
architecture, run specifically to check whether more network capacity
would resolve this instability, found that it does not: the wider
network recovers mechanism B in only 18% of seeds (down from 49%) and
recovers neither mechanism in 54% of seeds (up from 31%), despite a
better average reconstruction error. Bottleneck capacity is not the
limiting factor. A final addition, made after paper review, corrected the
paper's own terminology: mechanisms B and C were described throughout
as "invisible to Euclidean distance," but a simple radial-distance-
from-centroid statistic (radial_centroid_baseline.py, Section 5.6)
recovers both at oracle ARI (0.875, 0.880 across 80 seeds) matching or
exceeding the hand-informed pipeline's reference ceilings. The
collapse documented throughout this paper is real and unaffected, but
the correct, narrower explanation is that it is specific to K-Means'
multi-centroid partitioning of non-blob-shaped structure, not to
Euclidean distance as a general-purpose statistic -- the paper's
language was revised accordingly wherever it appeared (abstract,
Section 1.1, Section 4's benchmark table, Section 5.2, Section 6.1,
Section 6.5, Section 8, Section 10) rather than only in one place.
A later addition (v22) tested whether a *purpose-built* deep-clustering
objective, rather than a generic autoencoder, would change this picture:
DEC and IDEC, run at the wider architecture this paper's own Future Work
had called for, make the distance mechanism's recovery substantially
*more* reliable (55-59% of seeds vs. 20% for the generic autoencoder) but
make the relational mechanism's recovery *worse*, not better -- 0 of 80
seeds, down from the generic autoencoder's unreliable but nonzero 40-49%.
Added capacity and a properly engineered clustering loss sharpened this
paper's central collapse for the minority mechanism rather than resolving
it; this is reported as the more decisive, not more convenient, of the
two directions the result could have gone.
All of this is reported in the paper itself, not hidden.
Two planned extensions were left for future work in earlier versions: a
UMAP comparison, and validation on a real-world domain-matched dataset.
The domain-matched validation is now delivered (v20, Section 4.2b):
real, unmodified CIC-IDS2017 network-traffic rows were mapped to
mechanisms A/B/C after an empirical, pre-registered screening of every
candidate real attack type against each mechanism's geometric
definition (`data/cicids2017/CICIDS2017_validation_REPORT.md` documents
the full candidate comparison, not just the winner). The result
**strengthens** the paper's central claim for mechanisms B and C — naive
ARI is at or below chance on real traffic, not merely low — and
reproduces Section 5.6's radial-oracle-vs-hand-informed ordering for
both once that ordering is stated correctly. It also surfaces one new,
real-data-specific failure mode not present in the idealized synthetic
benchmark: Silhouette undercounts k for the distance mechanism (k=3
against a true k=4) in 20/20 resamples. Note on Section 5.6 consistency:
the mechanism-B real-data result is not an "ordering reversal" relative
to Section 5.6 — the ordering (hand-informed above radial-oracle) is the
same in both the synthetic and real-data settings, just with a much
wider gap on real data (`CHANGELOG_v19_to_v20.md`). A UMAP comparison
remains future
work (Section 9), as does domain-matched validation on banking/clinical
data and a jointly-occurring (rather than per-mechanism) real-data
interaction replication. The ICS-related literature search (Section 2,
Section 5.5) is explicitly flagged as non-exhaustive, not as a
systematic review.

## Directory structure

```
paper/            BlindSpots_Neurocomputing_v21_XRefAudited.docx  (submission file;
                  the only paper file kept in this package — no older docx/pdf,
                  no changelogs)
code/             all Python + one Node.js (paper generation) script, see below
code/cicids2017/  the 5-script CIC-IDS2017 domain-validation pipeline (Section 4.2)
data/             all generated CSV/JSON data (regeneratable from code/)
data/cicids2017/  test_A_distance.csv, test_B_relational.csv, test_C_scale.csv,
                  robustness_A_raw.csv, robustness_A_log.csv, robustness_B.csv,
                  robustness_C.csv, and CICIDS2017_validation_REPORT.md, all
                  produced by an actual run of code/cicids2017/ against the raw
                  CIC-IDS2017 archive (see Appendix A; the raw archive itself is
                  not redistributed here for size — download separately, see
                  code/cicids2017/raw/PLACE_RAW_CSVS_HERE.txt)
figures/          all 26 figures, renumbered fig01_...png through fig26_...png to
                  match the final manuscript's actual Figure 1-26 order exactly
                  (regeneratable from code/ and code/cicids2017/)
```

## Reproducing everything from scratch

```bash
pip install -r requirements.txt --break-system-packages
cd code

# 1. Generate the synthetic benchmark
python3 generate_hard_dataset.py

# 2. Core analysis (naive vs hand-informed pipelines, seed 42)
python3 analyze_pipelines.py

# 3. Multi-seed robustness + interaction ablation (N=80 seeds, ~70s)
python3 interaction_robustness_study.py

# 4. Theory verification (Proposition 1)
python3 theory_variance_budget.py

# 5. Dimensionality / noise-scale scaling sweeps (~15s)
python3 scaling_experiments.py

# 6. Real-data validation (Wine / Breast Cancer / Digits)
python3 real_data_validation.py

# 7. Stronger baselines (Sparse PCA, Kernel PCA, t-SNE)
python3 stronger_baselines.py

# 7a. (v19 addition) Isomap and MDS front ends, same protocol as above,
#     added after a literature search surfaced a closely related
#     comparison (Amate et al., 2026); see CHANGELOG_v18_to_v19.md
python3 isomap_mds_baseline.py

# 7b. Deep-clustering (autoencoder + K-Means) baseline: minimal and wide
#     architectures, each at N=80 seeds, run because a headline seed=42
#     run and a smaller N=20 pilot both turned out to be unrepresentative
#     (requires PyTorch and a GPU; several hours on a single GPU;
#     resumable, safe to re-run)
python3 deep_clustering_baseline.py

# 8. Effect size / power analysis / OLS interaction decomposition
python3 effect_size_power_stats.py

# 9. NMI/Purity + Silhouette-ARI correlation
python3 metrics_and_anticorrelation.py

# 10. Automatic mechanism-detection prototype (DBSCAN-angular / BIC-density)
python3 auto_diagnostic_pipeline.py

# 11. ICS pairwise kurtosis-anomaly screening, N=80 robustness
python3 ics_kurtosis_screening.py

# 11a. (v23, Priority 5) Eigengap-based routing confidence + HDBSCAN
#      within-pair clustering, replacing the GMM step above; produces
#      Figure 27 (requires `pip install hdbscan`)
python3 ICS_Eigengap_HDBSCAN.py

# 11b. Re-run ICS routing across the same 30-cell (relational_fraction, ratio)
#      grid used for Theorem 1 (450 runs total); confirms 100% routing
#      accuracy holds in every cell, not only the default configuration
#      (Section 5.5); produces Figure 15
python3 ics_kurtosis_parameter_sweep.py

# 12. All figures except Figures 5-6 (unified style, final/authoritative)
python3 make_figures_v3.py
# (make_figures.py and make_figures_v2.py are earlier, superseded
#  first-pass figure scripts, kept for provenance; you do not need to
#  run them -- make_figures_v3.py reads only saved data/ outputs and
#  regenerates every figure it covers from scratch, including Figure 17.)

# 13. Parameter-space generalization sweep tied to Theorem 1
#     (~15s at n_per_class=500; 850 runs total); produces Figures 5-6
python3 parameter_space_sweep.py
python3 make_figure_parameter_sweep.py

# 14. Re-run DBSCAN-angular baseline across the same N=80 seeds as ICS,
#     fixing a single-seed-vs-N=80 apples-to-oranges comparison
#     (~7s/seed at full n_per_class; run in batches, script is resumable
#     via `python3 dbscan_angular_multiseed_rerun.py <batch_size>`)
python3 dbscan_angular_multiseed_rerun.py 80

# 14a2. (v22 addition) Purpose-built deep-clustering baseline: DEC and IDEC
#       (Section 5.3.1), latent_dim=10, N=80 seeds. Run on Google Colab
#       (GPU runtime) via the notebook, not as a local script, since it
#       needs the same GPU dependency as deep_clustering_baseline.py above;
#       expect several hours. Produces dec_idec_results.json and
#       dec_idec_multiseed_raw.csv in data/.
#   -> open code/DEC_IDEC_Colab_Baseline.ipynb in Google Colab, Runtime > Run all
python3 dec_idec_figure.py   # regenerates Figure 13 from the CSV above

# 14b. Radial-distance-from-centroid baseline for mechanisms B and C
#      (N=80 seeds); stress-tests and corrects the paper's own
#      "invisible to Euclidean distance" language (Section 5.6)
python3 radial_centroid_baseline.py

# 14c. Regenerate Figures 9 and 17 to include the radial-centroid
#      baseline (all other figures unaffected; see make_figures_v3.py)
python3 make_figures_v4_radial_update.py

# 15. Rebuild the paper docx (requires Node.js + the `docx` npm package)
node generate_paper.js

# 16. (v6 addition) Re-run the mechanism-interaction ablation at 3 additional
#     parameter-grid configurations, to test whether the Section 6.3 null
#     result generalizes beyond the single originally-tested configuration
#     (~2 min at N=80 seeds/config)
python3 interaction_robustness_multiconfig.py

# 17. (v18 addition, reviewer Priority 1) Sample-complexity check on the
#     "Dimension-Free" framing: does the sample size needed to control
#     cross-mechanism noise grow like log(d) or like d*log(d)? Tests the
#     entrywise quantity (Corollary 1) and the spectral-norm quantity
#     (Theorem 1/2) separately, since they turn out to scale differently.
#     CPU-only, ~1-2 min; produces Figure 1.
python3 sample_complexity_simulation.py

# 18. (v18 addition, reviewer Priority 3) Fair baseline comparison:
#     Spectral Clustering and DBSCAN applied blindly (same naive PCA-
#     top-2 front end, no oracle feature selection) vs. naive KMeans, on
#     the synthetic benchmark and on Wine/Breast Cancer/Digits.
#     CPU-only, <1 min; produces Figure 12.
python3 baselines_comparison.py

# 19. (v18 addition, reviewer Priority 6) Sensitivity / breakdown-point
#     analysis: injects a shared-latent-factor cross-mechanism
#     correlation (Cholesky-style contamination) at increasing strength
#     and re-runs the hand-informed detectors to find each mechanism's
#     breakdown point. Replaces the i.i.d.-mechanism TOST interaction
#     test. CPU-only, ~1-2 min; produces Figure 25.
python3 sensitivity_robustness.py

# 20. (v20 addition) Domain-matched validation on real CIC-IDS2017 network
#     traffic (Section 4.2). Requires the raw CIC-IDS2017 archive
#     downloaded separately into cicids2017/raw/ (not redistributed here
#     for size reasons -- see Appendix A); the outputs of this exact
#     pipeline are already included in data/cicids2017/.
cd cicids2017
python3 01_extract_pool.py        # clean raw CSVs, build per-class candidate pools
# 20b. Empirical, pre-registered screening of every candidate attack type
#      against each mechanism's geometric definition
python3 02_characterize.py
# 20c. Build the 3 derived test sets (N=8,000 each)
python3 03_build_datasets.py
# 20d. Run naive / hand-informed / radial-oracle pipelines, 20 resamples each
#      (resumable in batches: python3 04_run_pipeline.py <start_seed> <end_seed>)
python3 04_run_pipeline.py 0 20
# 20e. Produce Figures 4-6
python3 05_visualize.py
cd ..
```

Each script prints its own results to stdout and writes a JSON/CSV file
to `data/`, in addition to any figures written to `figures/`. All
random seeds are fixed and printed in each script; rerunning should
reproduce every number in the paper exactly (modulo library-version
differences in scikit-learn/scipy).

Figure numbers in `figures/` (fig01-fig26) match the order figures
actually appear in `paper/BlindSpots_Neurocomputing_v21_XRefAudited.docx`:
Figures 1-3 (Section 3, theory: sample-complexity, eigenvalue spectrum,
correlation matrix), Figures 4-6 (Section 4.2, real CIC-IDS2017
domain-matched validation — moved earlier than in older package
revisions), Figure 7 (Section 4.3, Wine/Breast Cancer/Digits real-data
check), Figures 8-10 (Section 4.4, dimensionality/parameter sweeps),
Figure 11 (Section 5.2, Sparse PCA/Kernel PCA/t-SNE), Figure 12
(Section 5.2, blind Spectral Clustering/DBSCAN comparison), Figure 13
(Section 5.2, DEC/IDEC purpose-built deep-clustering comparison — no
longer numbered "7b"; see below), Figures 14-22 (Section 5.3/6, ICS
routing, distance/relational/scale mechanism panels, robustness, and
sensitivity results), Figures 23-24 (Section 6.3, interaction-ablation
results), Figure 25 (Section 6.3, sensitivity/breakdown-point analysis),
Figure 26 (Section 6.5, method x mechanism recovery summary matrix).

## What is genuinely new vs. reused

| Component | Status |
|---|---|
| `dataset_lib.py`, `generate_hard_dataset.py`, `analyze_pipelines.py`, `make_figures.py` | Unchanged core logic (paths repointed only) |
| `interaction_robustness_study.py` | Same logic, seed count raised 20 → 80 |
| `theory_variance_budget.py`, `scaling_experiments.py`, `real_data_validation.py`, `stronger_baselines.py`, `effect_size_power_stats.py`, `metrics_and_anticorrelation.py`, `auto_diagnostic_pipeline.py`, `make_figures_v2.py` | Added to extend the original empirical demonstration with theory, generalization checks, and statistical rigor |
| `ics_kurtosis_screening.py` | ICS-based pairwise kurtosis-anomaly screening, N=80 robustness, negative control against mechanism C |
| `ics_kurtosis_parameter_sweep.py` | Re-runs ICS routing across the same 30-cell grid used for Theorem 1 (450 runs); confirms 100% routing accuracy in every cell, not only the default configuration; produces Figure 15 |
| `interaction_robustness_multiconfig.py` | v6 addition: re-runs the Section 6.3 interaction ablation at 3 additional parameter-grid configurations (N=80 seeds each); produces Figure 23 |
| `deep_clustering_baseline.py` | Autoencoder + K-Means baseline; minimal and wide architectures at N=80 seeds each; reports a bimodal, seed-dependent instability that a single seed=42 report would have missed, and shows the instability is not resolved by a wider network |
| `radial_centroid_baseline.py` | Radial-distance-from-centroid oracle baseline for mechanisms B/C, N=80 seeds; found that a simple Euclidean scalar recovers both mechanisms comparably to the hand-informed pipeline, correcting the paper's "invisible to Euclidean distance" language throughout |
| `make_figures_v4_radial_update.py` | Regenerates Figures 9 and 17 only, to include the radial-centroid baseline |
| `DEC_IDEC_Colab_Baseline.ipynb` | Purpose-built deep-clustering comparison (DEC, IDEC), N=80 seeds each, run by the user on Google Colab (GPU) and integrated here; makes mechanism A recovery more reliable than the generic autoencoder baseline but reduces mechanism B recovery to 0 of 80 seeds; underlying data is `dec_idec_results.json`/`dec_idec_multiseed_raw.csv`, producing **Figure 13** in the final docx, embedded directly as `figures/fig13_dec_idec_comparison.png` and now correctly referenced (filename, figure number, and caption) by `generate_paper.js`. |
| `generate_paper.js` | Builds the manuscript, including Section 5.5 and Figure 14 (the ICS finding) |
| `interaction_robustness_multiconfig.py` | v6 addition: re-runs the Section 6.3 interaction ablation at 3 additional parameter-grid configurations (N=80 seeds each) to test whether the null-interaction finding generalizes beyond the single originally-tested configuration; it does (null holds at all 4 points tested) |
| `sample_complexity_simulation.py` | v18 addition, reviewer Priority 1: empirically separates two sample-complexity rates conflated by the "Dimension-Free" framing -- finds n needed to control the entrywise quantity (eps_hat) fits log(d) better, while n needed to control the spectral-norm quantity that actually enters Theorem 1 (\|\|E_hat\|\|_2) fits d\*log(d) better, matching Theorem 2's own stated regime hypothesis; produces Figure 1, now written into Section 3 (v21) |
| `baselines_comparison.py` | v18 addition, reviewer Priority 3: grid-searched Spectral Clustering and DBSCAN vs. naive KMeans, applied blindly (naive PCA-top-2 front end, no oracle feature selection) on the synthetic benchmark and on Wine/Breast Cancer/Digits; found DBSCAN blindly recovers mechanism B (relational) at ARI=0.72 vs. KMeans's 0.06, a materially weaker "no generic method sees it" story than previously reported for that mechanism; mechanism C (scale) stays invisible to all three blindly; produces Figure 12, now written into Section 5.3 with Table 2 (v21) |
| `sensitivity_robustness.py` | v18 addition, reviewer Priority 6: Cholesky/shared-latent-factor cross-mechanism correlation injection at increasing strength, re-running the hand-informed detectors to find each mechanism's breakdown point; complements (does not replace) the i.i.d.-mechanism TOST interaction test; breakdown points found at c=0.60 (A), c=0.40 (B), c=0.10 (C) -- mechanism C is far more fragile than A or B; produces Figure 25, now written into Section 6.3 (v21) |

## Package audit (v21)

A full audit of this package (code, data, figures, and the paper's own
internal consistency) found and fixed the following, beyond the
Figure 1/12/25 integration described above and in `CHANGELOG_v20_to_v21.md`:

- **Portability bug, 22 scripts + `generate_paper.js`:** every top-level
  script in `code/` except `sample_complexity_simulation.py`,
  `baselines_comparison.py`, `sensitivity_robustness.py`, and
  `dataset_lib.py` had `DATA_DIR`/`FIG_DIR` hardcoded to an absolute
  sandbox path (`/home/claude/revision/...`, one script used
  `/home/claude/extracted/...`) that does not exist outside this
  package's original build environment; running any of them would have
  failed with `FileNotFoundError` for anyone else. `generate_paper.js`
  had the same bug for its figures directory and its output path (and
  its output path additionally pointed at a filename that doesn't match
  this package's versioned naming). All 22 scripts and `generate_paper.js`
  now resolve paths relative to their own file location
  (`BASE = os.path.dirname(os.path.abspath(__file__))` in Python,
  `__dirname` in Node), matching the pattern already used correctly by
  the `code/cicids2017/` scripts. `deep_clustering_baseline.py` had a
  separate, milder bug (`DATA_DIR = "data"` instead of `"../data"`,
  wrong relative depth) that would also have failed under this README's
  own `cd code` reproduction instructions; also fixed.
- **`data/interaction_robustness_raw.csv` was 0 bytes** (the only
  zero-byte or corrupt file found in the package; all 26 figure PNGs
  validated as intact). This file is read by
  `effect_size_power_stats.py` and `make_figures_v3.py`. Fixed by
  re-running `interaction_robustness_study.py`; the regenerated
  `interaction_robustness_results.json` and `multiseed_robustness_raw.csv`
  are byte-identical to the versions already in this package, confirming
  the benchmark numbers are exactly reproducible and that no other file
  needed to change.
- **`generate_paper.js` was significantly out of sync with the current
  paper**, beyond the path bug above: its figure references (`Img(...)`
  calls) used a numbering scheme that predated a figure-renumbering pass
  (see the v17→v18 changes described above), with no code path for
  Figures 18-25 at all — and, once that was noticed, a deeper problem
  underneath it: the script's actual prose (every section, both new
  theorems, Section 4.2b, Table 2, Section 5.6, Section 6.6, and
  Appendix B) was still frozen at roughly the pre-v17 manuscript, so
  fixing only the figure references would not have made it able to
  rebuild the current paper from scratch as README step 15 describes.
  Both layers are now fixed: the script was regenerated section-by-section
  directly from `Multi_Mechanism_Blind_Spots_v21_FINAL.docx`'s own
  paragraph/table/figure content (title through References, including
  both appendices). A first round of spot-checking (round-tripping both
  documents through `pandoc` and eyeballing the diff) looked clean, but
  a full, non-sampled diff of every line turned up three more real gaps
  that the spot-check had missed: (1) each figure's image is followed in
  the source by a short, plain "Figure N" label paragraph *and then* the
  full italic caption as a separate paragraph — a genuine duplication in
  the source docx, reproduced faithfully rather than "cleaned up"; (2) of
  the document's 12 tables, 9 use a plain header (a bottom border, no
  bold) while 3 — the CIC-IDS2017 domain-validation subtables — use a
  bold, shaded header; this was verified by inspecting each table's
  actual `w:rPr`/`w:shd` XML individually rather than assuming one style
  for all 12, and the script now reproduces that split; (3) the initial
  text-extraction pass had normalized non-breaking spaces (`\u00a0`, used
  throughout for things like "c = 0" and "vs. 0.06") to regular spaces,
  which is fixed. After all three fixes, a full re-diff shows every
  remaining line-level difference is the already-diagnosed `pandoc`
  heading-detection quirk (and the literal-text period-escaping that
  follows from it) or a table's border-dash length changing with column
  width — normalizing both away, the two documents' word counts match
  exactly (15,942 = 15,942). The heading-outline-level gap noted earlier
  is also resolved: the first fix attempt added a `paragraphStyles` array
  that duplicated (rather than overrode) the `docx` library's own
  auto-generated Heading1/Heading2 style entries, so the first,
  outline-less copy of each ID silently won; the working fix instead
  configures the library's own default-style generator directly
  (`styles.default.heading1` / `heading2`, which is the API this version
  of the `docx` package actually reads), and `styles.xml` now carries a
  single, correct `w:outlineLvl` entry per heading level with no
  duplicate style IDs, confirmed by inspecting the generated `styles.xml`
  directly. `pandoc`'s own docx reader still doesn't surface these as
  `#`-headings when round-tripped to Markdown; that's confirmed (via an
  isolated, from-scratch minimal test with an identical stylesheet
  showing the same non-detection) to be a `pandoc`-specific heuristic
  unrelated to `w:outlineLvl`, not a defect in the file — Word keys its
  Navigation Pane and Table-of-Contents field off `w:outlineLvl` and
  `w:pStyle`, both of which are now correctly set. Output file:
  `Multi_Mechanism_Blind_Spots_v21_REGENERATED.docx`.
- **Dangling references to two changelog files** (`CHANGELOG_v17_to_v18.md`,
  `CHANGELOG_v8_to_v11_FINAL.md`) that this README referenced but that
  were not actually included in the package. Rather than fabricate their
  detailed contents, the dangling references were removed from the
  "Submission file" note above; the one-line summaries that were already
  written there are kept, since they describe changes already reflected
  in the current paper text itself.

## Known gaps (see paper Section 8-9)

- No formal random-matrix-theory proof of eigenvalue spread (Proposition 1
  is a verified structural argument for this generator family, not a
  general theorem).
- A domain-matched real dataset is now included for the network-security
  domain (CIC-IDS2017, Section 4.2). Analogous domain-matched real-data
  validation for other domains (banking, clinical) with plausible
  relational/ratio or density/scale subgroups remains open.
- No UMAP comparison — not completed in this iteration due to
  environment/time constraints.
- No verified 2025-2026 literature update for Related Work — flagged as
  an open item rather than fabricated.
- `code/generate_paper.js`'s internal section-number labels (e.g. "Sec.
  5.5", "Sec. 6.6") lag one level of consolidation behind the submitted
  docx's actual numbering (e.g. "Section 5.3", "Section 6.5") in several
  older table rows — a pre-existing desync from an earlier manual
  consolidation pass that folded several `generate_paper.js` H2 headings
  into single numbered sections in the docx. The docx (`paper/`) is the
  source of truth for section numbers; `generate_paper.js` is kept
  functional and its *content* stays in sync, but regenerating a docx
  from it will not exactly reproduce the submitted numbering without a
  full re-consolidation pass, which has not been done.
- A purpose-built deep-clustering comparison (DEC/IDEC, Section 5.2,
  v22) is now included, but used a single fixed architecture and
  hyperparameter setting taken from the original DEC/IDEC papers; no
  systematic hyperparameter or architecture sweep for DEC/IDEC
  specifically was performed, unlike Section 4.3's sweep for the naive
  pipeline. A multi-task or disentangled variant aimed at recovering
  both mechanisms jointly (Section 9) was not attempted.
- Section 5.5's ICS finding is an empirical extension of established
  elliptical-mixture ICS theory to a non-elliptical (ratio-line) case,
  not a new closed-form theorem; automatic within-pair thresholding for
  the ICS-detected relational signal remains unsolved (oracle ceiling
  0.938 vs. best practical automatic result -0.074).

## Reviewer-response tracking (as of v21)

A full-professor reviewer sent a harsh but substantive rejection of v18,
organized around 6 priorities (source docs: `Paper_Revision_Action_Plan`,
`Paper_Revision_Methodology`, `Paper_Revision_Code_Guide` — kept outside
this package, alongside the paper, code zip, and this reviewer feedback,
as the three items the author is working from). Status:

| # | Priority | Status |
|---|---|---|
| 1 | Drop "Dimension-Free" framing; reframe Theorem 1 as a sample-complexity / stability result | **Done (v21)** — `sample_complexity_simulation.py` (Figure 1) is now written into Section 3, immediately after Theorem 2: the framing is defensible for the entrywise quantity (Corollary 1, ~log d samples) but not, without qualification, for the spectral-norm quantity that actually enters Theorem 1's bound (~d log d samples, matching Theorem 2's own stated regime). The Abstract and Section 3 now state this qualification explicitly; the paper's title is unchanged (a title change was judged out of scope for this revision — see `CHANGELOG_v20_to_v21.md`). |
| 2 | Soften "standard pipelines fail on real data" narrative; reposition as a representation-learning preprocessing step for high-dimensional/manifold data | **Partially done** — Abstract and Section 5.3 edited to fold in the Priority-3 finding (DBSCAN's blind partial recovery of mechanism B), narrowing the "only ICS routes correctly" claim. A new Related Work paragraph (plus 2 new references: Johnstone 2001, Baik/Ben Arous/Péché 2005) was also added, placing Theorem 1 relative to the spiked-covariance/BBP phase-transition literature -- a related-work gap found during revision discussion, not originally one of the reviewer's 6 priorities, but addressed for the same reason (an expert reviewer would likely have raised it). The broader reframing toward "representation-learning preprocessing for high-dimensional/manifold data" (a bigger narrative shift, not just these edits) is not yet done. |
| 3 | Compare against Spectral Clustering / DBSCAN / Kernel K-Means instead of K-Means on concentric/linear structures | **Done (v21)** — `baselines_comparison.py` (Figure 12), now fully written into Section 5.3 with Table 2, plus a real-dataset (Wine/Breast Cancer/Digits) generalization check. DBSCAN applied blindly (no oracle feature selection) on the naive PCA-top-2 front end recovers mechanism B (relational) at ARI=0.72, materially higher than naive KMeans (ARI=0.06); Spectral Clustering does not (ARI=0.02). This measurably weakens the "no generic front end sees mechanism B" framing, and the Abstract, Section 5.3, and Section 6.6 now state this honestly rather than only citing the new baseline. Mechanism C (scale) remains invisible to all three methods blindly (ARI ~0). |
| 4 | Compare against DEC/IDEC (not a 2-neuron autoencoder); report accuracy vs. inference-time/no-GPU trade-off | **Done (v22)** — `DEC_IDEC_Colab_Baseline.ipynb` was run on Google Colab (GPU runtime), N=80 seeds each at `latent_dim=10`; integrated into Section 5.2, Table 5, and Figure 13 (`CHANGELOG_v21_to_v22.md`). Result: DEC/IDEC make mechanism A recovery substantially more reliable (55–59% vs. 20% of seeds) but reduce mechanism B recovery to 0 of 80 seeds (from the generic autoencoder's 40–49%) — sharpening, not resolving, the paper's central collapse finding. The inference-time/no-GPU trade-off half of this priority was not separately benchmarked. |
| 5 | Replace GMM final-clustering step with eigengap-based automatic thresholding + HDBSCAN | **Done (v23)** — `code/ICS_Eigengap_HDBSCAN.py` replaces the two GMM-based steps in `ics_kurtosis_screening.py`: (i) an eigengap rule on the sorted 7-pair kurtosis-deviation spectrum now decides routing confidence automatically instead of a bare argmax; (ii) HDBSCAN, applied to the same log-projection space the GMM used, replaces the fixed-prior GMM for within-pair minority detection, with `min_cluster_size` set from a generic 3% of n rather than the true 12% relational fraction. Run at N=80 seeds (1000-1079): routing accuracy 100%, routing confident on 100% of seeds, mean ARI 0.831 (SD 0.059) vs. an oracle ceiling of 0.938 (SD 0.005) — HDBSCAN recovers most, not all, of the achievable signal without any minority-fraction assumption. Negative-control run on the true scale pair (mechanism C) gives mean ARI 0.072, consistent with mechanism C not being this signal's target. `data/ics_eigengap_hdbscan_results.json`, `data/ics_eigengap_hdbscan_raw.csv`, Figure 27. |
| 6 | Replace the i.i.d.-mechanism TOST interaction test with a Cholesky-injected-correlation sensitivity/breakdown-point analysis | **Done (v21)** — `sensitivity_robustness.py` (Figure 25), now fully written into Section 6.3 alongside the original interaction test (as a complementary check, not a replacement) with a summary table. Breakdown points (mean ARI < 50% of c=0 baseline) differ sharply by mechanism: A (distance) at c=0.60, B (relational) at c=0.40, C (scale) at c=0.10 — mechanism C is far more fragile to cross-mechanism contamination than A or B. Limitations now flags this analysis's N=5 seeds as notably lower-powered than this paper's N=80 standard elsewhere. |

### Dependency notes (Priorities 4 and 5)

Two components in the reviewer-response table require dependencies
beyond the base `requirements.txt`:

- **Priority 4 (DEC/IDEC via PyTorch)** requires `torch` and a GPU.
  `DEC_IDEC_Colab_Baseline.ipynb` is written to run standalone on Google
  Colab (GPU runtime); its outputs are integrated into Section 5.2,
  Table 5, and Figure 13. See `CHANGELOG_v21_to_v22.md`.
- **Priority 5 (`ICS_Eigengap_HDBSCAN.py`)** requires the `hdbscan`
  package (`pip install hdbscan`). The eigengap rank-selection step
  uses only numpy; only the final clustering call depends on `hdbscan`.
  The script is written to run standalone on Google Colab.
- Priorities 1, 3, and 6 depend only on numpy/pandas/scipy/scikit-learn/
  matplotlib, all listed in `requirements.txt`.
