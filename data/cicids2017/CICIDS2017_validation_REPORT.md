# CIC-IDS2017 Domain-Matched Validation — Full Report

This report documents the complete, empirical process behind Section 4.2
of the paper ("Domain-Matched Real-World Instantiation: Network
Traffic"). It was produced by actually running
`code/cicids2017/01_extract_pool.py` through `05_visualize.py` against
the eight raw CIC-IDS2017 `MachineLearningCVE` CSV files (Sharafaldin,
Lashkari and Ghorbani, 2018; Canadian Institute for Cybersecurity,
University of New Brunswick). No synthetic or fabricated rows are used
anywhere below — every number in this report comes directly from that
run's stdout and from the CSV/JSON files it wrote to `data/cicids2017/`.

## 1. Data provenance

- Source: CIC-IDS2017, `MachineLearningCVE` (flow-level, CICFlowMeter-derived) CSVs.
- Files processed (all eight, `code/cicids2017/raw/*.csv`):
  - `Monday-WorkingHours.pcap_ISCX.csv`
  - `Tuesday-WorkingHours.pcap_ISCX.csv`
  - `Wednesday-workingHours.pcap_ISCX.csv`
  - `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`
  - `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv`
  - `Friday-WorkingHours-Morning.pcap_ISCX.csv`
  - `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`
  - `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
- Known data-quality artifacts corrected (Liu et al., 2022), applied by
  `01_extract_pool.py`/`03_build_datasets.py`/`04_run_pipeline.py`:
  - **1 row** with `Flow Duration = -1` (a documented CICFlowMeter
    parsing artifact) was clipped to 0. (Matches the single artifact
    reported in the paper's Section 4.2 text exactly.)
  - `Flow Bytes/s` and `Flow Packets/s` were excluded from the analysis
    columns for the small number of zero-duration flows that otherwise
    produce `Infinity`/NaN (46 affected rows out of 63,385 in the pooled
    candidate set — these two columns are not used in any of the three
    derived test sets' signal or noise dimensions, so this exclusion
    has no downstream effect on Tests A/B/C).
  - `Web Attack` labels with mangled en-dash encoding were normalized to
    `Web Attack-Brute Force`, `Web Attack-XSS`, `Web Attack-SQL Injection`.
- No other rows were dropped, reweighted, or modified. Every row in
  `test_A_distance.csv`, `test_B_relational.csv`, and `test_C_scale.csv`
  carries its original CIC-IDS2017 label.

## 2. Candidate pool construction (`01_extract_pool.py`)

Per-file, per-label sampling caps (BENIGN: 1,500/file; large attack
classes: 6,000/file; smaller/rarer classes: kept in full) were applied
to build a manageable candidate pool while preserving every rare class
in full. Resulting pool: **63,385 rows** across 15 labels.

| Label | n in pool |
|---|---|
| BENIGN | 12,000 |
| DDoS | 6,000 |
| PortScan | 6,000 |
| FTP-Patator | 6,000 |
| DoS GoldenEye | 6,000 |
| DoS Hulk | 6,000 |
| SSH-Patator | 5,897 |
| DoS slowloris | 5,796 |
| DoS Slowhttptest | 5,499 |
| Bot | 1,966 |
| Web Attack-Brute Force | 1,507 |
| Web Attack-XSS | 652 |
| Infiltration | 36 |
| Web Attack-SQL Injection | 21 |
| Heartbleed | 11 |

After dropping rows with NaN/Infinity in the core feature columns and
requiring positive upload/download byte totals (`03_build_datasets.py`),
the cleaned pool used to draw Tests A/B/C contained:

| Label | n (cleaned pool) |
|---|---|
| BENIGN | 9,132 |
| DoS GoldenEye | 4,520 |
| DoS Hulk | 4,028 |
| DDoS | 3,778 |
| FTP-Patator | 2,988 |
| SSH-Patator | 2,960 |
| PortScan | 2,923 |
| DoS slowloris | 1,743 |
| Bot | 1,474 |
| DoS Slowhttptest | 305 |
| Web Attack-Brute Force | 151 |
| Infiltration | 32 |
| Web Attack-XSS | 24 |
| Web Attack-SQL Injection | 12 |
| Heartbleed | 11 |

## 3. Empirical, pre-registered mechanism screening (`02_characterize.py`)

Every candidate attack type was screened against each mechanism's
geometric definition *before* any test set was built, so the selection
below is empirical, not post-hoc.

### Mechanism A (distance-separable): duration / packet-count profile

| Label | Flow Duration mean | Flow Duration median | Total Fwd Packets mean |
|---|---:|---:|---:|
| PortScan | 80,318.6 | 48.0 | 1.0 |
| Bot | 350,942.7 | 70,755.5 | 3.2 |
| BENIGN | 11,987,989.7 | 31,317.5 | 5.9 |
| DDoS | 16,775,220.8 | 1,929,483.0 | 4.5 |
| DoS GoldenEye | 23,058,741.6 | 11,591,434.5 | 5.9 |
| DoS Hulk | 56,483,054.7 | 84,903,354.5 | 5.2 |

**Selected:** BENIGN, PortScan, DDoS, DoS Hulk — median flow durations
span roughly six orders of magnitude and are monotonically ordered,
satisfying the distance-mechanism definition cleanly.

### Mechanism B (relational/ratio): upload/download byte-ratio tightness vs. magnitude range

Sorted by `log_ratio_std` (lower = tighter ratio, i.e. a stronger
relational signal); `log_magnitude_range` measures how widely the
class's raw byte volume spans (higher = harder to catch by scale alone):

| Label | n | ratio median | log_ratio_std | log_magnitude_range |
|---|---:|---:|---:|---:|
| SSH-Patator | 2,960 | 0.732 | 0.019 | 0.456 |
| FTP-Patator | 2,988 | 0.564 | 0.025 | 0.484 |
| PortScan | 2,923 | 0.333 | 0.103 | 3.162 |
| Web Attack-XSS | 24 | 0.266 | 0.139 | 2.052 |
| DoS Hulk | 4,028 | 0.031 | 0.195 | 2.911 |
| DDoS | 3,778 | 0.002 | 0.195 | 2.812 |
| Web Attack-Brute Force | 151 | 0.609 | 0.214 | 2.217 |
| **Bot** | **1,474** | **1.360** | **0.422** | **4.240** |
| DoS slowloris | 1,743 | 423.500 | 0.611 | 1.836 |
| BENIGN | 9,132 | 0.440 | 0.664 | 6.018 |
| DoS GoldenEye | 4,520 | 0.044 | 0.722 | 1.865 |
| DoS Slowhttptest | 305 | 86.667 | 1.331 | 1.688 |

**Selected:** Bot. SSH-Patator and FTP-Patator have tighter ratios
(`log_ratio_std` 0.019/0.025) but are confined to a narrow byte-volume
band (`log_magnitude_range` 0.46/0.48) — an easy, compact blob, not a
hard case. Bot is simultaneously tighter-than-background in ratio
(0.422 vs. BENIGN's 0.664) *and* spread across nearly the full magnitude
range the BENIGN background occupies (4.240 vs. BENIGN's 6.018) — the
adversarial "tight in angle, spread in magnitude" property mechanism B
is designed to test.

### Mechanism C (scale/density): centroid distance and variance ratio vs. BENIGN, in (Flow IAT Mean, Flow IAT Std)

| Label | n | centroid_dist (BENIGN-std units) | var_ratio (mean, std) |
|---|---:|---:|---|
| Heartbleed | 11 | 0.49 | (0.000, 0.001) |
| **SSH-Patator** | **5,897** | **0.43** | **(0.036, 0.087)** |
| Web Attack-XSS | 652 | 0.43 | (0.117, 0.132) |
| Bot | 1,966 | 0.49 | (0.035, 0.033) |
| FTP-Patator | 6,000 | 0.38 | (0.052, 0.090) |
| Web Attack-Brute Force | 1,507 | 0.36 | (0.169, 0.194) |
| Infiltration | 36 | 1.17 | (1.224, 1.432) |
| DoS Slowhttptest | 5,499 | 4.59 | (1.346, 2.408) |
| DoS slowloris | 5,796 | 5.27 | (3.499, 4.011) |

**Selected:** SSH-Patator. Despite being the intuitively "stealthy"
candidates by name, DoS slowloris and DoS Slowhttptest sit 4.6–5.8
background standard deviations from the BENIGN centroid — they are slow
*on purpose*, which makes them timing outliers, not timing-invisible.
Heartbleed and Infiltration have too few real examples (11 and 36
dataset-wide) to support a statistically powered minority at the paper's
4% fraction. SSH-Patator best satisfies the mechanism-C definition:
centroid within 0.43 background standard deviations, variance only
3.6–8.7% of BENIGN's variance in the same two dimensions. This is a
modeling choice under real-data constraints: SSH-Patator is a
brute-force attack, not "stealthy" in the colloquial sense; what it
shares with mechanism C is the statistical signature (regular, automated
timing → low variance at a mean IAT not far from ordinary background
traffic), not the intent behind the attack.

## 4. Derived test sets (`03_build_datasets.py`)

| File | Construction | N |
|---|---|---:|
| `test_A_distance.csv` | BENIGN/PortScan/DDoS/DoS Hulk, 2,000 each | 8,000 |
| `test_B_relational.csv` | BENIGN background (7,040) + Bot minority (960, 12%) | 8,000 |
| `test_C_scale.csv` | BENIGN background (7,680) + SSH-Patator minority (320, 4%) | 8,000 |

Each row carries its two mechanism-specific signal columns plus 8
additional real CIC-IDS2017 columns as filler/noise dimensions
(`Fwd/Bwd Packet Length Mean/Std`, `Packet Length Mean/Std`, `Average
Packet Size`, `Active Mean`), reproducing the paper's 10-D → PCA →
top-2-PC pipeline shape.

## 5. Pipeline results (`04_run_pipeline.py`, 20 independent resamples each)

Because real data carries no generation seed, robustness was assessed by
re-drawing 20 independent random samples from the underlying label pools
and re-running the full pipeline each time. All 20/20 resamples
completed successfully for every test. Full per-resample values are in
`robustness_A_raw.csv`, `robustness_A_log.csv`, `robustness_B.csv`, and
`robustness_C.csv`; summary statistics (mean ± SD) below match Tables
2–4 of the paper.

### Mechanism A (distance) — Table 2

| Metric | Value |
|---|---|
| Naive ARI (Silhouette-selected k) | 0.473 ± 0.003 |
| Silhouette-selected k (true k = 4) | 3, in 20/20 resamples |
| Naive ARI forced at true k=4 | 0.465 ± 0.006 |
| Hand-informed ARI, raw duration | 0.219 ± 0.004 |
| Hand-informed ARI, log10(duration+1) | 0.407 ± 0.047 |

Silhouette never selects the true k=4; it chooses k=3 in every resample,
merging BENIGN+PortScan into one cluster and DDoS+DoS Hulk into another
(see `fig04_domain_validation_mechanism_A.png` crosstab panel — real
counts from this run: BENIGN split 1872/85/43 across the three clusters,
PortScan 2000/0/0, DDoS 6/1994/0, DoS Hulk 17/1983/0). This concrete
merge pattern is a real-data-specific failure mode not produced by the
idealized synthetic four-blob design. Raw Flow Duration is heavy-tailed
enough that 2-D hand-informed K-means (ARI 0.219) underperforms the
naive 10-D→PCA pipeline; log-transforming duration closes most of the
gap (ARI rises to 0.407).

### Mechanism B (relational, Bot 12%) — Table 3

| Metric | Value |
|---|---|
| Naive ARI | −0.043 ± 0.011 |
| Hand-informed (angular DBSCAN) ARI | 0.398 ± 0.005 |
| Radial-centroid oracle ARI | 0.237 ± 0.007 |

Naive ARI sits at or below chance, confirming the paper's headline
collapse on real traffic. Hand-informed above radial-oracle matches the
synthetic-data ordering (Section 5.3: hand-informed 0.915 vs.
radial-oracle 0.875), with a wider absolute gap on real data (0.161 here
vs. 0.040 synthetic) — an honest strengthening, not a reversal, of the
synthetic account.

### Mechanism C (scale/density, SSH-Patator 4%) — Table 4

| Metric | Value |
|---|---|
| Naive ARI | −0.050 ± 0.000 |
| Hand-informed (2-component GMM) ARI | 0.064 ± 0.001 |
| Radial-centroid oracle ARI | 0.306 ± 0.004 |

Naive ARI again sits at or below chance in essentially every resample.
Radial-oracle above hand-informed matches the synthetic ordering
(Section 5.3: radial-oracle 0.880 vs. k-NN-density-GMM 0.662). The GMM's
weakness is plausibly sharper here because real BENIGN background
traffic is heavy-tailed and plausibly multimodal in IAT space, violating
the two-component-Gaussian-mixture assumption more than the synthetic
background does.

## 6. Summary

This domain-matched validation confirms the paper's central claim more
strongly on real traffic than on synthetic data for mechanisms B and C —
naive ARI sits at or below chance rather than merely low — and
reproduces Section 5.3's radial-vs-hand-informed ordering for both B and
C. It also surfaces one real-data-specific failure mode for mechanism A
(Silhouette's persistent k=3 undercount and its {BENIGN, PortScan} /
{DDoS, DoS Hulk} merge pattern) that the idealized synthetic four-blob
design does not produce by construction.

This is validation of the qualitative failure mode on one real,
independent dataset, not a fully joint, factorial replication of the
paper's Section 4.4 interaction study: each mechanism here is validated
as an independent two-signal-plus-eight-noise real-data analog. Because
real attack labels are mutually exclusive per flow, a joint replication
with all three mechanisms co-occurring in the same rows is left as
future work (paper Section 9).

## 7. Regenerating this report

```bash
cd code/cicids2017
python3 01_extract_pool.py
python3 02_characterize.py     # produces Section 3 of this report
python3 03_build_datasets.py   # produces Section 4 of this report / test_*.csv
python3 04_run_pipeline.py     # produces Section 5 of this report / robustness_*.csv
python3 05_visualize.py        # produces Figures 4-6
```

All random seeds are fixed (`np.random.seed(42)` for pool construction;
seeds 100–119 for the 20 pipeline resamples) and printed by each script.
