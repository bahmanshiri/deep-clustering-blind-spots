# Multi-Mechanism Blind Spots in PCA-Preprocessed K-Means

Official reproducibility package for the manuscript submitted to *Neurocomputing* (Elsevier).

## Overview

This repository provides the complete manuscript, source code, generated data, and figures for full reproduction of every result reported in the paper.

- **Manuscript:** `paper/BlindSpots_Neurocomputing_v21_XRefAudited.docx`
- **Figures:** 27 figures (`figures/fig01_...png`-`figures/fig27_...png`), numbered to match their order of appearance in the manuscript.

## Repository Structure

```
paper/            Submission manuscript (.docx)
code/             Python analysis/figure scripts + generate_paper.js (Node.js)
code/cicids2017/  CIC-IDS2017 domain-validation pipeline
data/             Generated CSV/JSON outputs (regeneratable from code/)
data/cicids2017/  CIC-IDS2017-derived datasets and validation report
figures/          All 27 manuscript figures (regeneratable from code/)
requirements.txt  Python dependencies
Dockerfile        Containerized environment
LICENSE           MIT (code)
```

## Requirements

- Python 3.10+
- Node.js (for `generate_paper.js`)
- Optional: PyTorch + GPU (DEC/IDEC baseline), `hdbscan` package (eigengap/HDBSCAN routing)

```bash
pip install -r requirements.txt --break-system-packages
```

## Reproducing Results

All scripts run from `code/` unless noted, write outputs to `data/` and `figures/`, and use fixed, printed random seeds for exact reproducibility.

```bash
cd code

# Core pipeline
python3 generate_hard_dataset.py
python3 analyze_pipelines.py
python3 interaction_robustness_study.py
python3 theory_variance_budget.py
python3 scaling_experiments.py
python3 real_data_validation.py
python3 stronger_baselines.py
python3 isomap_mds_baseline.py

# Deep-clustering baselines (requires PyTorch/GPU)
python3 deep_clustering_baseline.py
# code/DEC_IDEC_Colab_Baseline.ipynb  (run on Google Colab, GPU runtime)
python3 dec_idec_figure.py

# Statistics and diagnostics
python3 effect_size_power_stats.py
python3 metrics_and_anticorrelation.py
python3 auto_diagnostic_pipeline.py

# ICS routing and eigengap/HDBSCAN (requires `pip install hdbscan`)
python3 ics_kurtosis_screening.py
python3 ICS_Eigengap_HDBSCAN.py
python3 ics_kurtosis_parameter_sweep.py

# Figures
python3 make_figures_v3.py
python3 parameter_space_sweep.py
python3 make_figure_parameter_sweep.py
python3 radial_centroid_baseline.py
python3 make_figures_v4_radial_update.py

# Robustness / sensitivity
python3 dbscan_angular_multiseed_rerun.py 80
python3 interaction_robustness_multiconfig.py
python3 sample_complexity_simulation.py
python3 baselines_comparison.py
python3 sensitivity_robustness.py

# Rebuild manuscript from source
node generate_paper.js

# CIC-IDS2017 domain-matched validation (requires raw archive; see code/cicids2017/raw/)
cd cicids2017
python3 01_extract_pool.py
python3 02_characterize.py
python3 03_build_datasets.py
python3 04_run_pipeline.py 0 20
python3 05_visualize.py
```

## Citation

If you use this code or data, please cite the associated manuscript (citation details to be added upon publication).

## License

Code is released under the MIT License (see `LICENSE`).
