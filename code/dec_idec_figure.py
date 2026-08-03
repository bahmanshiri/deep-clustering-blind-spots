"""
dec_idec_figure.py
===================
Regenerates Figure 13 (Section 5.2) from the DEC/IDEC N=80-seed results
produced by DEC_IDEC_Colab_Baseline.ipynb (dec_idec_results.json /
dec_idec_multiseed_raw.csv).

Compares mean ARI per mechanism (A: distance, B: relational, C: scale/density)
across four deep-representation baselines:
  - Autoencoder (minimal arch.), N=80 seeds       <- Table 1, existing row
  - Autoencoder (wide arch.), N=80 seeds          <- Table 1, existing row
  - DEC (Xie et al., 2016), latent=10, N=80 seeds <- new, this package
  - IDEC (Guo et al., 2017), latent=10, N=80 seeds <- new, this package

The first two rows' means are taken directly from the paper's Table 1
(already-published numbers, reproduced here only for the comparison plot).
The DEC/IDEC numbers are recomputed directly from dec_idec_multiseed_raw.csv
so the figure never goes out of sync with the raw data.

Usage:
    python3 dec_idec_figure.py
    (run from anywhere; reads ../data/dec_idec_multiseed_raw.csv,
     writes ../figures/fig13_dec_idec_comparison.png)
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "..", "data")
FIG_DIR = os.path.join(BASE, "..", "figures")

RAW_CSV = os.path.join(DATA_DIR, "dec_idec_multiseed_raw.csv")
OUT_PNG = os.path.join(FIG_DIR, "fig13_dec_idec_comparison.png")

# Table 1 reference rows (already published; not recomputed here).
TABLE1_AUTOENCODER_MINIMAL = {"A": 0.12, "B": 0.32, "C": -0.00}
TABLE1_AUTOENCODER_WIDE = {"A": 0.16, "B": 0.19, "C": -0.00}


def load_dec_idec_means(raw_csv=RAW_CSV):
    df = pd.read_csv(raw_csv)
    means = {}
    for method in ["dec", "idec"]:
        d = df[df.method == method]
        means[method] = {
            "A": d.native_ari_a.mean(),
            "B": d.native_ari_b.mean(),
            "C": d.native_ari_c.mean(),
        }
    return means


def make_figure(means, out_png=OUT_PNG):
    methods = [
        "Autoencoder\n(minimal, N=80)",
        "Autoencoder\n(wide, N=80)",
        "DEC\n(latent=10, N=80)",
        "IDEC\n(latent=10, N=80)",
    ]
    mean_A = [TABLE1_AUTOENCODER_MINIMAL["A"], TABLE1_AUTOENCODER_WIDE["A"],
              means["dec"]["A"], means["idec"]["A"]]
    mean_B = [TABLE1_AUTOENCODER_MINIMAL["B"], TABLE1_AUTOENCODER_WIDE["B"],
              means["dec"]["B"], means["idec"]["B"]]
    mean_C = [TABLE1_AUTOENCODER_MINIMAL["C"], TABLE1_AUTOENCODER_WIDE["C"],
              means["dec"]["C"], means["idec"]["C"]]

    x = np.arange(len(methods))
    w = 0.25
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    b1 = ax.bar(x - w, mean_A, w, label="Mechanism A (distance)", color="#3B6EA5")
    b2 = ax.bar(x, mean_B, w, label="Mechanism B (relational)", color="#D98E04")
    b3 = ax.bar(x + w, mean_C, w, label="Mechanism C (scale/density)", color="#8A8A8A")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean ARI across N=80 seeds")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_title(
        "Deep-clustering baselines: capacity buys reliability on A, not visibility of B or C",
        fontsize=10.5,
    )
    ax.legend(fontsize=8, loc="upper right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.2f}",
                (bar.get_x() + bar.get_width() / 2, h),
                textcoords="offset points",
                xytext=(0, 2 if h >= 0 else -10),
                ha="center",
                fontsize=7,
            )

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    means = load_dec_idec_means()
    print("DEC/IDEC means recomputed from raw CSV:")
    for m, v in means.items():
        print(f"  {m}: {v}")
    make_figure(means)
