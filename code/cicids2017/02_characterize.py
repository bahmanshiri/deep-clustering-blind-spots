import pandas as pd
import numpy as np
import os

pd.set_option('display.width', 160)
pd.set_option('display.max_columns', 30)

BASE = os.path.dirname(os.path.abspath(__file__))
pool = pd.read_pickle(os.path.join(BASE, "pool.pkl"))
print("Loaded pool:", pool.shape)

num_cols = [c for c in pool.columns if c not in ('Label', 'source_file')]
for c in num_cols:
    pool[c] = pd.to_numeric(pool[c], errors='coerce')
pool[num_cols] = pool[num_cols].replace([np.inf, -np.inf], np.nan)

print("\nNaN counts (top offenders):")
print(pool[num_cols].isna().sum().sort_values(ascending=False).head(8))

# drop rows with any NaN in the core columns we care about for this diagnostic
core = ['Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
        'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
        'Flow IAT Mean', 'Flow IAT Std', 'Fwd IAT Mean', 'Fwd IAT Std',
        'Bwd IAT Mean', 'Bwd IAT Std', 'Packet Length Mean', 'Packet Length Std',
        'Down/Up Ratio', 'Average Packet Size']
before = len(pool)
pool = pool.dropna(subset=core).reset_index(drop=True)
print(f"\nDropped {before-len(pool)} rows with NaN in core cols -> {len(pool)} remain")

print("\n" + "="*100)
print("MECHANISM A candidates: duration/packet-count profile per label")
print("="*100)
a_candidates = ['BENIGN', 'PortScan', 'DDoS', 'DoS Hulk', 'DoS GoldenEye', 'Bot']
g = pool[pool.Label.isin(a_candidates)].groupby('Label')[['Flow Duration', 'Total Fwd Packets', 'Total Backward Packets']]
print(g.agg(['mean', 'std', 'median']).round(1))

print("\n" + "="*100)
print("MECHANISM B candidates: Fwd/Bwd byte ratio tightness (want LOW within-group CV, WIDE magnitude range)")
print("="*100)
pool['upload_bytes'] = pool['Total Length of Fwd Packets']
pool['download_bytes'] = pool['Total Length of Bwd Packets']
# avoid div by zero
valid = (pool['upload_bytes'] > 0) & (pool['download_bytes'] > 0)
pb = pool[valid].copy()
pb['ud_ratio'] = pb['upload_bytes'] / pb['download_bytes']
pb['log_magnitude'] = np.log10(pb['upload_bytes'] + pb['download_bytes'])

b_candidates = ['BENIGN', 'FTP-Patator', 'SSH-Patator', 'Bot', 'Web Attack-Brute Force',
                'Web Attack-XSS', 'PortScan', 'DDoS', 'DoS Hulk', 'DoS GoldenEye',
                'DoS slowloris', 'DoS Slowhttptest']
rows = []
for lab in b_candidates:
    sub = pb[pb.Label == lab]
    if len(sub) < 5:
        continue
    r = sub['ud_ratio']
    logr = np.log10(r.replace(0, np.nan).dropna())
    rows.append({
        'Label': lab, 'n': len(sub),
        'ratio_median': r.median(), 'ratio_mean': r.mean(),
        'log_ratio_std': logr.std(),  # CV-like measure robust to scale, in log space
        'log_magnitude_min': sub['log_magnitude'].min(),
        'log_magnitude_max': sub['log_magnitude'].max(),
        'log_magnitude_range': sub['log_magnitude'].max() - sub['log_magnitude'].min(),
    })
bdf = pd.DataFrame(rows).sort_values('log_ratio_std')
print(bdf.round(3).to_string(index=False))

print("\n" + "="*100)
print("MECHANISM C candidates: variance/centroid check vs BENIGN background, several feature pairs")
print("="*100)
c_candidates = ['DoS slowloris', 'DoS Slowhttptest', 'Infiltration', 'Heartbleed', 'Bot',
                'Web Attack-Brute Force', 'Web Attack-XSS', 'FTP-Patator', 'SSH-Patator']
feature_pairs = [
    ('Flow IAT Mean', 'Flow IAT Std'),
    ('Fwd IAT Mean', 'Fwd IAT Std'),
    ('Bwd IAT Mean', 'Bwd IAT Std'),
    ('Packet Length Mean', 'Packet Length Std'),
    ('Average Packet Size', 'Down/Up Ratio'),
]
benign = pool[pool.Label == 'BENIGN']
for f1, f2 in feature_pairs:
    print(f"\n--- pair: ({f1}, {f2}) ---")
    b_mean = benign[[f1, f2]].mean()
    b_std = benign[[f1, f2]].std()
    print(f"BENIGN centroid: {f1}={b_mean[f1]:.2f}, {f2}={b_mean[f2]:.2f}  |  spread(std): {f1}={b_std[f1]:.2f}, {f2}={b_std[f2]:.2f}")
    for lab in c_candidates:
        sub = pool[pool.Label == lab]
        if len(sub) < 10:
            continue
        s_mean = sub[[f1, f2]].mean()
        s_std = sub[[f1, f2]].std()
        # normalized distance between centroids (in units of BENIGN std)
        dist = np.sqrt(((s_mean - b_mean) / b_std.replace(0, np.nan))**2).sum()
        var_ratio_f1 = s_std[f1] / b_std[f1] if b_std[f1] > 0 else np.nan
        var_ratio_f2 = s_std[f2] / b_std[f2] if b_std[f2] > 0 else np.nan
        print(f"  {lab:28s} n={len(sub):5d}  centroid_dist(in benign-std units)={dist:6.2f}  var_ratio=({var_ratio_f1:.3f},{var_ratio_f2:.3f})")
