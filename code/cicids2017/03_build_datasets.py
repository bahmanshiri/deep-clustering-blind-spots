import pandas as pd
import numpy as np
import os

np.random.seed(42)
pd.set_option('display.width', 160)

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_OUT_DIR = os.path.normpath(os.path.join(BASE, "..", "..", "data", "cicids2017"))
os.makedirs(DATA_OUT_DIR, exist_ok=True)

pool = pd.read_pickle(os.path.join(BASE, "pool.pkl"))
num_cols = [c for c in pool.columns if c not in ('Label', 'source_file')]
for c in num_cols:
    pool[c] = pd.to_numeric(pool[c], errors='coerce')
pool[num_cols] = pool[num_cols].replace([np.inf, -np.inf], np.nan)

CORE = ['Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
        'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
        'Flow IAT Mean', 'Flow IAT Std', 'Fwd IAT Mean', 'Fwd IAT Std',
        'Bwd IAT Mean', 'Bwd IAT Std', 'Packet Length Mean', 'Packet Length Std',
        'Average Packet Size', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
        'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Active Mean']
pool = pool.dropna(subset=CORE).reset_index(drop=True)
# drop degenerate all-zero-variance rows that can break log transforms
pool = pool[(pool['Total Length of Fwd Packets'] > 0) & (pool['Total Length of Bwd Packets'] > 0)].reset_index(drop=True)
n_neg = (pool['Flow Duration'] < 0).sum()
if n_neg:
    print(f"NOTE: clipping {n_neg} row(s) with negative Flow Duration (CICFlowMeter artifact) to 0")
    pool['Flow Duration'] = pool['Flow Duration'].clip(lower=0)
print("Cleaned pool:", pool.shape)
print(pool.Label.value_counts())

NOISE_POOL = ['Fwd Packet Length Mean', 'Fwd Packet Length Std',
              'Bwd Packet Length Mean', 'Bwd Packet Length Std',
              'Packet Length Mean', 'Packet Length Std',
              'Average Packet Size', 'Active Mean']

def sample_label(label, n, seed):
    sub = pool[pool.Label == label]
    if len(sub) < n:
        print(f"  ! WARNING: requested {n} of '{label}' but only {len(sub)} available -> using all")
        return sub.copy()
    return sub.sample(n=n, random_state=seed).copy()

# ---------------------------------------------------------------------------
# TEST A -- distance-separable mechanism
# signal features: Flow Duration (log10), Total Fwd Packets
# classes: BENIGN / PortScan / DDoS / DoS Hulk  (n=2000 each, N=8000, matches paper)
# ---------------------------------------------------------------------------
print("\n=== Building Test A (distance) ===")
A_LABELS = ['BENIGN', 'PortScan', 'DDoS', 'DoS Hulk']
a_parts = [sample_label(l, 2000, seed=1) for l in A_LABELS]
A = pd.concat(a_parts, ignore_index=True)
A['sig1_raw_duration'] = A['Flow Duration']
A['sig1_log_duration'] = np.log10(A['Flow Duration'] + 1.0)
A['sig2_fwd_packets'] = A['Total Fwd Packets']
A['y_class'] = A['Label'].map({l: i for i, l in enumerate(A_LABELS)})
for i, nc in enumerate(NOISE_POOL):
    A[f'noise{i+1}'] = A[nc]
A_out = A[['sig1_raw_duration', 'sig1_log_duration', 'sig2_fwd_packets'] +
          [f'noise{i+1}' for i in range(len(NOISE_POOL))] + ['y_class', 'Label', 'source_file']]
print(A_out['Label'].value_counts())
A_out.to_csv(os.path.join(DATA_OUT_DIR, 'test_A_distance.csv'), index=False)

# ---------------------------------------------------------------------------
# TEST B -- relational/ratio mechanism
# signal features: upload_bytes (Total Length Fwd), download_bytes (Total Length Bwd), raw scale
# background = BENIGN, minority (12%) = Bot  (empirically: tight-ish ratio, WIDE magnitude range)
# ---------------------------------------------------------------------------
print("\n=== Building Test B (relational/ratio) ===")
N_B = 8000
FRAC_B = 0.12
n_minor_b = int(round(N_B * FRAC_B))
n_bg_b = N_B - n_minor_b
bg_b = sample_label('BENIGN', n_bg_b, seed=2)
minor_b = sample_label('Bot', n_minor_b, seed=2)
B = pd.concat([bg_b, minor_b], ignore_index=True)
B['sig1_upload_bytes'] = B['Total Length of Fwd Packets']
B['sig2_download_bytes'] = B['Total Length of Bwd Packets']
B['y_minority'] = (B['Label'] == 'Bot').astype(int)
for i, nc in enumerate(NOISE_POOL):
    B[f'noise{i+1}'] = B[nc]
B_out = B[['sig1_upload_bytes', 'sig2_download_bytes'] +
          [f'noise{i+1}' for i in range(len(NOISE_POOL))] + ['y_minority', 'Label', 'source_file']]
print(f"minority n={n_minor_b} ({FRAC_B:.0%}), background n={n_bg_b}")
print(B_out['Label'].value_counts())
B_out.to_csv(os.path.join(DATA_OUT_DIR, 'test_B_relational.csv'), index=False)

# ---------------------------------------------------------------------------
# TEST C -- scale/density mechanism
# signal features: Flow IAT Mean, Flow IAT Std, raw scale
# background = BENIGN, minority (4%) = SSH-Patator (empirically: centroid close to BENIGN, variance ~4-9% of BENIGN's)
# ---------------------------------------------------------------------------
print("\n=== Building Test C (scale/density) ===")
N_C = 8000
FRAC_C = 0.04
n_minor_c = int(round(N_C * FRAC_C))
n_bg_c = N_C - n_minor_c
bg_c = sample_label('BENIGN', n_bg_c, seed=3)
minor_c = sample_label('SSH-Patator', n_minor_c, seed=3)
C = pd.concat([bg_c, minor_c], ignore_index=True)
C['sig1_flow_iat_mean'] = C['Flow IAT Mean']
C['sig2_flow_iat_std'] = C['Flow IAT Std']
C['y_minority'] = (C['Label'] == 'SSH-Patator').astype(int)
for i, nc in enumerate(NOISE_POOL):
    C[f'noise{i+1}'] = C[nc]
C_out = C[['sig1_flow_iat_mean', 'sig2_flow_iat_std'] +
          [f'noise{i+1}' for i in range(len(NOISE_POOL))] + ['y_minority', 'Label', 'source_file']]
print(f"minority n={n_minor_c} ({FRAC_C:.0%}), background n={n_bg_c}")
print(C_out['Label'].value_counts())
C_out.to_csv(os.path.join(DATA_OUT_DIR, 'test_C_scale.csv'), index=False)

print("\nDone. Saved test_A_distance.csv, test_B_relational.csv, test_C_scale.csv")
