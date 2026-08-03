import pandas as pd
import numpy as np
import glob, os, gc

np.random.seed(42)

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "raw")
OUT_PATH = os.path.join(BASE, "pool.parquet")

USE_COLS = [
    ' Flow Duration',
    ' Total Fwd Packets',
    ' Total Backward Packets',
    'Total Length of Fwd Packets',
    ' Total Length of Bwd Packets',
    ' Fwd Packet Length Mean',
    ' Fwd Packet Length Std',
    'Bwd Packet Length Mean' if False else ' Bwd Packet Length Mean',
    ' Bwd Packet Length Std',
    'Flow Bytes/s',
    ' Flow Packets/s',
    ' Flow IAT Mean',
    ' Flow IAT Std',
    ' Fwd IAT Mean',
    ' Fwd IAT Std',
    ' Bwd IAT Mean',
    ' Bwd IAT Std',
    ' Packet Length Mean',
    ' Packet Length Std',
    ' Down/Up Ratio',
    ' Average Packet Size',
    'Active Mean',
    'Idle Mean',
    ' Label',
]

# per-label sampling caps (None = take all)
BENIGN_PER_FILE_CAP = 1500
LARGE_ATTACK_CAP = 6000   # for PortScan, DDoS, DoS Hulk etc.

files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
print(f"Found {len(files)} files")
if not files:
    raise SystemExit(
        f"No CSV files found in {DATA_DIR}. Download the 8 raw CIC-IDS2017 "
        "MachineLearningCVE CSVs (Sharafaldin, Lashkari and Ghorbani, 2018) "
        f"and place them in {DATA_DIR} before running this script. See "
        "Appendix A / README.md for details."
    )

pieces = []
for fp in files:
    fname = os.path.basename(fp)
    print(f"\n--- {fname} ---")
    df = pd.read_csv(fp, usecols=USE_COLS, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df['Label'] = df['Label'].astype(str).str.strip()
    # fix mangled encoding for Web Attack labels (en-dash became replacement char)
    df['Label'] = df['Label'].str.replace(r'^Web Attack.*Brute Force$', 'Web Attack-Brute Force', regex=True)
    df['Label'] = df['Label'].str.replace(r'^Web Attack.*XSS$', 'Web Attack-XSS', regex=True)
    df['Label'] = df['Label'].str.replace(r'^Web Attack.*Sql Injection$', 'Web Attack-SQL Injection', regex=True)
    df['source_file'] = fname

    counts = df['Label'].value_counts()
    print(counts.to_dict())

    keep_parts = []
    for label, grp in df.groupby('Label'):
        n = len(grp)
        if label == 'BENIGN':
            cap = BENIGN_PER_FILE_CAP
        elif n > LARGE_ATTACK_CAP:
            cap = LARGE_ATTACK_CAP
        else:
            cap = n  # keep all of rare/small classes
        if n > cap:
            grp = grp.sample(n=cap, random_state=42)
        keep_parts.append(grp)
    kept = pd.concat(keep_parts, ignore_index=True)
    print(f"kept {len(kept)} / {len(df)} rows")
    pieces.append(kept)
    del df, kept, keep_parts
    gc.collect()

pool = pd.concat(pieces, ignore_index=True)
print(f"\n=== TOTAL POOL: {len(pool)} rows ===")
print(pool['Label'].value_counts())

pool.to_pickle(OUT_PATH.replace('.parquet', '.pkl'))
print(f"\nSaved to {OUT_PATH.replace('.parquet', '.pkl')}")
