import pandas as pd
import numpy as np
import warnings, json, time, sys, os
warnings.filterwarnings('ignore')

SEED_START = int(sys.argv[1]) if len(sys.argv) > 1 else 0
SEED_END = int(sys.argv[2]) if len(sys.argv) > 2 else 20  # exclusive
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(BASE, "..", "..", "data", "cicids2017"))
CHECKPOINT_PATH = os.path.join(BASE, 'checkpoint_results.json')

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score, f1_score, precision_score, recall_score, normalized_mutual_info_score

t0 = time.time()
pd.set_option('display.width', 160)

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
pool = pool[(pool['Total Length of Fwd Packets'] > 0) & (pool['Total Length of Bwd Packets'] > 0)].reset_index(drop=True)
n_neg = (pool['Flow Duration'] < 0).sum()
if n_neg:
    print(f"NOTE: clipping {n_neg} row(s) with negative Flow Duration (CICFlowMeter artifact) to 0")
    pool['Flow Duration'] = pool['Flow Duration'].clip(lower=0)

NOISE_POOL = ['Fwd Packet Length Mean', 'Fwd Packet Length Std',
              'Bwd Packet Length Mean', 'Bwd Packet Length Std',
              'Packet Length Mean', 'Packet Length Std',
              'Average Packet Size', 'Active Mean']

def sample_label(label, n, seed):
    sub = pool[pool.Label == label]
    if len(sub) < n:
        return sub.copy()
    return sub.sample(n=n, random_state=seed).copy()

# ---------------------------------------------------------------------------
# dataset builders (parameterized by seed, for resampling-based robustness)
# ---------------------------------------------------------------------------
A_LABELS = ['BENIGN', 'PortScan', 'DDoS', 'DoS Hulk']

def build_A(seed):
    parts = [sample_label(l, 2000, seed=seed) for l in A_LABELS]
    df = pd.concat(parts, ignore_index=True)
    sig_raw = df[['Flow Duration', 'Total Fwd Packets']].to_numpy(dtype=float)
    sig_log = np.column_stack([np.log10(df['Flow Duration'].to_numpy(dtype=float) + 1.0),
                                df['Total Fwd Packets'].to_numpy(dtype=float)])
    noise = df[NOISE_POOL].to_numpy(dtype=float)
    y = df['Label'].map({l: i for i, l in enumerate(A_LABELS)}).to_numpy()
    return sig_raw, sig_log, noise, y

def build_B(seed, n_total=8000, frac=0.12):
    n_minor = int(round(n_total * frac)); n_bg = n_total - n_minor
    bg = sample_label('BENIGN', n_bg, seed=seed)
    minor = sample_label('Bot', n_minor, seed=seed)
    df = pd.concat([bg, minor], ignore_index=True)
    sig = df[['Total Length of Fwd Packets', 'Total Length of Bwd Packets']].to_numpy(dtype=float)
    noise = df[NOISE_POOL].to_numpy(dtype=float)
    y = (df['Label'] == 'Bot').astype(int).to_numpy()
    return sig, noise, y

def build_C(seed, n_total=8000, frac=0.04):
    n_minor = int(round(n_total * frac)); n_bg = n_total - n_minor
    bg = sample_label('BENIGN', n_bg, seed=seed)
    minor = sample_label('SSH-Patator', n_minor, seed=seed)
    df = pd.concat([bg, minor], ignore_index=True)
    sig = df[['Flow IAT Mean', 'Flow IAT Std']].to_numpy(dtype=float)
    noise = df[NOISE_POOL].to_numpy(dtype=float)
    y = (df['Label'] == 'SSH-Patator').astype(int).to_numpy()
    return sig, noise, y

# ---------------------------------------------------------------------------
# pipeline stages
# ---------------------------------------------------------------------------
def naive_pipeline(sig, noise, y_true, true_k, k_range=range(2, 7), seed=0):
    X = np.column_stack([sig, noise])
    Xs = StandardScaler().fit_transform(X)
    pcs = PCA(n_components=2, random_state=seed).fit_transform(Xs)
    best_k, best_sil, best_labels = None, -2, None
    sil_by_k = {}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(pcs)
        sil = silhouette_score(pcs, km.labels_)
        sil_by_k[k] = sil
        if sil > best_sil:
            best_k, best_sil, best_labels = k, sil, km.labels_
    ari_selected = adjusted_rand_score(y_true, best_labels)
    # also force true_k for reference (does correct-k help?)
    km_truek = KMeans(n_clusters=true_k, n_init=10, random_state=seed).fit(pcs)
    ari_truek = adjusted_rand_score(y_true, km_truek.labels_)
    nmi_selected = normalized_mutual_info_score(y_true, best_labels)
    return dict(k_selected=best_k, silhouette_selected=best_sil, ari_selected=ari_selected,
                ari_at_true_k=ari_truek, nmi_selected=nmi_selected, sil_by_k=sil_by_k,
                pred_labels=best_labels)

def minority_f1(y_true, pred_labels):
    # map smallest predicted cluster (excluding DBSCAN noise=-1 treated as own cluster) -> predicted minority
    vals, counts = np.unique(pred_labels, return_counts=True)
    order = np.argsort(counts)
    smallest_label = vals[order[0]]
    pred_minor = (pred_labels == smallest_label).astype(int)
    # try both polarity (in case smallest cluster is actually majority-like by fluke) and take better F1
    f1a = f1_score(y_true, pred_minor, zero_division=0)
    f1b = f1_score(y_true, 1 - pred_minor, zero_division=0)
    if f1b > f1a:
        pred_minor = 1 - pred_minor
    return dict(f1=f1_score(y_true, pred_minor, zero_division=0),
                precision=precision_score(y_true, pred_minor, zero_division=0),
                recall=recall_score(y_true, pred_minor, zero_division=0))

def hand_informed_A(sig, y_true, seed):
    Xs = StandardScaler().fit_transform(sig)
    km = KMeans(n_clusters=4, n_init=10, random_state=seed).fit(Xs)
    return dict(ari=adjusted_rand_score(y_true, km.labels_),
                nmi=normalized_mutual_info_score(y_true, km.labels_))

def hand_informed_B_angular(sig, y_true, seed):
    theta = np.arctan2(sig[:, 1], sig[:, 0]).reshape(-1, 1)
    theta_s = StandardScaler().fit_transform(theta)
    best = None
    for eps in np.linspace(0.02, 1.0, 25):
        db = DBSCAN(eps=eps, min_samples=15).fit(theta_s)
        labels = db.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters < 1:
            continue
        ari = adjusted_rand_score(y_true, labels)
        if best is None or ari > best[0]:
            best = (ari, eps, labels)
    if best is None:
        return dict(ari=0.0, f1=0.0, precision=0.0, recall=0.0, eps=None)
    ari, eps, labels = best
    m = minority_f1(y_true, labels)
    return dict(ari=ari, eps=eps, **m)

def hand_informed_C_gmm(sig, y_true, seed):
    Xs = StandardScaler().fit_transform(sig)
    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=seed, n_init=5).fit(Xs)
    labels = gmm.predict(Xs)
    ari = adjusted_rand_score(y_true, labels)
    m = minority_f1(y_true, labels)
    return dict(ari=ari, **m)

def radial_centroid_oracle(sig, y_true):
    centroid = sig.mean(axis=0)
    dist = np.sqrt(((sig - centroid) ** 2).sum(axis=1))
    order = np.argsort(dist)
    best = dict(f1=-1)
    # oracle grid search over thresholds, both directions (closer=minority / farther=minority)
    thresholds = np.percentile(dist, np.linspace(1, 99, 99))
    for th in thresholds:
        for pred in [(dist <= th).astype(int), (dist > th).astype(int)]:
            f1 = f1_score(y_true, pred, zero_division=0)
            if f1 > best['f1']:
                best = dict(f1=f1, threshold=th,
                            precision=precision_score(y_true, pred, zero_division=0),
                            recall=recall_score(y_true, pred, zero_division=0),
                            ari=adjusted_rand_score(y_true, pred))
    return best

# ---------------------------------------------------------------------------
# multi-seed robustness loop
# ---------------------------------------------------------------------------
if os.path.exists(CHECKPOINT_PATH):
    with open(CHECKPOINT_PATH) as f:
        results = json.load(f)
else:
    results = {'A_raw': [], 'A_log': [], 'B': [], 'C': []}

print(f"Running seeds [{SEED_START}, {SEED_END}) ...")
for s in range(SEED_START, SEED_END):
    seed = 100 + s
    # --- A ---
    sig_raw, sig_log, noise, yA = build_A(seed)
    naive_raw = naive_pipeline(sig_raw, noise, yA, true_k=4, seed=seed)
    hand_raw = hand_informed_A(sig_raw, yA, seed)
    results['A_raw'].append(dict(seed=seed, naive_ari=naive_raw['ari_selected'],
                                  naive_ari_truek=naive_raw['ari_at_true_k'],
                                  naive_k=naive_raw['k_selected'], hand_ari=hand_raw['ari']))
    naive_log = naive_pipeline(sig_log, noise, yA, true_k=4, seed=seed)
    hand_log = hand_informed_A(sig_log, yA, seed)
    results['A_log'].append(dict(seed=seed, naive_ari=naive_log['ari_selected'],
                                  naive_ari_truek=naive_log['ari_at_true_k'],
                                  naive_k=naive_log['k_selected'], hand_ari=hand_log['ari']))
    # --- B ---
    sigB, noiseB, yB = build_B(seed)
    naiveB = naive_pipeline(sigB, noiseB, yB, true_k=2, seed=seed)
    handB = hand_informed_B_angular(sigB, yB, seed)
    oracleB = radial_centroid_oracle(sigB, yB)
    f1_naiveB = minority_f1(yB, naiveB['pred_labels'])
    results['B'].append(dict(seed=seed, naive_ari=naiveB['ari_selected'], naive_k=naiveB['k_selected'],
                              naive_f1=f1_naiveB['f1'], hand_ari=handB['ari'], hand_f1=handB['f1'],
                              oracle_f1=oracleB['f1'], oracle_ari=oracleB['ari']))
    # --- C ---
    sigC, noiseC, yC = build_C(seed)
    naiveC = naive_pipeline(sigC, noiseC, yC, true_k=2, seed=seed)
    handC = hand_informed_C_gmm(sigC, yC, seed)
    oracleC = radial_centroid_oracle(sigC, yC)
    f1_naiveC = minority_f1(yC, naiveC['pred_labels'])
    results['C'].append(dict(seed=seed, naive_ari=naiveC['ari_selected'], naive_k=naiveC['k_selected'],
                              naive_f1=f1_naiveC['f1'], hand_ari=handC['ari'], hand_f1=handC['f1'],
                              oracle_f1=oracleC['f1'], oracle_ari=oracleC['ari']))
    print(f"  seed {s+1} done  ({time.time()-t0:.0f}s elapsed)")

with open(CHECKPOINT_PATH, 'w') as f:
    json.dump(results, f)
print(f"Checkpoint saved: A_raw n={len(results['A_raw'])}, B n={len(results['B'])}, C n={len(results['C'])}")

# ---------------------------------------------------------------------------
# aggregate + report (based on whatever is in the checkpoint so far)
# ---------------------------------------------------------------------------
def summarize(rows, cols):
    df = pd.DataFrame(rows)
    out = {}
    for c in cols:
        out[c] = f"{df[c].mean():.3f} ± {df[c].std():.3f}  [{df[c].min():.3f}, {df[c].max():.3f}]"
    return out, df

print("\n" + "="*100)
print("TEST A (distance mechanism) -- RAW duration/packet-count")
s, dfA_raw = summarize(results['A_raw'], ['naive_ari', 'naive_ari_truek', 'hand_ari'])
for k, v in s.items(): print(f"  {k:20s}: {v}")
print("  naive k selected distribution:", dfA_raw['naive_k'].value_counts().to_dict())

print("\nTEST A (distance mechanism) -- LOG10 duration/packet-count")
s, dfA_log = summarize(results['A_log'], ['naive_ari', 'naive_ari_truek', 'hand_ari'])
for k, v in s.items(): print(f"  {k:20s}: {v}")
print("  naive k selected distribution:", dfA_log['naive_k'].value_counts().to_dict())

print("\n" + "="*100)
print("TEST B (relational/ratio mechanism) -- Bot minority (12%) in BENIGN background")
s, dfB = summarize(results['B'], ['naive_ari', 'naive_f1', 'hand_ari', 'hand_f1', 'oracle_f1', 'oracle_ari'])
for k, v in s.items(): print(f"  {k:20s}: {v}")
print("  naive k selected distribution:", dfB['naive_k'].value_counts().to_dict())

print("\n" + "="*100)
print("TEST C (scale/density mechanism) -- SSH-Patator minority (4%) in BENIGN background")
s, dfC = summarize(results['C'], ['naive_ari', 'naive_f1', 'hand_ari', 'hand_f1', 'oracle_f1', 'oracle_ari'])
for k, v in s.items(): print(f"  {k:20s}: {v}")
print("  naive k selected distribution:", dfC['naive_k'].value_counts().to_dict())

# save everything
all_df = {'A_raw': dfA_raw, 'A_log': dfA_log, 'B': dfB, 'C': dfC}
with open(os.path.join(DATA_DIR, 'robustness_results.json'), 'w') as f:
    json.dump({k: v.to_dict('records') for k, v in all_df.items()}, f, indent=2, default=str)

for k, v in all_df.items():
    v.to_csv(os.path.join(DATA_DIR, f'robustness_{k}.csv'), index=False)

print(f"\nTotal time: {time.time()-t0:.0f}s")
print("Saved robustness_*.csv and robustness_results.json")
