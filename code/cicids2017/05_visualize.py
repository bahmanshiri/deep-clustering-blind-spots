import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import warnings, os
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(BASE, "..", "..", "data", "cicids2017"))
FIG_DIR = os.path.normpath(os.path.join(BASE, "..", "..", "figures"))
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({'font.size': 10, 'figure.dpi': 130})

NOISE_POOL = ['noise1','noise2','noise3','noise4','noise5','noise6','noise7','noise8']

# =====================================================================
# TEST A -- distance mechanism: true classes + naive-pipeline confusion
# =====================================================================
A = pd.read_csv(os.path.join(DATA_DIR, 'test_A_distance.csv'))
labels4 = ['BENIGN', 'PortScan', 'DDoS', 'DoS Hulk']
colors4 = {'BENIGN': '#4C72B0', 'PortScan': '#DD8452', 'DDoS': '#55A868', 'DoS Hulk': '#C44E52'}

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

ax = axes[0]
for lab in labels4:
    sub = A[A.Label == lab]
    ax.scatter(sub['sig1_log_duration'], sub['sig2_fwd_packets'], s=4, alpha=0.35, label=lab, color=colors4[lab])
ax.set_xlabel('log10(Flow Duration + 1)'); ax.set_ylabel('Total Fwd Packets')
ax.set_title('Mechanism A: true labels\n(real CIC-IDS2017)')
ax.legend(markerscale=4, fontsize=8)

# run naive pipeline on this exact saved dataset to get its k=3 clustering for the confusion panel
X = np.column_stack([A['sig1_log_duration'], A['sig2_fwd_packets']] + [A[n] for n in NOISE_POOL])
Xs = StandardScaler().fit_transform(X)
pcs = PCA(n_components=2, random_state=1).fit_transform(Xs)
km3 = KMeans(n_clusters=3, n_init=10, random_state=1).fit(pcs)
ari3 = adjusted_rand_score(A['y_class'], km3.labels_)

ax = axes[1]
sc = ax.scatter(pcs[:, 0], pcs[:, 1], c=km3.labels_, cmap='viridis', s=4, alpha=0.4)
ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
ax.set_title(f'Naive pipeline: PCA space,\nk=3 chosen by Silhouette (ARI={ari3:.2f})')

ax = axes[2]
ct = pd.crosstab(A['Label'], km3.labels_)
ct = ct.reindex(labels4)
im = ax.imshow(ct.values, cmap='Blues', aspect='auto')
ax.set_xticks(range(ct.shape[1])); ax.set_xticklabels([f'cluster {c}' for c in ct.columns])
ax.set_yticks(range(4)); ax.set_yticklabels(labels4)
for i in range(ct.shape[0]):
    for j in range(ct.shape[1]):
        ax.text(j, i, ct.values[i, j], ha='center', va='center',
                 color='white' if ct.values[i, j] > ct.values.max()/2 else 'black', fontsize=9)
ax.set_title('True label × naive cluster\n(which classes get merged)')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig04_domain_validation_mechanism_A.png'), bbox_inches='tight')
plt.close()
print("Test A merged-class crosstab:\n", ct)
print(f"Test A: naive pipeline selected k=3 (Silhouette), ARI={ari3:.4f}")

# =====================================================================
# TEST B -- relational/ratio mechanism
# =====================================================================
B = pd.read_csv(os.path.join(DATA_DIR, 'test_B_relational.csv'))
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

ax = axes[0]
bg = B[B.y_minority == 0]; mn = B[B.y_minority == 1]
ax.scatter(bg['sig1_upload_bytes'], bg['sig2_download_bytes'], s=4, alpha=0.25, color='#4C72B0', label='BENIGN (background)')
ax.scatter(mn['sig1_upload_bytes'], mn['sig2_download_bytes'], s=6, alpha=0.6, color='#C44E52', label='Bot (minority, 12%)')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('upload bytes (log scale)'); ax.set_ylabel('download bytes (log scale)')
ax.set_title('Mechanism B: raw magnitude view\n(minority hidden among background)')
ax.legend(markerscale=3, fontsize=8)

ax = axes[1]
theta_bg = np.degrees(np.arctan2(bg['sig2_download_bytes'], bg['sig1_upload_bytes']))
theta_mn = np.degrees(np.arctan2(mn['sig2_download_bytes'], mn['sig1_upload_bytes']))
ax.hist(theta_bg, bins=60, alpha=0.5, color='#4C72B0', label='BENIGN', density=True)
ax.hist(theta_mn, bins=60, alpha=0.6, color='#C44E52', label='Bot', density=True)
ax.set_xlabel('angle = atan2(download, upload)  [deg]'); ax.set_ylabel('density')
ax.set_title('Same data, angular view:\nBot is tight in angle, spread in magnitude')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig05_domain_validation_mechanism_B.png'), bbox_inches='tight')
plt.close()

# =====================================================================
# TEST C -- scale/density mechanism
# =====================================================================
C = pd.read_csv(os.path.join(DATA_DIR, 'test_C_scale.csv'))
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

ax = axes[0]
bg = C[C.y_minority == 0]; mn = C[C.y_minority == 1]
ax.scatter(bg['sig1_flow_iat_mean'], bg['sig2_flow_iat_std'], s=4, alpha=0.2, color='#4C72B0', label='BENIGN (background)')
ax.scatter(mn['sig1_flow_iat_mean'], mn['sig2_flow_iat_std'], s=8, alpha=0.7, color='#C44E52', label='SSH-Patator (minority, 4%)')
ax.set_xscale('symlog'); ax.set_yscale('symlog')
ax.set_xlabel('Flow IAT Mean (symlog)'); ax.set_ylabel('Flow IAT Std (symlog)')
ax.set_title('Mechanism C: full range\n(minority sits inside background cloud)')
ax.legend(markerscale=2, fontsize=8)

ax = axes[1]
# zoom near the shared centroid to show the density difference directly
lim = np.percentile(bg['sig1_flow_iat_mean'], 90)
limy = np.percentile(bg['sig2_flow_iat_std'], 90)
ax.scatter(bg['sig1_flow_iat_mean'], bg['sig2_flow_iat_std'], s=5, alpha=0.25, color='#4C72B0', label='BENIGN')
ax.scatter(mn['sig1_flow_iat_mean'], mn['sig2_flow_iat_std'], s=10, alpha=0.8, color='#C44E52', label='SSH-Patator')
ax.set_xlim(-lim*0.05, lim); ax.set_ylim(-limy*0.05, limy)
ax.set_xlabel('Flow IAT Mean'); ax.set_ylabel('Flow IAT Std')
ax.set_title('Zoomed near shared centroid:\nsame location, visibly tighter spread')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig06_domain_validation_mechanism_C.png'), bbox_inches='tight')
plt.close()

print("Saved fig04_domain_validation_mechanism_A.png, fig05_domain_validation_mechanism_B.png, "
      "fig06_domain_validation_mechanism_C.png (Section 4.2, Figures 4-6)")
