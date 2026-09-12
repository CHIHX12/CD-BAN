"""
run_embedding_viz.py
====================
Reviewer R1-1: visualise latent embeddings (PCA / UMAP) to show whether
compounds organise by affinity.

We extract the fused bilinear-attention representation f (256-d) from the
best classifier (seed 49) for the 1,850 out-of-distribution fuzzy-zone
compounds (the set the paper claims "organises into a continuous affinity
gradient"), reduce to 2D with PCA and UMAP, and colour by true log10K.

Outputs:
  results/embedding_viz/fuzzy_embeddings.npy   raw fused representations
  results/embedding_viz/embedding_viz.png/.svg  PCA + UMAP coloured by log10K
  results/embedding_viz/run_log.txt
"""
import os, sys, warnings, yaml
warnings.filterwarnings('ignore')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from scipy.stats import pearsonr, kendalltau, spearmanr
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models import CDBAN
from dataloader import smiles_to_pyg
from utils import set_seed
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

OUT = os.path.join(ROOT, 'results', 'embedding_viz'); os.makedirs(OUT, exist_ok=True)
log = []; pr = lambda s='': (print(s, flush=True), log.append(str(s)))
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 49
MODEL_PATH = os.path.join(ROOT, f'results/seed_{SEED}/best_model_epoch_69.pth')

with open(os.path.join(ROOT, 'configs/CDBAN.yaml')) as f: cfg = yaml.safe_load(f)
set_seed(SEED)
model = CDBAN(**cfg).to(dev)
model.load_state_dict(torch.load(MODEL_PATH, map_location=dev))
model.eval()
pr(f"Loaded {MODEL_PATH}")

_C = {}
def _pyg(s):
    if s not in _C: _C[s] = smiles_to_pyg(s)
    return _C[s]

class DS(Dataset):
    def __init__(s, df): s.df = df
    def __len__(s): return len(s.df)
    def __getitem__(s, i):
        r = s.df.iloc[i]
        return _pyg(r['SMILES_Guest']), _pyg(r['SMILES_Host'])
def coll(b):
    g, h = zip(*b)
    return Batch.from_data_list(list(g)), Batch.from_data_list(list(h))

def extract_f(df):
    Fs, scores = [], []
    dl = DataLoader(DS(df), batch_size=64, shuffle=False, collate_fn=coll)
    with torch.no_grad():
        for g, h in dl:
            g, h = g.to(dev), h.to(dev)
            # model is in .eval(); mode='train' only selects the return tuple that exposes f
            _, _, f, score = model(g, h, mode='train')
            Fs.append(f.cpu().numpy())
            scores.append(torch.sigmoid(score.squeeze(1)).cpu().numpy())
    return np.concatenate(Fs), np.concatenate(scores)

fz = pd.read_csv(os.path.join(ROOT, 'data/binary/fuzzy.csv'))
F, pweak = extract_f(fz)
yk = fz['log10K'].values
pr(f"fuzzy fused embeddings: {F.shape} (n={len(fz)})")
np.save(os.path.join(OUT, 'fuzzy_embeddings.npy'), F)
np.save(os.path.join(OUT, 'fuzzy_pweak.npy'), pweak)

Fs = StandardScaler().fit_transform(F)

pca = PCA(n_components=2, random_state=0)
pca2 = pca.fit_transform(Fs)
pr(f"PCA explained variance (2 comp): {pca.explained_variance_ratio_.sum():.3f}")

umap2 = None
try:
    import umap
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1, metric='cosine')
    umap2 = reducer.fit_transform(Fs)
    pr("UMAP done")
except Exception as e:
    pr(f"UMAP failed: {e}")

# sanity: does the model output still correlate with affinity on this set?
pr(f"P(Weak) vs log10K: Pearson={pearsonr(pweak, yk)[0]:.3f}  Kendall={kendalltau(pweak, yk)[0]:.3f}")

# ---- plot ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
panels = [('PCA', pca2), ('UMAP', umap2)]
for ax, (title, pts) in zip(axes, panels):
    if pts is None:
        ax.axis('off'); ax.set_title(title + ' (unavailable)'); continue
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=yk, cmap='viridis', s=8, alpha=0.85, edgecolors='none')
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label('true log$_{10}$K', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(title + ' dim 1'); ax.set_ylabel(title + ' dim 2')
    ax.tick_params(labelsize=9)
fig.suptitle('Latent representation of 1,850 out-of-distribution fuzzy-zone compounds, coloured by affinity',
             fontsize=12, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUT, 'embedding_viz.png'), dpi=600)
fig.savefig(os.path.join(OUT, 'embedding_viz.svg'))
plt.close(fig)
pr("Saved: embedding_viz.png/.svg, fuzzy_embeddings.npy, run_log.txt")
open(os.path.join(OUT, 'run_log.txt'), 'w').write('\n'.join(log))
