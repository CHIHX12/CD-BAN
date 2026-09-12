"""
run_calibration.py
==================
Reviewer R1-2: report calibration plots. For the binary classifier we plot
predicted P(Weak) vs the true weak-binding frequency (reliability diagram) on
the held-out test set (n=120), plus a histogram of predicted probabilities, and
report the Expected Calibration Error (ECE) and Brier score.

Outputs:
  results/calibration/calibration.png/.svg
  results/calibration/calibration_metrics.csv
  results/calibration/run_log.txt
"""
import os, sys, warnings, yaml
warnings.filterwarnings('ignore')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models import CDBAN
from dataloader import smiles_to_pyg
from utils import set_seed

OUT = os.path.join(ROOT, 'results', 'calibration'); os.makedirs(OUT, exist_ok=True)
log = []; pr = lambda s='': (print(s, flush=True), log.append(str(s)))
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 49
MODEL_PATH = os.path.join(ROOT, f'results/seed_{SEED}/best_model_epoch_69.pth')

with open(os.path.join(ROOT, 'configs/CDBAN.yaml')) as f: cfg = yaml.safe_load(f)
set_seed(SEED)
model = CDBAN(**cfg).to(dev)
model.load_state_dict(torch.load(MODEL_PATH, map_location=dev))
model.eval()

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

te = pd.read_csv(os.path.join(ROOT, 'data/binary/test.csv'))
dl = DataLoader(DS(te), batch_size=64, shuffle=False, collate_fn=coll)
probs = []
with torch.no_grad():
    for g, h in dl:
        g, h = g.to(dev), h.to(dev)
        _, _, f, score = model(g, h, mode='train')
        probs.append(torch.sigmoid(score.squeeze(1)).cpu().numpy())
pweak = np.concatenate(probs)
y = te['label'].values.astype(float)          # 1 = weak, 0 = strong
pr(f"test set: n={len(y)}  weak={int(y.sum())}  strong={int((1-y).sum())}")

# Brier score
brier = float(np.mean((pweak - y) ** 2))
# ECE (10 bins)
nbins = 10
bins = np.linspace(0, 1, nbins + 1)
idx = np.digitize(pweak, bins[1:-1])
ece = 0.0
conf, acc, cnt = [], [], []
for b in range(nbins):
    m = idx == b
    if m.sum() == 0: continue
    conf.append(pweak[m].mean()); acc.append(y[m].mean()); cnt.append(int(m.sum()))
ece = float(np.mean(np.abs(np.array(conf) - np.array(acc)) * np.array(cnt)) / len(y))
pr(f"Brier score = {brier:.4f}   ECE({nbins} bins) = {ece:.4f}")

pd.DataFrame({'metric': ['brier', 'ece_10bin'], 'value': [round(brier, 4), round(ece, 4)]}) \
  .to_csv(os.path.join(OUT, 'calibration_metrics.csv'), index=False)

# ---- plot ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
ax = axes[0]
# reliability curve
xs = [0.5]; ys = [0.5]
for c, a, n in zip(conf, acc, cnt):
    xs.append(c); ys.append(a)
xs, ys = np.array(xs), np.array(ys)
order = np.argsort(xs)
ax.plot(xs[order], ys[order], 'o-', color='#2a7f9e', lw=2, label='model')
ax.plot([0, 1], [0, 1], '--', color='grey', lw=1, label='perfectly calibrated')
ax.fill_between(xs[order], ys[order], xs[order], alpha=0.15, color='#2a7f9e')
ax.set_xlabel('mean predicted P(Weak)'); ax.set_ylabel('observed weak-binding frequency')
ax.set_title(f'Reliability diagram (test, n={len(y)})\nECE={ece:.3f}', fontsize=11, fontweight='bold')
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(fontsize=9); ax.grid(ls=':', alpha=0.4)

ax = axes[1]
ax.hist(pweak, bins=20, range=(0, 1), color='#2a7f9e', alpha=0.8, edgecolor='white')
ax.set_xlabel('predicted P(Weak)'); ax.set_ylabel('count')
ax.set_title(f'Distribution of predicted probabilities\nBrier={brier:.3f}', fontsize=11, fontweight='bold')
ax.grid(ls=':', alpha=0.4)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'calibration.png'), dpi=600)
fig.savefig(os.path.join(OUT, 'calibration.svg'))
plt.close(fig)
pr("Saved: calibration.png/.svg, calibration_metrics.csv, run_log.txt")
open(os.path.join(OUT, 'run_log.txt'), 'w').write('\n'.join(log))
