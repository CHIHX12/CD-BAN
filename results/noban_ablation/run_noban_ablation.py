"""
run_noban_ablation.py
=====================
Reviewer R1-1: "a comparison with a simpler binary classifier to see if similar
correlations arise WITHOUT the proposed attention mechanism."

We train TWO models under IDENTICAL conditions (same 838 extremes, same seeds,
same optimiser/epochs) and compare:
  - CD-BAN        : dual-branch GCN + bilinear attention (the proposed model)
  - CD-BAN-noBAN  : dual-branch GCN + simple feature CONCATENATION (no attention)

For each we report (i) held-out test AUROC and (ii) the fuzzy-zone correlation
(P(Weak) vs true log10K, n=1850) -- the emergent-gradient signal. If no-BAN
recovers a comparable fuzzy correlation, the gradient is not specific to the
bilinear attention; if it collapses, the attention contributes.

Outputs:
  results/noban_ablation/noban_ablation.csv   per-seed metrics for both models
  results/noban_ablation/noban_ablation.png/.svg
  results/noban_ablation/run_log.txt
"""
import os, sys, warnings, yaml
warnings.filterwarnings('ignore')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr, kendalltau
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models import CDBAN, MolecularGCN, MLPDecoder
from dataloader import smiles_to_pyg
from utils import set_seed

OUT = os.path.join(ROOT, 'results', 'noban_ablation'); os.makedirs(OUT, exist_ok=True)
log = []; pr = lambda s='': (print(s, flush=True), log.append(str(s)))
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# MAIN-TEXT PROTOCOL (matches main.py/trainer.py): lr 5e-5, up to 100 epochs,
# model selected by max validation AUROC. This makes the BAN arm reproduce the
# headline numbers (~0.925 AUROC / -0.468 fuzzy Pearson) so the no-BAN arm is a
# fair, like-for-like comparison.
EPOCHS = 100; LR = 5e-5

with open(os.path.join(ROOT, 'configs/CDBAN.yaml')) as f: cfg = yaml.safe_load(f)

class NoBAN(nn.Module):
    """Same dual-branch GCN, but fuse by simple concatenation (no bilinear attention)."""
    def __init__(s, **c):
        super().__init__()
        s.mg = c['GUEST']['MAX_NODES']; s.mh = c['HOST']['MAX_NODES']
        s.ge = MolecularGCN(c['GUEST']['NODE_IN_FEATS'], c['GUEST']['NODE_IN_EMBEDDING'], c['GUEST']['HIDDEN_LAYERS'])
        s.he = MolecularGCN(c['HOST']['NODE_IN_FEATS'], c['HOST']['NODE_IN_EMBEDDING'], c['HOST']['HIDDEN_LAYERS'])
        s.mlp = MLPDecoder(c['DECODER']['IN_DIM'], c['DECODER']['HIDDEN_DIM'], c['DECODER']['OUT_DIM'], 1)
    def forward(s, g, h):
        v_g = s.ge(g, s.mg)   # (B, max_nodes, 128)
        v_h = s.he(h, s.mh)   # (B, max_nodes, 128)
        vg = v_g.mean(dim=1)  # mean-pool over nodes -> (B, 128)
        vh = v_h.mean(dim=1)  # -> (B, 128)
        f = torch.cat([vg, vh], dim=1)   # (B, 256)
        return s.mlp(f)

_C = {}
def _pyg(x):
    if x not in _C: _C[x] = smiles_to_pyg(x)
    return _C[x]
class DS(Dataset):
    def __init__(s, g, h, y): s.g = [_pyg(a) for a in g]; s.h = [_pyg(b) for b in h]; s.y = list(y)
    def __len__(s): return len(s.y)
    def __getitem__(s, i): return s.g[i], s.h[i], float(s.y[i])
def coll(b):
    g, h, y = zip(*b)
    return Batch.from_data_list(list(g)), Batch.from_data_list(list(h)), torch.tensor(y)

tr = pd.read_csv(os.path.join(ROOT, 'data/binary/train.csv'))
va = pd.read_csv(os.path.join(ROOT, 'data/binary/val.csv'))
te = pd.read_csv(os.path.join(ROOT, 'data/binary/test.csv'))
fz = pd.read_csv(os.path.join(ROOT, 'data/binary/fuzzy.csv'))

def forward_scores(model, glist, hlist, is_noban):
    ds = DS(glist, hlist, [0]*len(glist))
    dl = DataLoader(ds, batch_size=128, shuffle=False, collate_fn=coll)
    out = []
    with torch.no_grad():
        for g, h, y in dl:
            g, h = g.to(dev), h.to(dev)
            if is_noban:
                sc = model(g, h)
            else:
                _, _, sc, _ = model(g, h, mode='eval')
            out.append(torch.sigmoid(sc.squeeze(1)).cpu().numpy())
    return np.concatenate(out)

def train_eval(make_model, seed):
    set_seed(seed)
    model = make_model().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cfg['CLASSIFIER']['POS_WEIGHT']], device=dev))
    is_noban = isinstance(model, NoBAN)
    dl = DataLoader(DS(tr['SMILES_Guest'].values, tr['SMILES_Host'].values, tr['label'].values),
                    batch_size=64, shuffle=True, collate_fn=coll, drop_last=True)
    best = 0; best_state = None
    for ep in range(EPOCHS):
        model.train()
        for g, h, y in dl:
            g, h, y = g.to(dev), h.to(dev), y.float().to(dev)
            sc = model(g, h) if is_noban else model(g, h, mode='train')[3]
            loss = bce(sc.squeeze(1), y)
            opt.zero_grad(); loss.backward(); opt.step()
        # val AUROC for early-stopping bookkeeping
        pv = forward_scores(model, va['SMILES_Guest'].values, va['SMILES_Host'].values, is_noban)
        auc = roc_auc_score(va['label'].values, pv) if va['label'].nunique() > 1 else 0
        if auc > best:
            best = auc; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    # test AUROC
    pt = forward_scores(model, te['SMILES_Guest'].values, te['SMILES_Host'].values, is_noban)
    test_auc = roc_auc_score(te['label'].values, pt)
    # fuzzy correlation (P(Weak) vs log10K)
    pf = forward_scores(model, fz['SMILES_Guest'].values, fz['SMILES_Host'].values, is_noban)
    yk = fz['log10K'].values
    return dict(seed=seed, test_AUROC=round(test_auc, 3),
                fuzzy_Pearson=round(pearsonr(pf, yk)[0], 3),
                fuzzy_Kendall=round(kendalltau(pf, yk)[0], 3))

rows = []
for seed in range(42, 52):
    ban = train_eval(lambda: CDBAN(**cfg), seed)
    noban = train_eval(lambda: NoBAN(**cfg), seed)
    rows.append(dict(seed=seed, BAN_test_AUROC=ban['test_AUROC'], BAN_fuzzy_Pearson=ban['fuzzy_Pearson'],
                     BAN_fuzzy_Kendall=ban['fuzzy_Kendall'],
                     NoBAN_test_AUROC=noban['test_AUROC'], NoBAN_fuzzy_Pearson=noban['fuzzy_Pearson'],
                     NoBAN_fuzzy_Kendall=noban['fuzzy_Kendall']))
    pr(f"seed {seed}: BAN  AUROC={ban['test_AUROC']:.3f} fuzzyP={ban['fuzzy_Pearson']:+.3f} | "
       f"noBAN AUROC={noban['test_AUROC']:.3f} fuzzyP={noban['fuzzy_Pearson']:+.3f}")

res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, 'noban_ablation.csv'), index=False)
pr("\n=== SUMMARY (10 seeds) ===")
pr(f"CD-BAN     test AUROC = {res.BAN_test_AUROC.mean():.3f} +/- {res.BAN_test_AUROC.std():.3f}   "
   f"fuzzy Pearson = {res.BAN_fuzzy_Pearson.mean():+.3f}   fuzzy Kendall = {res.BAN_fuzzy_Kendall.mean():+.3f}")
pr(f"no-BAN     test AUROC = {res.NoBAN_test_AUROC.mean():.3f} +/- {res.NoBAN_test_AUROC.std():.3f}   "
   f"fuzzy Pearson = {res.NoBAN_fuzzy_Pearson.mean():+.3f}   fuzzy Kendall = {res.NoBAN_fuzzy_Kendall.mean():+.3f}")

# ---- plot ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
x = np.arange(10); w = 0.38
ax.bar(x - w/2, res.BAN_test_AUROC, w, label='CD-BAN', color='#c0392b')
ax.bar(x + w/2, res.NoBAN_test_AUROC, w, label='no-BAN (concat)', color='#2a7f9e')
ax.axhline(res.BAN_test_AUROC.mean(), color='#c0392b', ls='--', lw=1)
ax.axhline(res.NoBAN_test_AUROC.mean(), color='#2a7f9e', ls='--', lw=1)
ax.set_xticks(x); ax.set_xticklabels([str(s) for s in res.seed], fontsize=8)
ax.set_ylabel('held-out test AUROC'); ax.set_xlabel('seed')
ax.set_title('Classification: BAN vs no-BAN (per seed)', fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.grid(axis='y', ls=':', alpha=0.4)

ax = axes[1]
ax.bar(x - w/2, res.BAN_fuzzy_Kendall, w, label='CD-BAN', color='#c0392b')
ax.bar(x + w/2, res.NoBAN_fuzzy_Kendall, w, label='no-BAN (concat)', color='#2a7f9e')
ax.axhline(res.BAN_fuzzy_Kendall.mean(), color='#c0392b', ls='--', lw=1)
ax.axhline(res.NoBAN_fuzzy_Kendall.mean(), color='#2a7f9e', ls='--', lw=1)
ax.set_xticks(x); ax.set_xticklabels([str(s) for s in res.seed], fontsize=8)
ax.set_ylabel('fuzzy-zone Kendall tau (P(Weak) vs log10K)'); ax.set_xlabel('seed')
ax.set_title('Emergent gradient: BAN vs no-BAN (per seed)', fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.grid(axis='y', ls=':', alpha=0.4)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'noban_ablation.png'), dpi=600)
fig.savefig(os.path.join(OUT, 'noban_ablation.svg'))
plt.close(fig)
pr("\nSaved: noban_ablation.csv, noban_ablation.png/.svg, run_log.txt")
open(os.path.join(OUT, 'run_log.txt'), 'w').write('\n'.join(log))
