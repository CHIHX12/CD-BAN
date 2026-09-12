"""
run_host_derivative_coldstart.py
================================
Reviewer R1-4: the external tests keep the host fixed (mostly beta-CD). Examine
how the model handles DIFFERENT HOST DERIVATIVES. We run a strict cold start in
which ALL substituted/modified beta-CDs (SBE, CM, Me, DM, TM, Ac, Suc, SO4) are
held out entirely -- the model is trained only on native alpha/beta/gamma +
HP-beta and then tested on the modified-beta derivatives it never saw. This is a
harder, more direct test of "host-derivative generalisation" than the
4-family leave-one-out.

Outputs:
  results/host_derivative_coldstart/host_derivative_coldstart.csv
  results/host_derivative_coldstart/run_log.txt
"""
import os, sys, warnings, yaml
warnings.filterwarnings('ignore')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
from models import MolecularGCN, MLPDecoder
from ban import BANLayer
from torch.nn.utils.weight_norm import weight_norm
from dataloader import smiles_to_pyg
from utils import set_seed

OUT = os.path.join(ROOT, 'results', 'host_derivative_coldstart'); os.makedirs(OUT, exist_ok=True)
log = []; pr = lambda s='': (print(s, flush=True), log.append(str(s)))
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 60

MODIFIED = ['sulfobutylether', 'carboxymethyl', 'methyl', 'acetyl', 'succinyl', 'sulfate', 'di-o-methyl', 'trimethyl']
def is_modified(h):
    h = str(h).lower()
    return any(k in h for k in MODIFIED)

with open(os.path.join(ROOT, 'configs/CDBAN.yaml')) as f: cfg = yaml.safe_load(f)

class MT(nn.Module):
    def __init__(s, **c):
        super().__init__()
        s.mg = c['GUEST']['MAX_NODES']; s.mh = c['HOST']['MAX_NODES']
        s.ge = MolecularGCN(c['GUEST']['NODE_IN_FEATS'], c['GUEST']['NODE_IN_EMBEDDING'], c['GUEST']['HIDDEN_LAYERS'])
        s.he = MolecularGCN(c['HOST']['NODE_IN_FEATS'], c['HOST']['NODE_IN_EMBEDDING'], c['HOST']['HIDDEN_LAYERS'])
        s.bcn = weight_norm(BANLayer(c['GUEST']['HIDDEN_LAYERS'][-1], c['HOST']['HIDDEN_LAYERS'][-1],
                                     c['DECODER']['IN_DIM'], c['BCN']['HEADS']), name='h_mat', dim=None)
        s.cls = MLPDecoder(c['DECODER']['IN_DIM'], c['DECODER']['HIDDEN_DIM'], c['DECODER']['OUT_DIM'], 1)
        s.reg = MLPDecoder(c['DECODER']['IN_DIM'], c['DECODER']['HIDDEN_DIM'], c['DECODER']['OUT_DIM'], 1)
    def forward(s, g, h):
        f, _ = s.bcn(s.ge(g, s.mg), s.he(h, s.mh))
        return s.cls(f).squeeze(1), s.reg(f).squeeze(1)

_C = {}
def _pyg(x):
    if x not in _C: _C[x] = smiles_to_pyg(x)
    return _C[x]
class DS(Dataset):
    def __init__(s, df): s.it = [(_pyg(r['SMILES_Guest']), _pyg(r['SMILES_Host']), float(r['label']), float(r['log10K'])) for _, r in df.iterrows()]
    def __len__(s): return len(s.it)
    def __getitem__(s, i): return s.it[i]
def coll(b):
    g, h, y, k = zip(*b)
    return Batch.from_data_list(list(g)), Batch.from_data_list(list(h)), torch.tensor(y), torch.tensor(k, dtype=torch.float)

ext = pd.concat([pd.read_csv(os.path.join(ROOT, 'data/binary/' + f)) for f in
                 ('train.csv', 'val.csv', 'test.csv')], ignore_index=True)
fz = pd.read_csv(os.path.join(ROOT, 'data/binary/fuzzy.csv')); fz['label'] = -1

mod_ext = ext[ext['Host'].map(is_modified)]
mod_fz  = fz[fz['Host'].map(is_modified)]
tr_ext  = ext[~ext['Host'].map(is_modified)]
tr_fz   = fz[~fz['Host'].map(is_modified)]
pr(f"modified-beta extremes: {len(mod_ext)} (strong={int((mod_ext.label==0).sum())})  | "
   f"modified-beta fuzzy: {len(mod_fz)}")
pr(f"training (native+HP-beta) extremes: {len(tr_ext)}  fuzzy: {len(tr_fz)}")

@torch.no_grad()
def predict(model, ds):
    model.eval(); C, R = [], []
    for g, h, y, k in DataLoader(ds, batch_size=128, shuffle=False, collate_fn=coll):
        cl, rg = model(g.to(dev), h.to(dev))
        C.extend(torch.sigmoid(cl).cpu().tolist()); R.extend(rg.cpu().tolist())
    return np.array(C), np.array(R)

ds_tr = DS(pd.concat([tr_ext, tr_fz], ignore_index=True))
ds_te = DS(mod_ext); ds_fz = DS(mod_fz)

rows = []
for seed in range(42, 52):
    set_seed(seed)
    model = MT(**cfg).to(dev); opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cfg['CLASSIFIER']['POS_WEIGHT']], device=dev)); mse = nn.MSELoss()
    dl = DataLoader(ds_tr, batch_size=64, shuffle=True, collate_fn=coll, drop_last=True)
    for ep in range(EPOCHS):
        model.train()
        for g, h, y, k in dl:
            g, h, y, k = g.to(dev), h.to(dev), y.to(dev), k.to(dev)
            cl, rg = model(g, h); m = (y >= 0)
            loss = (bce(cl[m], y[m].float()) if m.any() else 0) + mse(rg, k)
            opt.zero_grad(); loss.backward(); opt.step()
    pc, _ = predict(model, ds_te)
    _, rr = predict(model, ds_fz)
    auc = roc_auc_score(mod_ext['label'].values, pc) if mod_ext['label'].nunique() > 1 else float('nan')
    pear = pearsonr(mod_fz['log10K'].values, rr)[0] if len(mod_fz) > 2 else float('nan')
    rows.append(dict(seed=seed, AUROC=round(auc, 3), Pearson=round(pear, 3)))
    pr(f"  seed {seed}: modified-beta AUROC={auc:.3f}  fuzzy Pearson={pear:+.3f}")

res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, 'host_derivative_coldstart.csv'), index=False)
pr(f"\n=== HOST-DERIVATIVE COLD START (all modified-beta held out, 10 seeds) ===")
pr(f"  AUROC   = {res.AUROC.mean():.3f} +/- {res.AUROC.std():.3f}")
pr(f"  Pearson = {res.Pearson.mean():+.3f} +/- {res.Pearson.std():.3f}")
pr("  (warm baselines: AUROC ~0.95, Pearson ~0.65; 4-family cold-start mean AUROC 0.835)")
pr("Saved: host_derivative_coldstart.csv, run_log.txt")
open(os.path.join(OUT, 'run_log.txt'), 'w').write('\n'.join(log))
