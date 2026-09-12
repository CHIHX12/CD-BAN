"""
train_multitask.py
==================
A fresh multi-task CD-BAN (seed 42) that does BOTH:
  - classification : P(weak) on the labelled extremes  (BCE)
  - regression     : log10K directly, across the full range  (MSE)

Motivation: the classifier + hand calibration formula gives only a weak K
estimate (R^2 ~ 0.18). A dedicated regression head trained on real log10K
(extremes + fuzzy zone) should estimate K far better, with no formula.

Data:
  regression target log10K : train.csv (extremes) + 80% of fuzzy.csv
  classification label     : labelled extremes only
  held-out test            : test.csv (extremes) + 20% of fuzzy.csv

Run:  python results/binary_multitask/train_multitask.py
"""
import os, sys, warnings, yaml
warnings.filterwarnings('ignore')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error
from scipy.stats import pearsonr
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models import MolecularGCN, MLPDecoder
from ban import BANLayer
from torch.nn.utils.weight_norm import weight_norm
from dataloader import smiles_to_pyg
from utils import set_seed

OUT = os.path.join(ROOT,'results','binary_multitask'); os.makedirs(OUT, exist_ok=True)
log=[]; pr=lambda s='': (print(s), log.append(str(s)))
SEED=42; set_seed(SEED)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------- data ----------
tr = pd.read_csv(os.path.join(ROOT,'data/binary/train.csv'))
te = pd.read_csv(os.path.join(ROOT,'data/binary/test.csv'))
fz = pd.read_csv(os.path.join(ROOT,'data/binary/fuzzy.csv')); fz['label']=-1
fz = fz.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
ncut = int(0.8*len(fz)); fz_tr, fz_te = fz.iloc[:ncut], fz.iloc[ncut:]
train_df = pd.concat([tr, fz_tr], ignore_index=True)
pr(f"Train: {len(train_df)} (extremes {len(tr)} + fuzzy {len(fz_tr)})")
pr(f"Test : extremes {len(te)} (classification) + fuzzy {len(fz_te)} (regression)\n")

_CACHE={}
def _pyg(smi):
    if smi not in _CACHE: _CACHE[smi]=smiles_to_pyg(smi)
    return _CACHE[smi]
class DS(Dataset):
    def __init__(s, df):
        s.df=df.reset_index(drop=True)
        s.items=[( _pyg(r['SMILES_Guest']), _pyg(r['SMILES_Host']),
                   float(r['label']), float(r['log10K']) ) for _,r in s.df.iterrows()]
    def __len__(s): return len(s.items)
    def __getitem__(s, i): return s.items[i]
def coll(b):
    g,h,y,k=zip(*b)
    return (Batch.from_data_list(list(g)), Batch.from_data_list(list(h)),
            torch.tensor(y), torch.tensor(k,dtype=torch.float))

# ---------- multi-task model ----------
class MultiTask(nn.Module):
    def __init__(s, **c):
        super().__init__()
        s.mg=c['GUEST']['MAX_NODES']; s.mh=c['HOST']['MAX_NODES']
        s.ge=MolecularGCN(c['GUEST']['NODE_IN_FEATS'],c['GUEST']['NODE_IN_EMBEDDING'],c['GUEST']['HIDDEN_LAYERS'])
        s.he=MolecularGCN(c['HOST']['NODE_IN_FEATS'],c['HOST']['NODE_IN_EMBEDDING'],c['HOST']['HIDDEN_LAYERS'])
        s.bcn=weight_norm(BANLayer(c['GUEST']['HIDDEN_LAYERS'][-1],c['HOST']['HIDDEN_LAYERS'][-1],
                                   c['DECODER']['IN_DIM'],c['BCN']['HEADS']),name='h_mat',dim=None)
        s.cls=MLPDecoder(c['DECODER']['IN_DIM'],c['DECODER']['HIDDEN_DIM'],c['DECODER']['OUT_DIM'],1)
        s.reg=MLPDecoder(c['DECODER']['IN_DIM'],c['DECODER']['HIDDEN_DIM'],c['DECODER']['OUT_DIM'],1)
    def forward(s, g, h):
        f,_=s.bcn(s.ge(g,s.mg), s.he(h,s.mh))
        return s.cls(f).squeeze(1), s.reg(f).squeeze(1)

with open(os.path.join(ROOT,'configs/CDBAN.yaml')) as f: cfg=yaml.safe_load(f)
model=MultiTask(**cfg).to(dev)
opt=torch.optim.Adam(model.parameters(), lr=5e-4)
pos_w=torch.tensor([cfg['CLASSIFIER']['POS_WEIGHT']], device=dev)
bce=nn.BCEWithLogitsLoss(pos_weight=pos_w); mse=nn.MSELoss()

tl=DataLoader(DS(train_df),batch_size=64,shuffle=True,collate_fn=coll,drop_last=True)
EPOCHS=80; LAMBDA=1.0
pr(f"Training multi-task (seed {SEED}, {EPOCHS} epochs, device={dev})...")
for ep in range(EPOCHS):
    model.train(); tot=0
    for g,h,y,k in tl:
        g,h,y,k=g.to(dev),h.to(dev),y.to(dev),k.to(dev)
        clog,reg=model(g,h)
        m=(y>=0)                                   # labelled mask
        lc=bce(clog[m], y[m].float()) if m.any() else torch.tensor(0.,device=dev)
        lr=mse(reg, k)                             # regression on all
        loss=lc+LAMBDA*lr
        opt.zero_grad(); loss.backward(); opt.step(); tot+=loss.item()
    if (ep+1)%20==0: pr(f"  epoch {ep+1:3d}  loss={tot/len(tl):.4f}")

# ---------- evaluation ----------
@torch.no_grad()
def predict(df):
    model.eval(); C,R=[],[]
    dl=DataLoader(DS(df),batch_size=128,shuffle=False,collate_fn=coll)
    for g,h,y,k in dl:
        clog,reg=model(g.to(dev),h.to(dev))
        C.extend(torch.sigmoid(clog).cpu().tolist()); R.extend(reg.cpu().tolist())
    return np.array(C), np.array(R)

# classification on test extremes
pc,_=predict(te); auc=roc_auc_score(te['label'].values, pc)
# regression on held-out fuzzy
_,pr_fz=predict(fz_te); yk=fz_te['log10K'].values
r2=r2_score(yk,pr_fz); rmse=np.sqrt(mean_squared_error(yk,pr_fz)); pear=pearsonr(yk,pr_fz)[0]
# regression on test extremes too
_,pr_te=predict(te)

pr(f"\n=== RESULTS (seed {SEED}) ===")
pr(f"  CLASSIFICATION (test extremes, n={len(te)}):  AUROC = {auc:.3f}")
pr(f"  REGRESSION (held-out fuzzy, n={len(fz_te)}):   R² = {r2:.3f}  RMSE = {rmse:.3f}  Pearson = {pear:.3f}")
pr(f"     (old classifier+formula on fuzzy was: Pearson r ~ -0.42, R² ~ 0.18)")

# ---------- plot ----------
fig,ax=plt.subplots(1,2,figsize=(13,5.6))
ax[0].scatter(yk,pr_fz,s=10,alpha=0.4,c='#2a7f9e')
lim=[1.8,4.2]; ax[0].plot(lim,lim,'k--',alpha=0.6)
ax[0].set_xlabel('true log10K (held-out fuzzy)'); ax[0].set_ylabel('predicted log10K')
ax[0].set_title(f'Regression head — fuzzy zone\nR²={r2:.2f}  Pearson={pear:.2f}  RMSE={rmse:.2f}',fontweight='bold')
ax[0].grid(ls=':',alpha=0.4)
from sklearn.metrics import roc_curve
fpr,tpr,_=roc_curve(te['label'].values,pc)
ax[1].plot(fpr,tpr,lw=2.5,c='#c0392b',label=f'AUROC={auc:.3f}'); ax[1].plot([0,1],[0,1],'k--',alpha=0.5)
ax[1].set_xlabel('FPR'); ax[1].set_ylabel('TPR'); ax[1].set_title('Classification head — test',fontweight='bold')
ax[1].legend(loc='lower right'); ax[1].grid(ls=':',alpha=0.4)
fig.suptitle(f'Multi-task CD-BAN (classification + regression, seed {SEED})',fontsize=14,fontweight='bold')
fig.tight_layout(rect=[0,0,1,0.95])
fig.savefig(os.path.join(OUT,'multitask_results.png'),dpi=600); fig.savefig(os.path.join(OUT,'multitask_results.svg')); plt.close()

torch.save(model.state_dict(), os.path.join(OUT,f'multitask_seed{SEED}.pth'))
pr(f"\nSaved: multitask_results.png/.svg , multitask_seed{SEED}.pth , run_log.txt")
open(os.path.join(OUT,'run_log.txt'),'w').write('\n'.join(log))
