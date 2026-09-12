"""
run_emergence.py
===============
Controlled test of the EMERGENCE claim: a classifier trained only on the binary
extremes (never on the fuzzy zone) still ranks fuzzy-zone compounds by affinity.

To prove this is a genuine, task-driven effect (not an architecture / data
artifact) we compare four conditions and measure the fuzzy-zone rank correlation
(Kendall tau, Spearman, Pearson) of the model output vs true log10K on ALL 1,850
fuzzy compounds (none used in training):

  1. RANDOM   : untrained, random weights              -> null baseline
  2. SHUFFLED : trained on extremes with SHUFFLED labels-> null (data seen, task destroyed)
  3. REAL     : trained on extremes with real labels    -> the emergence
  4. (ref) supervised regression trained WITH fuzzy     -> ceiling (~0.65)

If REAL >> RANDOM, SHUFFLED, the fuzzy ordering emerges from learning the real
binary task, with zero fuzzy supervision.

Run:  python results/emergence_test/run_emergence.py
"""
import os, sys, warnings, yaml
warnings.filterwarnings('ignore')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from scipy.stats import pearsonr, spearmanr, kendalltau
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models import CDBAN
from dataloader import smiles_to_pyg
from utils import set_seed

OUT = os.path.join(ROOT,'results','emergence_test'); os.makedirs(OUT, exist_ok=True)
log=[]; pr=lambda s='': (print(s,flush=True), log.append(str(s)))
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED=42

tr = pd.read_csv(os.path.join(ROOT,'data/binary/train.csv'))   # extremes only
fz = pd.read_csv(os.path.join(ROOT,'data/binary/fuzzy.csv'))   # never in training
pr(f"Train extremes: {len(tr)} | fuzzy (test, all unseen): {len(fz)}\n")

_C={}
def _pyg(s):
    if s not in _C: _C[s]=smiles_to_pyg(s)
    return _C[s]
class DS(Dataset):
    def __init__(s,g,h,y): s.g=[_pyg(x) for x in g]; s.h=[_pyg(x) for x in h]; s.y=list(y)
    def __len__(s): return len(s.y)
    def __getitem__(s,i): return s.g[i], s.h[i], float(s.y[i])
def coll(b):
    g,h,y=zip(*b); return Batch.from_data_list(list(g)),Batch.from_data_list(list(h)),torch.tensor(y)

with open(os.path.join(ROOT,'configs/CDBAN.yaml')) as f: cfg=yaml.safe_load(f)

def train_clf(labels):
    set_seed(SEED)
    model=CDBAN(**cfg).to(dev); opt=torch.optim.Adam(model.parameters(),lr=5e-4)
    bce=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cfg['CLASSIFIER']['POS_WEIGHT']],device=dev))
    dl=DataLoader(DS(tr['SMILES_Guest'].values,tr['SMILES_Host'].values,labels),
                  batch_size=64,shuffle=True,collate_fn=coll,drop_last=True)
    for ep in range(80):
        model.train()
        for g,h,y in dl:
            g,h,y=g.to(dev),h.to(dev),y.float().to(dev)
            _,_,_,sc=model(g,h,mode='train')
            loss=bce(sc.squeeze(1),y); opt.zero_grad(); loss.backward(); opt.step()
    return model

@torch.no_grad()
def fuzzy_z(model):
    model.eval(); Z=[]
    dl=DataLoader(DS(fz['SMILES_Guest'].values,fz['SMILES_Host'].values,np.zeros(len(fz))),
                  batch_size=128,shuffle=False,collate_fn=coll)
    for g,h,y in dl:
        _,_,sc,_=model(g.to(dev),h.to(dev),mode='eval'); Z.extend(sc.squeeze(1).cpu().tolist())
    return np.array(Z)

yk=fz['log10K'].values
def corr(z,name):
    pe=pearsonr(z,yk)[0]; sp=spearmanr(z,yk)[0]; ta=kendalltau(z,yk)[0]
    pr(f"  {name:<28} Pearson={pe:+.3f}  Spearman={sp:+.3f}  Kendall_tau={ta:+.3f}")
    return dict(condition=name,pearson=round(pe,3),spearman=round(sp,3),kendall=round(ta,3))

pr("Fuzzy-zone ranking (model output z vs true log10K, n=1850):")
rows=[]
set_seed(SEED); rand=CDBAN(**cfg).to(dev)                 # 1. random init
rows.append(corr(fuzzy_z(rand),"1. RANDOM (untrained)"))
rows.append(corr(fuzzy_z(train_clf(np.random.RandomState(0).permutation(tr['label'].values))),
                 "2. SHUFFLED labels"))                    # 2. shuffled
rows.append(corr(fuzzy_z(train_clf(tr['label'].values)),  # 3. real -> emergence
                 "3. REAL labels (emergence)"))
rows.append(dict(condition="4. Supervised reg (with fuzzy)",pearson=0.65,spearman=np.nan,kendall=np.nan))
pr("  4. Supervised reg (with fuzzy)  Pearson=+0.650  (ceiling, from multitask)")

res=pd.DataFrame(rows); res.to_csv(os.path.join(OUT,'emergence_test.csv'),index=False)

# bar chart: |Kendall tau| (use |pearson| for the ref ceiling)
fig,ax=plt.subplots(figsize=(8.5,5.6))
labs=[r['condition'] for r in rows]
vals=[abs(rows[0]['kendall']),abs(rows[1]['kendall']),abs(rows[2]['kendall']),np.nan]
valsP=[abs(r['pearson']) for r in rows]
x=np.arange(4); w=0.38
ax.bar(x-w/2,[abs(rows[i]['kendall']) if not np.isnan(rows[i]['kendall']) else 0 for i in range(4)],w,label='|Kendall tau|',color='#2a7f9e')
ax.bar(x+w/2,valsP,w,label='|Pearson|',color='#c0392b')
for i in range(4):
    if not np.isnan(rows[i]['kendall']): ax.text(i-w/2,abs(rows[i]['kendall'])+0.01,f"{abs(rows[i]['kendall']):.2f}",ha='center',fontsize=9)
    ax.text(i+w/2,valsP[i]+0.01,f"{valsP[i]:.2f}",ha='center',fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(['Random\n(untrained)','Shuffled\nlabels','REAL\n(emergence)','Supervised\n(+fuzzy)'],fontsize=10)
ax.set_ylabel('|fuzzy-zone rank correlation|'); ax.set_ylim(0,0.8)
ax.set_title('Emergence test: fuzzy-zone ordering vs controls\n(classifier never trained on fuzzy)',fontsize=12,fontweight='bold')
ax.legend(loc='upper left',fontsize=10); ax.grid(axis='y',ls=':',alpha=0.4)
fig.tight_layout(); fig.savefig(os.path.join(OUT,'emergence_test.png'),dpi=600); fig.savefig(os.path.join(OUT,'emergence_test.svg')); plt.close()

pr("\n=== VERDICT ===")
real=abs(rows[2]['kendall']); null=max(abs(rows[0]['kendall']),abs(rows[1]['kendall']))
pr(f"  REAL |tau|={real:.3f}  vs  null(random/shuffled) |tau|<={null:.3f}")
pr("  -> emergence is GENUINE" if real>3*max(null,0.02) else "  -> emergence NOT clearly above null")
pr(f"\nSaved: emergence_test.png/.svg , emergence_test.csv , run_log.txt")
open(os.path.join(OUT,'run_log.txt'),'w').write('\n'.join(log))
