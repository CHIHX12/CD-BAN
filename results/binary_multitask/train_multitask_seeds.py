"""
train_multitask_seeds.py
========================
Multi-seed (42-51) multi-task CD-BAN with linear recalibration of the
regression head.  Confirms the fuzzy-zone K estimate is stable and that a
simple linear recalibration restores R^2 to ~Pearson^2.

Fixed fuzzy split (independent of model seed): 70% train / 15% calib / 15% test.
For each model seed: train -> raw fuzzy R^2/Pearson -> fit linear recal on the
calib split -> recalibrated R^2 on the test split.  Classification AUROC on
data/binary/test.csv.

Run:  python results/binary_multitask/train_multitask_seeds.py
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
log=[]; pr=lambda s='': (print(s,flush=True), log.append(str(s)))
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------- fixed data split ----------
tr = pd.read_csv(os.path.join(ROOT,'data/binary/train.csv'))
te = pd.read_csv(os.path.join(ROOT,'data/binary/test.csv'))
fz = pd.read_csv(os.path.join(ROOT,'data/binary/fuzzy.csv')); fz['label']=-1
fz = fz.sample(frac=1.0, random_state=123).reset_index(drop=True)
n1,n2=int(0.70*len(fz)), int(0.85*len(fz))
fz_tr, fz_cal, fz_te = fz.iloc[:n1], fz.iloc[n1:n2], fz.iloc[n2:]
train_df = pd.concat([tr, fz_tr], ignore_index=True)
pr(f"Train {len(train_df)} | fuzzy calib {len(fz_cal)} | fuzzy test {len(fz_te)} | cls test {len(te)}\n")

_CACHE={}
def _pyg(s):
    if s not in _CACHE: _CACHE[s]=smiles_to_pyg(s)
    return _CACHE[s]
class DS(Dataset):
    def __init__(s,df): s.items=[(_pyg(r['SMILES_Guest']),_pyg(r['SMILES_Host']),float(r['label']),float(r['log10K'])) for _,r in df.iterrows()]
    def __len__(s): return len(s.items)
    def __getitem__(s,i): return s.items[i]
def coll(b):
    g,h,y,k=zip(*b)
    return (Batch.from_data_list(list(g)),Batch.from_data_list(list(h)),torch.tensor(y),torch.tensor(k,dtype=torch.float))

class MultiTask(nn.Module):
    def __init__(s,**c):
        super().__init__()
        s.mg=c['GUEST']['MAX_NODES']; s.mh=c['HOST']['MAX_NODES']
        s.ge=MolecularGCN(c['GUEST']['NODE_IN_FEATS'],c['GUEST']['NODE_IN_EMBEDDING'],c['GUEST']['HIDDEN_LAYERS'])
        s.he=MolecularGCN(c['HOST']['NODE_IN_FEATS'],c['HOST']['NODE_IN_EMBEDDING'],c['HOST']['HIDDEN_LAYERS'])
        s.bcn=weight_norm(BANLayer(c['GUEST']['HIDDEN_LAYERS'][-1],c['HOST']['HIDDEN_LAYERS'][-1],c['DECODER']['IN_DIM'],c['BCN']['HEADS']),name='h_mat',dim=None)
        s.cls=MLPDecoder(c['DECODER']['IN_DIM'],c['DECODER']['HIDDEN_DIM'],c['DECODER']['OUT_DIM'],1)
        s.reg=MLPDecoder(c['DECODER']['IN_DIM'],c['DECODER']['HIDDEN_DIM'],c['DECODER']['OUT_DIM'],1)
    def forward(s,g,h):
        f,_=s.bcn(s.ge(g,s.mg),s.he(h,s.mh)); return s.cls(f).squeeze(1), s.reg(f).squeeze(1)

with open(os.path.join(ROOT,'configs/CDBAN.yaml')) as f: cfg=yaml.safe_load(f)
ds_train=DS(train_df); ds_cal=DS(fz_cal); ds_te_f=DS(fz_te); ds_te_c=DS(te)

@torch.no_grad()
def preds(model, ds):
    model.eval(); C,R=[],[]
    for g,h,y,k in DataLoader(ds,batch_size=128,shuffle=False,collate_fn=coll):
        cl,rg=model(g.to(dev),h.to(dev)); C.extend(torch.sigmoid(cl).cpu().tolist()); R.extend(rg.cpu().tolist())
    return np.array(C),np.array(R)

rows=[]; best=None
for seed in range(42,52):
    set_seed(seed)
    model=MultiTask(**cfg).to(dev); opt=torch.optim.Adam(model.parameters(),lr=5e-4)
    bce=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cfg['CLASSIFIER']['POS_WEIGHT']],device=dev)); mse=nn.MSELoss()
    dl=DataLoader(ds_train,batch_size=64,shuffle=True,collate_fn=coll,drop_last=True)
    for ep in range(80):
        model.train()
        for g,h,y,k in dl:
            g,h,y,k=g.to(dev),h.to(dev),y.to(dev),k.to(dev)
            cl,rg=model(g,h); m=(y>=0)
            loss=(bce(cl[m],y[m].float()) if m.any() else 0)+mse(rg,k)
            opt.zero_grad(); loss.backward(); opt.step()
    # eval
    pc,_=preds(model,ds_te_c); auc=roc_auc_score(te['label'].values,pc)
    _,rc=preds(model,ds_cal); _,rt=preds(model,ds_te_f)
    yk_cal=fz_cal['log10K'].values; yk_te=fz_te['log10K'].values
    a,b=np.polyfit(rc,yk_cal,1)                       # linear recalibration on calib split
    rt_cal=a*rt+b
    pear=pearsonr(yk_te,rt)[0]
    r2_raw=r2_score(yk_te,rt); r2_cal=r2_score(yk_te,rt_cal)
    rmse_cal=np.sqrt(mean_squared_error(yk_te,rt_cal))
    rows.append(dict(seed=seed,AUROC=round(auc,3),Pearson=round(pear,3),
                     R2_raw=round(r2_raw,3),R2_recal=round(r2_cal,3),RMSE_recal=round(rmse_cal,3)))
    pr(f"  seed {seed}: AUROC={auc:.3f}  fuzzy Pearson={pear:.3f}  R²(raw)={r2_raw:.3f}  R²(recal)={r2_cal:.3f}")
    if best is None or pear>best[0]: best=(pear,seed,yk_te,rt_cal,a,b,auc,pc)

res=pd.DataFrame(rows); res.to_csv(os.path.join(OUT,'multitask_seeds.csv'),index=False)
pr("\n=== SUMMARY (seeds 42-51) ===")
for c in ['AUROC','Pearson','R2_raw','R2_recal']:
    pr(f"  {c:<10} {res[c].mean():.3f} ± {res[c].std():.3f}")

# ---------- figure: recalibrated scatter (best seed) + per-seed bars ----------
pear,seed,yk,rt_cal,a,b,auc,pc=best
fig,ax=plt.subplots(1,2,figsize=(13,5.6))
ax[0].scatter(yk,rt_cal,s=12,alpha=0.45,c='#2a7f9e'); ax[0].plot([1.8,4.2],[1.8,4.2],'k--',alpha=0.6)
ax[0].set_xlabel('true log10K (held-out fuzzy)'); ax[0].set_ylabel('predicted log10K (recalibrated)')
ax[0].set_title(f'Regression head + recalibration (seed {seed})\nPearson={pear:.2f}  R²={r2_score(yk,rt_cal):.2f}',fontweight='bold')
ax[0].grid(ls=':',alpha=0.4)
x=np.arange(len(res)); w=0.38
ax[1].bar(x-w/2,res['AUROC'],w,label='Classification AUROC',color='#c0392b')
ax[1].bar(x+w/2,res['Pearson'],w,label='Fuzzy K Pearson',color='#2a7f9e')
ax[1].set_xticks(x); ax[1].set_xticklabels(res['seed']); ax[1].set_ylim(0,1)
ax[1].set_xlabel('seed'); ax[1].set_title('Stability across seeds 42-51',fontweight='bold')
ax[1].legend(fontsize=9,loc='lower right'); ax[1].grid(axis='y',ls=':',alpha=0.4)
fig.suptitle('Multi-task CD-BAN: classification + recalibrated regression (10 seeds)',fontsize=14,fontweight='bold')
fig.tight_layout(rect=[0,0,1,0.95])
fig.savefig(os.path.join(OUT,'multitask_seeds.png'),dpi=600); fig.savefig(os.path.join(OUT,'multitask_seeds.svg')); plt.close()
pr(f"\nSaved: multitask_seeds.png/.svg , multitask_seeds.csv , run_log_seeds.txt")
open(os.path.join(OUT,'run_log_seeds.txt'),'w').write('\n'.join(log))
