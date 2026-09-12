"""
run_coldstart.py
================
Cold-start generalisation across cyclodextrin families.

Leave-one-CD-family-out: train the multi-task CD-BAN on three CD families and
test on the held-out one (alpha / beta / gamma / HP-beta). This asks whether the
model generalises to a cyclodextrin TYPE it never saw during training.

Per held-out family we report:
  - classification AUROC on that family's labelled extremes
  - regression Pearson on that family's fuzzy-zone log10K (rank generalisation)

Compared against the warm (random-split) baseline (AUROC ~0.95, Pearson ~0.65).

Run:  python results/coldstart_cd/run_coldstart.py
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
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models import MolecularGCN, MLPDecoder
from ban import BANLayer
from torch.nn.utils.weight_norm import weight_norm
from dataloader import smiles_to_pyg
from utils import set_seed

OUT = os.path.join(ROOT,'results','coldstart_cd'); os.makedirs(OUT, exist_ok=True)
log=[]; pr=lambda s='': (print(s,flush=True), log.append(str(s)))
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED=42

def fam(h):
    h=str(h).lower()
    if 'alpha' in h: return 'alpha'
    if 'gamma' in h: return 'gamma'
    if 'hp' in h or 'hydroxypropyl' in h: return 'HP-beta'
    if 'beta' in h: return 'beta'
    return 'other'

ext = pd.concat([pd.read_csv(os.path.join(ROOT,'data/binary',f)) for f in
                 ('train.csv','val.csv','test.csv')], ignore_index=True)
ext = ext[ext['label'].isin([0,1])].reset_index(drop=True); ext['fam']=ext['Host'].map(fam)
fz  = pd.read_csv(os.path.join(ROOT,'data/binary/fuzzy.csv')); fz['label']=-1; fz['fam']=fz['Host'].map(fam)
pr("Extremes by family: "+ext['fam'].value_counts().to_dict().__str__())
pr("Fuzzy by family:    "+fz['fam'].value_counts().to_dict().__str__()+"\n")

_C={}
def _pyg(s):
    if s not in _C: _C[s]=smiles_to_pyg(s)
    return _C[s]
class DS(Dataset):
    def __init__(s,df): s.it=[(_pyg(r['SMILES_Guest']),_pyg(r['SMILES_Host']),float(r['label']),float(r['log10K'])) for _,r in df.iterrows()]
    def __len__(s): return len(s.it)
    def __getitem__(s,i): return s.it[i]
def coll(b):
    g,h,y,k=zip(*b)
    return (Batch.from_data_list(list(g)),Batch.from_data_list(list(h)),torch.tensor(y),torch.tensor(k,dtype=torch.float))

class MT(nn.Module):
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
@torch.no_grad()
def predict(model,ds):
    model.eval(); C,R=[],[]
    for g,h,y,k in DataLoader(ds,batch_size=128,shuffle=False,collate_fn=coll):
        cl,rg=model(g.to(dev),h.to(dev)); C.extend(torch.sigmoid(cl).cpu().tolist()); R.extend(rg.cpu().tolist())
    return np.array(C),np.array(R)

EPOCHS=60
allrows=[]
for held in ['alpha','beta','gamma','HP-beta']:
    tr_df = pd.concat([ext[ext.fam!=held], fz[fz.fam!=held]], ignore_index=True)
    te_ext= ext[ext.fam==held]; te_fz=fz[fz.fam==held]
    ds_tr=DS(tr_df); ds_e=DS(te_ext); ds_f=DS(te_fz)
    for seed in range(42,52):
        set_seed(seed)
        model=MT(**cfg).to(dev); opt=torch.optim.Adam(model.parameters(),lr=5e-4)
        bce=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cfg['CLASSIFIER']['POS_WEIGHT']],device=dev)); mse=nn.MSELoss()
        dl=DataLoader(ds_tr,batch_size=64,shuffle=True,collate_fn=coll,drop_last=True)
        for ep in range(EPOCHS):
            model.train()
            for g,h,y,k in dl:
                g,h,y,k=g.to(dev),h.to(dev),y.to(dev),k.to(dev)
                cl,rg=model(g,h); m=(y>=0)
                loss=(bce(cl[m],y[m].float()) if m.any() else 0)+mse(rg,k)
                opt.zero_grad(); loss.backward(); opt.step()
        pc,_=predict(model,ds_e)
        auc=roc_auc_score(te_ext['label'].values,pc) if te_ext['label'].nunique()>1 else float('nan')
        _,rr=predict(model,ds_f); pear=pearsonr(te_fz['log10K'].values,rr)[0] if len(te_fz)>2 else float('nan')
        allrows.append(dict(held_out_CD=held, seed=seed, n_ext=len(te_ext),
                            n_strong=int((te_ext.label==0).sum()), n_fuzzy=len(te_fz),
                            AUROC=round(auc,3), Pearson=round(pear,3)))
    sub=pd.DataFrame([r for r in allrows if r['held_out_CD']==held])
    pr(f"  hold-out {held:<8}: AUROC={sub.AUROC.mean():.3f}±{sub.AUROC.std():.3f}  "
       f"fuzzy Pearson={sub.Pearson.mean():.3f}±{sub.Pearson.std():.3f}  (ext n={len(te_ext)}, strong={int((te_ext.label==0).sum())})")

allres=pd.DataFrame(allrows); allres.to_csv(os.path.join(OUT,'coldstart_cd_allseeds.csv'),index=False)
res=allres.groupby('held_out_CD').agg(
    AUROC=('AUROC','mean'),AUROC_sd=('AUROC','std'),
    Pearson=('Pearson','mean'),Pearson_sd=('Pearson','std'),
    n_ext=('n_ext','first'),n_strong=('n_strong','first')).reset_index()
order={'alpha':0,'beta':1,'gamma':2,'HP-beta':3}; res=res.sort_values('held_out_CD',key=lambda s:s.map(order)).reset_index(drop=True)
res.to_csv(os.path.join(OUT,'coldstart_cd.csv'),index=False)
pr("\n=== COLD-START SUMMARY (leave-one-CD-family-out, 10 seeds 42-51) ===")
pr(res.round(3).to_string(index=False))
pr(f"\n  mean cold-start AUROC = {res.AUROC.mean():.3f} (warm baseline ~0.95)")
pr(f"  mean cold-start Pearson = {res.Pearson.mean():.3f} (warm baseline ~0.65)")

fig,ax=plt.subplots(figsize=(8.5,5.6))
x=np.arange(len(res)); w=0.38
ax.bar(x-w/2,res['AUROC'],w,yerr=res['AUROC_sd'],capsize=4,label='Classification AUROC',color='#c0392b')
ax.bar(x+w/2,res['Pearson'],w,yerr=res['Pearson_sd'],capsize=4,label='Fuzzy K Pearson',color='#2a7f9e')
ax.axhline(0.95,ls='--',c='#c0392b',lw=1,alpha=0.6,label='warm AUROC ~0.95')
ax.axhline(0.65,ls='--',c='#2a7f9e',lw=1,alpha=0.6,label='warm Pearson ~0.65')
for i,r in res.iterrows():
    ax.text(i-w/2,r['AUROC']+r['AUROC_sd']+0.01,f"{r['AUROC']:.2f}",ha='center',fontsize=9,fontweight='bold')
    ax.text(i+w/2,r['Pearson']+r['Pearson_sd']+0.01,f"{r['Pearson']:.2f}",ha='center',fontsize=9,fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels([f"{r.held_out_CD}\n(test n={int(r.n_ext)})" for _,r in res.iterrows()],fontsize=10)
ax.set_ylim(0,1.05); ax.set_ylabel('score'); ax.set_xlabel('held-out CD family (cold start)')
ax.set_title('Cold-start generalisation across cyclodextrin families\n(train on 3 families, test on the unseen one)',fontsize=12,fontweight='bold')
ax.legend(fontsize=9,loc='lower right'); ax.grid(axis='y',ls=':',alpha=0.4)
fig.tight_layout(); fig.savefig(os.path.join(OUT,'coldstart_cd.png'),dpi=600); fig.savefig(os.path.join(OUT,'coldstart_cd.svg')); plt.close()
pr(f"\nSaved: coldstart_cd.png/.svg , coldstart_cd.csv , run_log.txt")
open(os.path.join(OUT,'run_log.txt'),'w').write('\n'.join(log))
