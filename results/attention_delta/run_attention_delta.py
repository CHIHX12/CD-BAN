"""
run_attention_delta.py
======================
Class-conditional attention "delta" analysis.

Idea: aggregate each compound's BANLayer attention onto interpretable GUEST
(drug) atom categories, average within strong (label 0) and weak (label 1),
then look at the DELTA = strong - weak to find the attention signature that
best separates the two classes. Finally fit a logistic "formula" on those
attention features and report its accuracy.

Output figure: 3 panels  [ Weak (1) | Strong (0) | Strong - Weak ].

Run:  python results/attention_delta/run_attention_delta.py
"""
import os, sys, warnings, yaml
warnings.filterwarnings('ignore')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

import numpy as np, pandas as pd, torch
from rdkit import Chem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import cross_val_predict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models import CDBAN
from dataloader import smiles_to_pyg
from torch_geometric.data import Batch

OUT = os.path.join(ROOT, 'results', 'attention_delta')
os.makedirs(OUT, exist_ok=True)
log_lines = []
def log(s=''):
    print(s); log_lines.append(str(s))

# ---- interpretable guest-atom categories ----
CATS = ['Aromatic C', 'Aliphatic C', 'N', 'O', 'S', 'Halogen',
        'H-bond donor', 'H-bond acceptor', 'Ring atom', 'Non-ring atom']

def atom_cats(mol):
    """boolean membership matrix [n_atoms, n_cats]."""
    M = np.zeros((mol.GetNumAtoms(), len(CATS)))
    for a in mol.GetAtoms():
        i = a.GetIdx(); sym = a.GetSymbol(); arom = a.GetIsAromatic()
        nH = a.GetTotalNumHs()
        M[i,0] = sym == 'C' and arom
        M[i,1] = sym == 'C' and not arom
        M[i,2] = sym == 'N'
        M[i,3] = sym == 'O'
        M[i,4] = sym == 'S'
        M[i,5] = sym in ('F','Cl','Br','I')
        M[i,6] = sym in ('N','O') and nH > 0          # H-bond donor
        M[i,7] = sym in ('N','O')                      # H-bond acceptor
        M[i,8] = a.IsInRing()
        M[i,9] = not a.IsInRing()
    return M

# ---- data ----
df = pd.concat([pd.read_csv(os.path.join(ROOT,'data/binary',f))
                for f in ('train.csv','val.csv','test.csv')], ignore_index=True)
df = df[df['label'].isin([0,1])].reset_index(drop=True)
log(f"Pairs: {len(df)}  | strong(0)={int((df.label==0).sum())}  weak(1)={int((df.label==1).sum())}")

with open(os.path.join(ROOT,'configs/CDBAN.yaml')) as f: cfg = yaml.safe_load(f)
model = CDBAN(**cfg)
model.load_state_dict(torch.load(os.path.join(ROOT,'results/seed_49/best_model_epoch_69.pth'),
                                 map_location='cpu'))
model.eval()

# ---- per-compound guest attention profile (fraction of attention mass per category) ----
def softmax2d(x):
    x = x - x.max(); e = np.exp(x); return e / e.sum()

profiles, labels = [], []
with torch.no_grad():
    for _, r in df.iterrows():
        mol = Chem.MolFromSmiles(r['SMILES_Guest'])
        if mol is None: continue
        g = smiles_to_pyg(r['SMILES_Guest']); h = smiles_to_pyg(r['SMILES_Host'])
        n_g, n_h = g.x.shape[0], h.x.shape[0]
        bg = Batch.from_data_list([g]); bh = Batch.from_data_list([h])
        _, att = model.bcn(model.guest_extractor(bg, model.max_guest),
                           model.host_extractor(bh, model.max_host))
        a = att[0].numpy()[:, :n_g, :n_h]              # [2, n_g, n_h]
        gmass = np.zeros(n_g)
        for hh in range(a.shape[0]):
            p = softmax2d(a[hh])                        # distribution over all cells
            gmass += p.sum(axis=1)                      # mass per guest atom
        gmass /= gmass.sum() + 1e-9                     # normalise over guest atoms
        C = atom_cats(mol)[:n_g]                        # [n_g, n_cats]
        profiles.append(gmass @ C)                      # fraction of attention per category
        labels.append(int(r['label']))
X = np.array(profiles); y = np.array(labels)
log(f"Attention profiles: {X.shape}  ({len(CATS)} interpretable categories)\n")

# ---- class means + delta ----
weak = X[y==1].mean(0); strong = X[y==0].mean(0); delta = strong - weak
order = np.argsort(np.abs(delta))[::-1]
log("Most discriminative attention categories (|strong - weak|):")
for k in order:
    log(f"  {CATS[k]:<16} weak={weak[k]:.3f}  strong={strong[k]:.3f}  delta={delta[k]:+.3f}")

# ---- derive a 'formula': logistic regression on attention profile ----
clf = LogisticRegression(class_weight='balanced', max_iter=2000)
proba = cross_val_predict(clf, X, y, cv=5, method='predict_proba')[:,1]  # P(weak)
auc = roc_auc_score(y, proba)          # AUROC for separating weak vs strong
clf.fit(X, y)
bal = balanced_accuracy_score(y, clf.predict(X))
log(f"\nAttention-only logistic formula (5-fold CV):  AUROC = {auc:.3f}   balanced-acc = {bal:.3f}")
log("Formula weights (log-odds of WEAK per unit attention-fraction):")
for k in np.argsort(np.abs(clf.coef_[0]))[::-1]:
    log(f"  {CATS[k]:<16} w = {clf.coef_[0][k]:+.2f}")

# ---- figure: 3 panels [Weak | Strong | Delta] ----
ypos = np.arange(len(CATS))[::-1]
fig, ax = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
ax[0].barh(ypos, weak[ : ], color='#2a7f9e'); ax[0].set_title('Weak (label 1)', fontsize=14, fontweight='bold')
ax[1].barh(ypos, strong[:], color='#c0392b'); ax[1].set_title('Strong (label 0)', fontsize=14, fontweight='bold')
cols = ['#c0392b' if d>0 else '#2a7f9e' for d in delta]
ax[2].barh(ypos, delta, color=cols); ax[2].axvline(0, color='k', lw=0.8)
ax[2].set_title('Delta = Strong - Weak', fontsize=14, fontweight='bold')
for a_ in ax:
    a_.set_yticks(ypos); a_.set_yticklabels(CATS, fontsize=11); a_.grid(axis='x', ls=':', alpha=0.4)
ax[0].set_xlabel('mean attention fraction'); ax[1].set_xlabel('mean attention fraction')
ax[2].set_xlabel('Δ attention (red = more in STRONG)')
fig.suptitle(f'CD-BAN guest-atom attention signature  (AUROC of attention-only formula = {auc:.3f})',
             fontsize=15, fontweight='bold')
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig(os.path.join(OUT,'attention_delta.png'), dpi=600)
fig.savefig(os.path.join(OUT,'attention_delta.svg'))
plt.close()

pd.DataFrame({'category':CATS,'weak':weak,'strong':strong,'delta':delta,
              'logit_weight':clf.coef_[0]}).to_csv(os.path.join(OUT,'attention_delta.csv'), index=False)
log(f"\nSaved: attention_delta.png/.svg , attention_delta.csv , run_log.txt")
with open(os.path.join(OUT,'run_log.txt'),'w') as f: f.write('\n'.join(log_lines))
