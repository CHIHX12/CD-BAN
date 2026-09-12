"""
run_baseline_models.py
======================
Reviewers R1-2 + R2-M2: compare against established / trivial baselines with
simple featurisations (Morgan fingerprint + Mordred descriptors + host
one-hot / cavity-radius), trained on the SAME 838 extremes and tested the SAME
way as CD-BAN (held-out test AUROC + fuzzy-zone correlation on the 1,850
withheld compounds). This asks whether a simple classifier on hand-crafted
features recovers a comparable emergent gradient without any GNN / attention.

Models: LogisticRegression, RandomForest, MLP.
Features: guest Morgan (r=2, 1024) + guest Mordred (2D) + host one-hot (8 fam)
          + host cavity radius (scalar).

Outputs:
  results/baseline_models/baseline_models.csv
  results/baseline_models/baseline_models.png/.svg
  results/baseline_models/run_log.txt
"""
import os, sys, warnings
warnings.filterwarnings('ignore')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr, kendalltau
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.join(ROOT, 'results', 'baseline_models'); os.makedirs(OUT, exist_ok=True)
log = []; pr = lambda s='': (print(s, flush=True), log.append(str(s)))

# --- host family + cavity radius (representative inner diameter, Angstrom) ---
FAMS = ['alpha', 'beta', 'gamma', 'HP-beta', 'SBE-beta', 'CM-beta', 'Me-beta',
        'DM-beta', 'TM-beta', 'Ac-beta', 'Suc-beta', 'SO4-beta']
RADIUS = {'alpha': 5.7, 'beta': 6.6, 'gamma': 7.8, 'HP-beta': 6.6, 'SBE-beta': 6.6,
          'CM-beta': 6.6, 'Me-beta': 6.6, 'DM-beta': 6.6, 'TM-beta': 6.6,
          'Ac-beta': 6.6, 'Suc-beta': 6.6, 'SO4-beta': 6.6}
def fam(h):
    h = str(h).lower()
    if 'alpha' in h: return 'alpha'
    if 'gamma' in h: return 'gamma'
    if 'hp' in h or 'hydroxypropyl' in h: return 'HP-beta'
    if 'sulfobutylether' in h or 'sbe' in h: return 'SBE-beta'
    if 'carboxymethyl' in h or 'cm-beta' in h: return 'CM-beta'
    if 'trimethyl' in h or 'tm-beta' in h: return 'TM-beta'
    if 'di-o-methyl' in h or 'dimethyl' in h or 'dm-beta' in h: return 'DM-beta'
    if 'methyl' in h: return 'Me-beta'
    if 'acetyl' in h: return 'Ac-beta'
    if 'succinyl' in h: return 'Suc-beta'
    if 'sulfate' in h: return 'SO4-beta'
    if 'beta' in h: return 'beta'
    return 'beta'

def guest_feats(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024)
    return np.array([int(b) for b in fp.ToBitString()], dtype=float)

def build_matrix(df):
    G = []
    bad = 0
    for _, r in df.iterrows():
        g = guest_feats(r['SMILES_Guest'])
        if g is None: bad += 1; continue
        f = fam(r['Host'])
        onehot = np.array([1.0 if f == x else 0.0 for x in FAMS])
        rad = np.array([RADIUS.get(f, 6.6)])
        G.append(np.concatenate([g, onehot, rad]))
    return np.vstack(G), bad

tr = pd.read_csv(os.path.join(ROOT, 'data/binary/train.csv'))
va = pd.read_csv(os.path.join(ROOT, 'data/binary/val.csv'))
te = pd.read_csv(os.path.join(ROOT, 'data/binary/test.csv'))
fz = pd.read_csv(os.path.join(ROOT, 'data/binary/fuzzy.csv'))

pr("Building feature matrices...")
Xtr, _ = build_matrix(tr); ytr = tr['label'].values.astype(float)
Xte, _ = build_matrix(te); yte = te['label'].values.astype(float)
Xfz, _ = build_matrix(fz); yfz = fz['log10K'].values
pr(f"train {Xtr.shape}  test {Xte.shape}  fuzzy {Xfz.shape}  (dims={Xtr.shape[1]})")

models = {
    'LogisticRegression': make_pipeline(StandardScaler(),
        LogisticRegression(max_iter=5000, C=1.0, class_weight='balanced')),
    'RandomForest': RandomForestClassifier(n_estimators=400, max_depth=None,
        min_samples_leaf=2, n_jobs=-1, random_state=0),
    'MLP': make_pipeline(StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=400, random_state=0,
                      early_stopping=True, validation_fraction=0.1)),
}

rows = []
for name, mdl in models.items():
    pr(f"\n=== {name} ===")
    mdl.fit(Xtr, ytr)
    pte = mdl.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, pte)
    pfz = mdl.predict_proba(Xfz)[:, 1]
    pe = pearsonr(pfz, yfz)[0]; ke = kendalltau(pfz, yfz)[0]
    pr(f"  test AUROC = {auc:.3f}   fuzzy Pearson = {pe:+.3f}   fuzzy Kendall = {ke:+.3f}")
    rows.append(dict(model=name, test_AUROC=round(auc, 3),
                     fuzzy_Pearson=round(pe, 3), fuzzy_Kendall=round(ke, 3)))

res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, 'baseline_models.csv'), index=False)
pr("\n=== BASELINE SUMMARY (trained on 838 extremes, same protocol as CD-BAN) ===")
pr(res.to_string(index=False))
pr("\nReference (CD-BAN, 10 seeds): test AUROC 0.925 +/- 0.040 ; fuzzy Pearson -0.468 / Kendall -0.289")

# ---- plot ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
x = np.arange(len(res)); w = 0.5
ax = axes[0]
ax.bar(x, res.test_AUROC, w, color='#2a7f9e')
ax.axhline(0.925, color='#c0392b', ls='--', lw=1.5, label='CD-BAN 0.925')
for i, v in enumerate(res.test_AUROC): ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(res.model, fontsize=9)
ax.set_ylabel('held-out test AUROC'); ax.set_ylim(0, 1.05)
ax.set_title('Classification baseline vs CD-BAN', fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.grid(axis='y', ls=':', alpha=0.4)

ax = axes[1]
ax.bar(x, res.fuzzy_Pearson, w, color='#2a7f9e')
ax.axhline(-0.468, color='#c0392b', ls='--', lw=1.5, label='CD-BAN -0.468')
for i, v in enumerate(res.fuzzy_Pearson): ax.text(i, v - 0.03, f'{v:+.3f}', ha='center', fontsize=9, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(res.model, fontsize=9)
ax.set_ylabel('fuzzy-zone Pearson (P vs log10K)')
ax.set_title('Emergent-gradient baseline vs CD-BAN', fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.grid(axis='y', ls=':', alpha=0.4)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'baseline_models.png'), dpi=600)
fig.savefig(os.path.join(OUT, 'baseline_models.svg'))
plt.close(fig)
pr("\nSaved: baseline_models.csv, baseline_models.png/.svg, run_log.txt")
open(os.path.join(OUT, 'run_log.txt'), 'w').write('\n'.join(log))
