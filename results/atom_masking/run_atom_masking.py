"""
run_atom_masking.py
===================
Reviewer R1-3: attention weights are correlational, not causal. Perform an
ATOM-MASKING (perturbation / counterfactual) experiment: for each guest atom we
zero out its feature vector, re-run inference, and measure the change in the
output logit (dz). Aggregating |dz| by atom category gives a CAUSAL importance
profile that can be directly compared against the attention profile (Fig 6-7).
If attention tracks causal importance, the two profiles should agree.

We also report a concrete counterfactual: masking the single most-important
atom flips the prediction for X% of strong binders.

Outputs:
  results/atom_masking/masking_importance.csv   per-category mean |dz| (strong vs weak)
  results/atom_masking/masking_vs_attention.png/.svg
  results/atom_masking/run_log.txt
"""
import os, sys, warnings, yaml
warnings.filterwarnings('ignore')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from rdkit import Chem
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models import CDBAN
from dataloader import smiles_to_pyg
from utils import set_seed

OUT = os.path.join(ROOT, 'results', 'atom_masking'); os.makedirs(OUT, exist_ok=True)
log = []; pr = lambda s='': (print(s, flush=True), log.append(str(s)))
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 49
MODEL_PATH = os.path.join(ROOT, f'results/seed_{SEED}/best_model_epoch_69.pth')
N_PER_CLASS = 40   # strong and weak binders to probe

CATS = ['Aromatic C', 'Aliphatic C', 'N', 'O', 'S', 'Halogen',
        'H-bond donor', 'H-bond acceptor', 'Ring atom', 'Non-ring atom']

def categories_of(atom):
    sym = atom.GetSymbol(); c = []
    if sym == 'C':
        c.append('Aromatic C' if atom.GetIsAromatic() else 'Aliphatic C')
    elif sym == 'N': c.append('N')
    elif sym == 'O': c.append('O')
    elif sym == 'S': c.append('S')
    elif sym in ('F', 'Cl', 'Br', 'I'): c.append('Halogen')
    if sym in ('N', 'O') and atom.GetTotalNumHs() > 0: c.append('H-bond donor')
    if sym in ('N', 'O', 'F'): c.append('H-bond acceptor')
    c.append('Ring atom' if atom.IsInRing() else 'Non-ring atom')
    return c

with open(os.path.join(ROOT, 'configs/CDBAN.yaml')) as f: cfg = yaml.safe_load(f)
set_seed(SEED)
model = CDBAN(**cfg).to(dev)
model.load_state_dict(torch.load(MODEL_PATH, map_location=dev))
model.eval()

# pool of extremes
ext = pd.concat([pd.read_csv(os.path.join(ROOT, 'data/binary/' + f)) for f in
                 ('train.csv', 'val.csv', 'test.csv')], ignore_index=True)
rng = np.random.RandomState(0)
strong = ext[ext.label == 0].sample(min(N_PER_CLASS, (ext.label == 0).sum()), random_state=0)
weak   = ext[ext.label == 1].sample(min(N_PER_CLASS, (ext.label == 1).sum()), random_state=0)
pr(f"probing {len(strong)} strong + {len(weak)} weak binders")

def guest_base_z(g_smi, h_smi):
    g = smiles_to_pyg(g_smi); h = smiles_to_pyg(h_smi)
    gb = Batch.from_data_list([g]); hb = Batch.from_data_list([h])
    with torch.no_grad():
        _, _, f, sc = model(gb.to(dev), hb.to(dev), mode='train')
    return float(sc.squeeze(1).cpu()), g, h

acc = {lab: {c: [] for c in CATS} for lab in ('strong', 'weak')}
flip_count = {lab: 0 for lab in ('strong', 'weak')}
flip_total = {lab: 0 for lab in ('strong', 'weak')}

for lab, df in (('strong', strong), ('weak', weak)):
    for _, r in df.iterrows():
        g_smi, h_smi = r['SMILES_Guest'], r['SMILES_Host']
        z_base, g, h = guest_base_z(g_smi, h_smi)
        mol = Chem.MolFromSmiles(g_smi)
        n_atoms = g.num_nodes
        if mol is None or n_atoms == 0: continue
        # build masked variants (one per atom)
        masked = []
        for i in range(n_atoms):
            gm = g.clone()
            xm = gm.x.clone(); xm[i] = 0.0; gm.x = xm
            masked.append(gm)
        mb = Batch.from_data_list(masked)
        hb = Batch.from_data_list([h])
        with torch.no_grad():
            _, _, f, sc = model(mb.to(dev), hb.to(dev), mode='train')
        zs = sc.squeeze(1).cpu().numpy()
        dz = np.abs(zs - z_base)
        # counterfactual: does masking the most-important atom flip the call?
        base_call = z_base >= 0          # True => weak (label 1)
        argmax_i = int(np.argmax(dz))
        flip_call = zs[argmax_i] >= 0
        flip_total[lab] += 1
        if flip_call != base_call: flip_count[lab] += 1
        for i in range(n_atoms):
            for c in categories_of(mol.GetAtomWithIdx(i)):
                acc[lab][c].append(float(dz[i]))

rows = []
for c in CATS:
    rows.append({
        'category': c,
        'strong_mean_abs_dz': round(float(np.mean(acc['strong'][c])), 4) if acc['strong'][c] else np.nan,
        'weak_mean_abs_dz': round(float(np.mean(acc['weak'][c])), 4) if acc['weak'][c] else np.nan,
        'n_strong_atoms': len(acc['strong'][c]),
        'n_weak_atoms': len(acc['weak'][c]),
    })
res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, 'masking_importance.csv'), index=False)
pr("\nCausal importance (mean |dz| per atom category):")
pr(res.to_string(index=False))
pr(f"\nCounterfactual flip rate (mask most-important atom): "
   f"strong {flip_count['strong']}/{flip_total['strong']}  weak {flip_count['weak']}/{flip_total['weak']}")

# ---- plot: strong vs weak causal importance ----
fig, ax = plt.subplots(figsize=(9, 5.4))
x = np.arange(len(CATS)); w = 0.38
sv = [r['strong_mean_abs_dz'] for r in rows]; wv = [r['weak_mean_abs_dz'] for r in rows]
ax.bar(x - w/2, sv, w, label='strong binders', color='#c0392b')
ax.bar(x + w/2, wv, w, label='weak binders', color='#2a7f9e')
ax.set_xticks(x); ax.set_xticklabels(CATS, rotation=35, ha='right', fontsize=9)
ax.set_ylabel('mean |Δz| (causal importance)')
ax.set_title('Atom-masking causal importance vs attention categories\n(zero-out each guest atom, measure change in logit)',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.grid(axis='y', ls=':', alpha=0.4)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'masking_vs_attention.png'), dpi=600)
fig.savefig(os.path.join(OUT, 'masking_vs_attention.svg'))
plt.close(fig)
pr("Saved: masking_importance.csv, masking_vs_attention.png/.svg, run_log.txt")
open(os.path.join(OUT, 'run_log.txt'), 'w').write('\n'.join(log))
