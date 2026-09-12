"""
run_similarity.py
=================
Reviewer R2-minor5: how "similar" are the unseen external guests to the
training guests? Compute Morgan-fingerprint Tanimoto similarity of each of the
8 external guests against every training guest, and report max / mean /
fraction-above-threshold.

Outputs:
  results/similarity/external_similarity.csv   per-guest summary
  results/similarity/run_log.txt
"""
import os, sys, warnings
warnings.filterwarnings('ignore')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

OUT = os.path.join(ROOT, 'results', 'similarity'); os.makedirs(OUT, exist_ok=True)
log = []; pr = lambda s='': (print(s, flush=True), log.append(str(s)))

RADII = [2, 3]   # ECFP4 (radius 2) and ECFP6 (radius 3)

def fp(mol, radius):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=2048)

# training guests (canonical SMILES, deduplicated)
ext = pd.concat([pd.read_csv(os.path.join(ROOT, 'data/binary/' + f)) for f in
                 ('train.csv', 'val.csv', 'test.csv')], ignore_index=True)
train_guests = list(dict.fromkeys(ext['SMILES_Guest'].astype(str)))   # preserve order, dedupe
pr(f"unique training guests: {len(train_guests)}")

train_fps = {r: [] for r in RADII}
train_mols = []
for s in train_guests:
    m = Chem.MolFromSmiles(s)
    if m is None:
        continue
    train_mols.append(m)
    for r in RADII:
        train_fps[r].append(fp(m, r))

# external guests
ex = pd.read_csv(os.path.join(ROOT, 'results/tables/external_guest_smiles.csv'))
rows = []
for _, er in ex.iterrows():
    m = Chem.MolFromSmiles(er['CanonicalSMILES'])
    if m is None:
        pr(f"  {er['Drug']}: invalid SMILES"); continue
    rec = {'Drug': er['Drug'], 'CID': int(er['CID'])}
    sims = {}
    for r in RADII:
        q = fp(m, r)
        sims[r] = DataStructs.BulkTanimotoSimilarity(q, train_fps[r])
    for r in RADII:
        s = sorted(sims[r], reverse=True)
        rec[f'Tanimoto_r{r}_max'] = round(s[0], 3)
        rec[f'Tanimoto_r{r}_mean'] = round(float(sum(s)/len(s)), 4)
        rec[f'Tanimoto_r{r}_frac_gt_0.5'] = round(float(sum(1 for x in s if x > 0.5))/len(s), 4)
    # nearest-neighbour identity (radius 2)
    s2 = sims[2]
    nn_idx = int(max(range(len(s2)), key=lambda i: s2[i]))
    rec['nearest_train_guest_SMILES'] = train_guests[nn_idx]
    rows.append(rec)

res = pd.DataFrame(rows)
res = res[['Drug', 'CID',
           'Tanimoto_r2_max', 'Tanimoto_r2_mean', 'Tanimoto_r2_frac_gt_0.5',
           'Tanimoto_r3_max', 'Tanimoto_r3_mean', 'Tanimoto_r3_frac_gt_0.5',
           'nearest_train_guest_SMILES']]
res.to_csv(os.path.join(OUT, 'external_similarity.csv'), index=False)
pr("\nExternal guest similarity to training guests (Morgan Tanimoto):")
pr(res.drop(columns=['nearest_train_guest_SMILES']).to_string(index=False))
pr(f"\nMean max-Tanimoto (r2) across 8 guests = {res['Tanimoto_r2_max'].mean():.3f}")
pr("Saved: external_similarity.csv, run_log.txt")
open(os.path.join(OUT, 'run_log.txt'), 'w').write('\n'.join(log))
