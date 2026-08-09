"""
Run CD-BAN inference on 201 duplicate entries excluded from training.
These entries share the same host-guest pair but have conflicting K values
across sources. Uses best model (seed 49) to predict p_weak and assess
whether the model can distinguish high-affinity compounds.
"""
import sys, os, torch, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr
from models import CDBAN
from dataloader import CDBinaryDataset, cd_collate_func
from utils import set_seed
import yaml

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('configs/CDBAN.yaml') as f:
    cfg = yaml.safe_load(f)

model = CDBAN(**cfg).to(device)
model.load_state_dict(torch.load(
    'results/seed_49/best_model_epoch_69.pth', map_location=device))
model.eval()
print(f'Model loaded (seed 49, AUROC=0.9655)  device={device}')

RAW = '/home/nibiohnproj9/cycheng/cyclodextrin_research_data/curated_datasets/duplicate_entries.csv'
df = pd.read_csv(RAW)
df = df.rename(columns={'IsomericSMILES': 'SMILES_Guest',
                         'IsomericSMILES_Host': 'SMILES_Host'})
df['label'] = -1   # placeholder

print(f'Duplicate entries: {len(df)}')
print(f'  log10K range: {df.log10K.min():.2f} ~ {df.log10K.max():.2f}')
print(f'  High affinity (log10K > 4): {(df.log10K > 4).sum()}')

ds = CDBinaryDataset(df.index.values, df)
loader = DataLoader(ds, batch_size=32, shuffle=False,
                    num_workers=0, collate_fn=cd_collate_func)

probs = []
with torch.no_grad():
    for bg, bh, _ in loader:
        bg, bh = bg.to(device), bh.to(device)
        _, _, score, _ = model(bg, bh, mode='eval')
        p = torch.sigmoid(score.squeeze(1))
        probs.extend(p.cpu().tolist())

df['p_weak']   = probs
df['p_strong'] = 1 - df['p_weak']
df['pred_class'] = df['p_weak'].apply(lambda p: 'Weak (K<100)' if p >= 0.5 else 'Strong (K>10k)')

print('\n' + '='*65)
print('  CD-BAN Inference — Duplicate Entries (n=201)')
print('='*65)

r, _  = pearsonr(df['log10K'],  df['p_weak'])
sr, _ = spearmanr(df['log10K'], df['p_weak'])
print(f'\n  Pearson r  (log10K vs p_weak) = {r:.4f}')
print(f'  Spearman ρ (log10K vs p_weak) = {sr:.4f}')

print(f'\n  Mean predicted probability by log10K bin:')
print(f"  {'Range':<18} {'n':>4}  {'P(Weak)':>11}  {'P(Strong)':>13}  mean K")
print('  ' + '-'*62)
edges = [(0, 2.0,'<100'), (2.0, 3.0,'100-1k'), (3.0, 4.0,'1k-10k'), (4.0, 99,'> 10k')]
for lo, hi, label in edges:
    sub = df[(df['log10K'] >= lo) & (df['log10K'] < hi)]
    if len(sub) == 0: continue
    k_mean = 10**sub['log10K'].mean()
    print(f"  log10K {label:<10} {len(sub):>4}  "
          f"{sub.p_weak.mean():>11.4f}  "
          f"{sub.p_strong.mean():>13.4f}  {k_mean:>8.0f} M⁻¹")

hi_aff = df[df['log10K'] > 4.0].sort_values('log10K', ascending=False)
print(f'\n  High-affinity compounds (log10K > 4, K > 10,000 M⁻¹):')
print(f"  {'Guest':<25} {'Host':<25} {'log10K':>7} {'K(M-1)':>8}  {'p_weak':>7}  {'p_strong':>8}  Pred")
print('  ' + '-'*100)
for _, row in hi_aff.iterrows():
    guest = str(row.get('Guest', row['SMILES_Guest'][:15]))[:24]
    host  = str(row.get('Host',  row['SMILES_Host'][:15]))[:24]
    print(f"  {guest:<25} {host:<25} {row['log10K']:>7.3f} {10**row['log10K']:>8.0f}"
          f"  {row['p_weak']:>7.4f}  {row['p_strong']:>8.4f}  {row['pred_class']}")

print(f'\n  P(Weak) distribution (all 201):')
for q, lbl in [(.05,'5%'),(.25,'Q1'),(.5,'Median'),(.75,'Q3'),(.95,'95%')]:
    print(f'    {lbl:>8}: {df.p_weak.quantile(q):.4f}')

print(f'\n  Pred Strong (p<0.5): {(df.p_weak < 0.5).sum()}')
print(f'  Pred Weak   (p≥0.5): {(df.p_weak >= 0.5).sum()}')

out = 'results/seed_stability/duplicate_predictions.csv'
df[['Guest','Host','log10K','p_weak','p_strong','pred_class',
    'SMILES_Guest','SMILES_Host']].to_csv(out, index=False)
print(f'\n  -> Saved: {out}')
print('='*65)
