"""
Run inference on the fuzzy zone (100 ≤ K ≤ 10,000 M⁻¹) using the best model.
Analyses the continuous relationship between predicted P(Weak) and log10K.
"""
import sys, os
sys.path.insert(0, '.')

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr
from models import CDBAN
from dataloader import CDBinaryDataset, cd_collate_func
from utils import set_seed
import yaml, warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BEST_SEED  = 49                   # AUROC = 0.9655
MODEL_PATH = f'results/seed_{BEST_SEED}/best_model_epoch_69.pth'
FUZZY_CSV  = 'data/binary/fuzzy.csv'
CFG_PATH   = 'configs/CDBAN.yaml'

with open(CFG_PATH) as f:
    cfg = yaml.safe_load(f)

set_seed(42)

model = CDBAN(**cfg).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"Model loaded: {MODEL_PATH}")

df = pd.read_csv(FUZZY_CSV)
df['label'] = -1   # placeholder; not used during inference
print(f"Fuzzy zone: {len(df)} compounds  log10K=[{df.log10K.min():.2f},{df.log10K.max():.2f}]")

ds = CDBinaryDataset(df.index.values, df)
loader = DataLoader(ds, batch_size=64, shuffle=False,
                    num_workers=2, collate_fn=cd_collate_func)

all_probs = []
with torch.no_grad():
    for bg, bh, _ in loader:
        bg, bh = bg.to(device), bh.to(device)
        _, _, score, _ = model(bg, bh, mode='eval')
        prob = torch.sigmoid(score.squeeze(1))
        all_probs.extend(prob.cpu().tolist())

df['p_weak']   = all_probs
df['p_strong'] = 1 - df['p_weak']

print("\n" + "="*60)
print("  Inference Results")
print("="*60)

r_p, _ = pearsonr(df['log10K'], df['p_weak'])
sr_p, _= spearmanr(df['log10K'], df['p_weak'])
print(f"\n  p_weak vs log10K:  Pearson r={r_p:.3f}  Spearman={sr_p:.3f}")
print(f"  (expected negative: higher K -> lower P(Weak))")

bins   = [2.0, 2.5, 3.0, 3.5, 4.0]
labels = ['2.0-2.5\n(K=100-316)', '2.5-3.0\n(K=316-1k)',
          '3.0-3.5\n(K=1k-3k)', '3.5-4.0\n(K=3k-10k)']
df['bin'] = pd.cut(df['log10K'], bins=bins, labels=labels[:4])

print("\n  Mean predicted probability by log10K bin:")
print(f"  {'log10K bin':<20} {'n':>5}  {'P(Weak) ↓':>10}  {'P(Strong) ↑':>12}  {'mean K':>10}")
print("  " + "-"*60)
for label, grp in df.groupby('bin', observed=True):
    lbl_clean = label.replace('\n', ' ')
    print(f"  {lbl_clean:<20} {len(grp):>5}  "
          f"{grp.p_weak.mean():>10.4f}  "
          f"{grp.p_strong.mean():>12.4f}  "
          f"{10**grp.log10K.mean():>10.0f}")

near_100   = df[df['log10K'] < 2.2]
near_10000 = df[df['log10K'] > 3.8]

print(f"\n  Near K=100 (log10K<2.2, n={len(near_100)}):")
print(f"    mean P(Weak)   = {near_100.p_weak.mean():.4f}  (expected ~1.0)")
print(f"    mean P(Strong) = {near_100.p_strong.mean():.4f}  (expected ~0.0)")

print(f"\n  Near K=10000 (log10K>3.8, n={len(near_10000)}):")
print(f"    mean P(Weak)   = {near_10000.p_weak.mean():.4f}  (expected ~0.0)")
print(f"    mean P(Strong) = {near_10000.p_strong.mean():.4f}  (expected ~1.0)")

mid = df[(df['log10K'] > 2.8) & (df['log10K'] < 3.2)]
print(f"\n  Intermediate zone (log10K=2.8-3.2, K≈630-1585, n={len(mid)}):")
print(f"    mean P(Weak)   = {mid.p_weak.mean():.4f}  (expected ~0.5)")
print(f"    mean P(Strong) = {mid.p_strong.mean():.4f}")

print(f"\n  P(Weak) distribution (full fuzzy zone):")
for q, label in [(0.05,'5%'),(0.25,'25%'),(0.5,'50%'),(0.75,'75%'),(0.95,'95%')]:
    print(f"    {label:>4}: {df.p_weak.quantile(q):.4f}")

out = 'results/seed_stability/fuzzy_predictions.csv'
df.to_csv(out, index=False)
print(f"\n  -> Saved: {out}")
print("="*60)
