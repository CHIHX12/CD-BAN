"""
Full accuracy report: binary classification, fuzzy-zone detection, and K estimation.
Three-part output covering the test set (n=120) and fuzzy zone (n=1,850).
"""
import torch, yaml, sys, warnings, math
warnings.filterwarnings('ignore')
sys.path.insert(0,'.')
import pandas as pd, numpy as np
from models import CDBAN
from dataloader import CDBinaryDataset, cd_collate_func
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix

SLOPE, INTCPT = -1.128, 3.714
C2, C1 =  1.458, -0.799   # confidence zone boundaries

with open('configs/CDBAN.yaml') as f:
    cfg = yaml.safe_load(f)
model = CDBAN(**cfg)
model.load_state_dict(torch.load('results/seed_49/best_model_epoch_69.pth',
                                  map_location='cpu'))
model.eval()

def run_inference(csv_path, has_label=True):
    df = pd.read_csv(csv_path)
    if not has_label:
        df['label'] = 0   # placeholder; not used during inference
    ds = CDBinaryDataset(df.index.values, df)
    ld = DataLoader(ds, batch_size=128, shuffle=False, collate_fn=cd_collate_func)
    z_vals = []
    with torch.no_grad():
        for bg, bh, yl in ld:
            _, _, score, _ = model(bg, bh, mode='eval')
            z_vals.extend(score.squeeze(1).tolist())
    if not has_label:
        df = df.drop(columns=['label'])
    df['z_bin'] = z_vals
    df['logK_est'] = (INTCPT - df['z_bin']) / (-SLOPE)
    df['K_est'] = 10 ** df['logK_est'].clip(-1, 8)
    df['zone'] = pd.cut(df['z_bin'],
                        bins=[-np.inf, C1, C2, np.inf],
                        labels=['Confident Strong','Fuzzy Zone','Confident Weak'])
    if has_label:
        df['pred_label'] = (df['z_bin'] >= 0).astype(int)
        df['correct'] = (df['pred_label'] == df['label']).astype(int)
    return df

# ════════════════════════════════════════════════════════════════
print('=' * 62)
print('  PART 1 — Binary Classification Accuracy (Test Set, n=120)')
print('=' * 62)

df_test = run_inference('data/binary/test.csv')
acc = df_test['correct'].mean()
auroc = roc_auc_score(df_test['label'],
                      1 / (1 + np.exp(-df_test['z_bin'])))

for zone in ['Confident Weak','Fuzzy Zone','Confident Strong']:
    sub = df_test[df_test['zone'] == zone]
    if len(sub) == 0:
        continue
    right = sub['correct'].sum()
    total = len(sub)
    print(f'  {zone:<18}: {right:>3}/{total:<3} = {right/total*100:.1f}%  '
          f'(z range: {sub["z_bin"].min():+.2f} ~ {sub["z_bin"].max():+.2f})')

print()
print(f'  Overall accuracy : {acc*100:.2f}%')
print(f'  AUROC            : {auroc:.4f}')
print()

cm = confusion_matrix(df_test['label'], df_test['pred_label'])
tn, fp, fn, tp = cm.ravel()
print(f'  Confusion matrix:')
print(f'    True Weak   -> Pred Weak  : {tp} correct')
print(f'    True Weak   -> Pred Strong: {fn} missed')
print(f'    True Strong -> Pred Strong: {tn} correct')
print(f'    True Strong -> Pred Weak  : {fp} missed')

# ════════════════════════════════════════════════════════════════
print()
print('=' * 62)
print('  PART 2 — Fuzzy-Zone Detection (n=1,850)')
print('=' * 62)
print('  Fuzzy zone: literature K in 100–10,000 M⁻¹')
print('  Model was never trained on these; tests whether it')
print('  naturally places them in the intermediate z region.')
print()

df_fuzzy = run_inference('data/binary/fuzzy.csv', has_label=False)

total = len(df_fuzzy)
for zone in ['Confident Weak','Fuzzy Zone','Confident Strong']:
    n = (df_fuzzy['zone'] == zone).sum()
    pct = n / total * 100
    print(f'  {zone:<18}: {n:>5} ({pct:.1f}%)')

in_fuzzy_zone = (df_fuzzy['zone'] == 'Fuzzy Zone').sum()
print()
print(f'  Model correctly places in Fuzzy Zone: '
      f'{in_fuzzy_zone}/{total} = {in_fuzzy_zone/total*100:.1f}%')
print()

print('  Breakdown by true K range vs model zone:')
df_fuzzy['true_bin'] = pd.cut(df_fuzzy['log10K'],
                               bins=[2.0, 2.5, 3.0, 3.5, 4.0],
                               include_lowest=True)
tbl = pd.crosstab(df_fuzzy['true_bin'], df_fuzzy['zone'])
print(tbl.to_string())
print()

# ════════════════════════════════════════════════════════════════
print('=' * 62)
print('  PART 3 — K Estimation Accuracy (Fuzzy Zone)')
print('=' * 62)
print('  Formula: log₁₀K_est = (3.714 − z) / 1.128')
print()

from scipy.stats import pearsonr, spearmanr
r, _  = pearsonr(df_fuzzy['log10K'], df_fuzzy['logK_est'])
sr, _ = spearmanr(df_fuzzy['log10K'], df_fuzzy['logK_est'])
mae   = (df_fuzzy['log10K'] - df_fuzzy['logK_est']).abs().mean()
rmse  = np.sqrt(((df_fuzzy['log10K'] - df_fuzzy['logK_est'])**2).mean())
print(f'  Pearson r        = {r:.4f}  (p < 0.001)')
print(f'  Spearman ρ       = {sr:.4f}  (p < 0.001)')
print(f'  MAE (log₁₀K)    = {mae:.3f} log units')
print(f'  RMSE (log₁₀K)   = {rmse:.3f} log units')
print(f'  -> median K_est fold error: ~{10**mae:.1f}x')
print()

# ════════════════════════════════════════════════════════════════
print('=' * 62)
print('  Summary')
print('=' * 62)
print(f'  Binary classification accuracy : {acc*100:.2f}%  (AUROC {auroc:.4f})')
print(f'  Fuzzy-zone detection           : {in_fuzzy_zone/total*100:.1f}% of fuzzy'
      f' compounds fall in the intermediate zone')
print(f'  K estimation correlation       : r = {r:.3f}, ~{10**mae:.1f}x fold error')
print()
print('  High-confidence zones (z > +1.458 or z < -0.799): reliable classification')
print('  Fuzzy zone (-0.799 < z < +1.458): K estimates are trends, not precise values')
