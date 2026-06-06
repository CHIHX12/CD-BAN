"""
Verify that the binary calibration formula classifies as accurately as direct model prediction.
Compares Method A (z >= 0 -> Weak) vs Method B (K_est threshold).
"""
import torch, yaml, sys, warnings, math
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
import pandas as pd, numpy as np
from models import CDBAN
from dataloader import CDBinaryDataset, cd_collate_func
from torch.utils.data import DataLoader

SLOPE, INTCPT = -1.128, 3.714

with open('configs/CDBAN.yaml') as f:
    cfg = yaml.safe_load(f)
model = CDBAN(**cfg)
model.load_state_dict(torch.load('results/seed_49/best_model_epoch_69.pth', map_location='cpu'))
model.eval()

df = pd.read_csv('data/binary/test.csv')
ds = CDBinaryDataset(df.index.values, df)
ld = DataLoader(ds, batch_size=128, shuffle=False, collate_fn=cd_collate_func)

z_vals = []
y_vals = []
with torch.no_grad():
    for bg, bh, yl in ld:
        _, _, score, _ = model(bg, bh, mode='eval')
        z_vals.extend(score.squeeze(1).tolist())
        y_vals.extend(yl.tolist())

logK_true = list(df['log10K'])

z    = np.array(z_vals)
y    = np.array(y_vals)
logK = np.array(logK_true)

# Method A: direct model classification (z >= 0 -> Weak)
model_pred = (z >= 0).astype(float)

# Method B: calibration formula classification
#   logK_est = (3.714 - z) / 1.128
#   logK_est < 2 -> K < 100   -> Weak (1)
#   logK_est > 4 -> K > 10000 -> Strong (0)
#   intermediate -> use z sign (same as Method A)
logK_est     = (INTCPT - z) / (-SLOPE)
formula_pred = np.where(logK_est < 2.0, 1.0,
               np.where(logK_est > 4.0, 0.0,
               (z >= 0).astype(float)))

model_acc   = (model_pred == y).mean()
formula_acc = (formula_pred == y).mean()

print('=== Test Set (n={}) — Binary Formula Verification ==='.format(len(y)))
print()
print('Method A  Direct model classification (z >= 0 -> Weak) : {:.2f}%'.format(model_acc * 100))
print('Method B  Calibration formula (K_est threshold)         : {:.2f}%'.format(formula_acc * 100))
print()
print('Differences: ', end='')
diff = (model_pred != formula_pred).sum()
print('{} predictions differ'.format(diff))
print()

hdr = '{:>4}  {:>7}  {:>8}  {:>10}  {:>7}  {:>8}  {:>8}'
print(hdr.format('#', 'logK_true', 'z_bin', 'K_est', 'True', 'Model', 'Formula'))
print('-' * 68)

row_fmt = '{:>4}  {:>7.3f}  {:>+8.3f}  {:>10.1f}  {:>7}  {:>8}  {:>8}'

errors_model   = []
errors_formula = []

for i in range(len(z)):
    k_est  = 10 ** min(max(logK_est[i], -1), 8)
    m_ok   = 'OK' if model_pred[i] == y[i] else 'WRONG'
    f_ok   = 'OK' if formula_pred[i] == y[i] else 'WRONG'
    label  = 'Weak' if y[i] == 1 else 'Strong'
    show   = (i < 10) or (m_ok == 'WRONG') or (f_ok == 'WRONG')
    if show:
        print(row_fmt.format(i + 1, logK[i], z[i], k_est, label, m_ok, f_ok))
    if m_ok == 'WRONG':
        errors_model.append(i + 1)
    if f_ok == 'WRONG':
        errors_formula.append(i + 1)

print('-' * 68)
print('Model errors:   {} cases  {}'.format(len(errors_model), errors_model))
print('Formula errors: {} cases  {}'.format(len(errors_formula), errors_formula))
print()
print('Conclusion:')
if formula_acc == model_acc:
    print('  Formula classification accuracy == direct model classification  -> formula is equivalent ✓')
elif formula_acc > model_acc:
    print('  Formula classification accuracy > direct model classification')
else:
    d = (model_acc - formula_acc) * 100
    print('  Formula accuracy is {:.2f}% lower than direct model classification'.format(d))
    print('  Differing cases: {}'.format(int(diff)))
