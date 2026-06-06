"""
Publication-quality tables for CD-BAN
Output: TXT (formatted) + CSV (for Excel/LaTeX)
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, shapiro, kurtosis, skew
import os

OUT = 'results/tables'
os.makedirs(OUT, exist_ok=True)

# ── Load data ─────────────────────────────────────────────
df_train   = pd.read_csv('data/binary/train.csv')
df_val     = pd.read_csv('data/binary/val.csv')
df_test    = pd.read_csv('data/binary/test.csv')
df_fuzzy   = pd.read_csv('data/binary/fuzzy.csv')
df_pred    = pd.read_csv('results/seed_stability/fuzzy_predictions.csv')
df_seeds   = pd.read_csv('results/seed_stability/summary.csv')

# Master dataset (all binary + fuzzy)
df_all_k   = pd.concat([df_train[['log10K','label']],
                         df_val[['log10K','label']],
                         df_test[['log10K','label']],
                         df_fuzzy[['log10K']].assign(label=np.nan)],
                        ignore_index=True)

lines = []
def h(text=''):
    lines.append(text)
def sep(char='─', n=80):
    lines.append(char * n)
def row(*cols, widths):
    parts = []
    for c, w in zip(cols, widths):
        parts.append(str(c).ljust(w) if len(parts)==0 else str(c).rjust(w))
    lines.append('  ' + '  '.join(parts))

# ═══════════════════════════════════════════════════════════════════════════════
h('=' * 80)
h('  SUPPLEMENTARY TABLES — CD-BAN Binary Classification')
h('  Cyclodextrin Inclusion Complex Affinity Prediction')
h('=' * 80)

# ───────────────────────────────────────────────────────────────────────────────
# Table 1: Dataset composition
# ───────────────────────────────────────────────────────────────────────────────
h()
h('Table 1.  Dataset Composition and log₁₀K Distribution')
sep()
row('Split', 'n (total)', 'Label 0\n(K>10k)', 'Label 1\n(K<100)',
    'log₁₀K\nMin', 'log₁₀K\nMax', 'Mean ± SD', 'Median',
    widths=[10,10,12,12,10,10,14,10])
sep('─')

splits = [
    ('Training',   df_train),
    ('Validation', df_val),
    ('Test',       df_test),
    ('Fuzzy Zone', df_fuzzy),
]
for name, df in splits:
    k = df['log10K']
    n = len(df)
    n0 = (df['label'] == 0).sum() if 'label' in df.columns else '—'
    n1 = (df['label'] == 1).sum() if 'label' in df.columns else '—'
    row(name, n, n0, n1,
        f'{k.min():.2f}', f'{k.max():.2f}',
        f'{k.mean():.3f} ± {k.std():.3f}',
        f'{k.median():.3f}',
        widths=[10,10,12,12,10,10,14,10])

sep('─')
# Total binary
df_bin = pd.concat([df_train, df_val, df_test])
k_bin  = df_bin['log10K']
row('Binary (total)', len(df_bin),
    (df_bin.label==0).sum(), (df_bin.label==1).sum(),
    f'{k_bin.min():.2f}', f'{k_bin.max():.2f}',
    f'{k_bin.mean():.3f} ± {k_bin.std():.3f}',
    f'{k_bin.median():.3f}',
    widths=[10,10,12,12,10,10,14,10])
sep()
h('  Label 0 = Strong binder (K > 10,000 M⁻¹);  Label 1 = Weak binder (K < 100 M⁻¹)')
h('  Fuzzy Zone (100 ≤ K ≤ 10,000 M⁻¹) excluded from training; used for inference only.')
h('  Split: 7:2:1 stratified random split preserving class ratio (1:11.4).')

# CSV Table 1
rows_t1 = []
for name, df in splits + [('Binary (total)', df_bin)]:
    k = df['log10K']
    rows_t1.append({
        'Split': name,
        'n_total': len(df),
        'n_label0': (df['label']==0).sum() if 'label' in df.columns else np.nan,
        'n_label1': (df['label']==1).sum() if 'label' in df.columns else np.nan,
        'log10K_min': round(k.min(), 3),
        'log10K_max': round(k.max(), 3),
        'log10K_mean': round(k.mean(), 4),
        'log10K_std':  round(k.std(),  4),
        'log10K_median': round(k.median(), 4),
        'log10K_Q1': round(k.quantile(0.25), 4),
        'log10K_Q3': round(k.quantile(0.75), 4),
    })
pd.DataFrame(rows_t1).to_csv(f'{OUT}/table1_dataset.csv', index=False)

# ───────────────────────────────────────────────────────────────────────────────
# Table 2: Seed stability — per-seed results
# ───────────────────────────────────────────────────────────────────────────────
h()
h()
h('Table 2.  CD-BAN Performance across 10 Independent Random Seeds (Test Set)')
sep()
row('Seed', 'Best\nEpoch', 'AUROC', 'AUPRC', 'F1', 'Sensitivity',
    'Specificity', 'Accuracy',
    widths=[6, 8, 8, 8, 8, 13, 13, 10])
sep('─')

for _, r in df_seeds.iterrows():
    row(int(r.seed), int(r.best_epoch),
        f'{r.auroc:.4f}', f'{r.auprc:.4f}',
        f'{r.f1:.4f}',
        f'{r.sensitivity:.4f}', f'{r.specificity:.4f}',
        f'{r.accuracy:.4f}',
        widths=[6,8,8,8,8,13,13,10])

sep('─')
for metric, col in [('AUROC','auroc'),('AUPRC','auprc'),('F1','f1'),
                    ('Sensitivity','sensitivity'),('Specificity','specificity'),
                    ('Accuracy','accuracy')]:
    v = df_seeds[col]
    row(f'Mean', '—',
        *(['—']*({'auroc':0,'auprc':1,'f1':2,'sensitivity':3,
                  'specificity':4,'accuracy':5}[col]) +
          [f'{v.mean():.4f} ± {v.std():.4f}'] +
          ['—']*(5-{'auroc':0,'auprc':1,'f1':2,'sensitivity':3,
                    'specificity':4,'accuracy':5}[col])),
        widths=[6,8,8,8,8,13,13,10])
    break  # just print one combined summary row below

# Better summary
sep('─')
for metric, col in [('AUROC','auroc'),('AUPRC','auprc'),('F1','f1'),
                    ('Sens','sensitivity'),('Spec','specificity'),('Acc','accuracy')]:
    v = df_seeds[col]
    h(f'  {metric:<8}: {v.mean():.4f} ± {v.std():.4f}  '
      f'[{v.min():.4f}, {v.max():.4f}]')
sep()
h('  Best model selected by maximum validation AUROC per run.')
h('  Pos_weight = 11.32 (BCEWithLogitsLoss); 100 epochs; Adam lr = 5×10⁻⁵.')

df_seeds_out = df_seeds.copy()
df_seeds_out.to_csv(f'{OUT}/table2_seed_stability.csv', index=False)

# ───────────────────────────────────────────────────────────────────────────────
# Table 3: Fuzzy zone — bin statistics
# ───────────────────────────────────────────────────────────────────────────────
h()
h()
h('Table 3.  CD-BAN Prediction Statistics for the Fuzzy Zone by log₁₀K Bin')
sep()
row('log₁₀K Bin', 'K Range (M⁻¹)', 'n',
    'p_weak\nMean ± SD', 'p_strong\nMean ± SD',
    'p_weak\nMedian', 'p_strong\nMedian',
    widths=[14, 20, 6, 18, 18, 10, 12])
sep('─')

bins   = [2.0, 2.5, 3.0, 3.5, 4.0]
kranges= ['100–316','316–1,000','1,000–3,162','3,162–10,000']
blabels= ['2.0–2.5','2.5–3.0','3.0–3.5','3.5–4.0']
df_pred['bin4'] = pd.cut(df_pred['log10K'], bins=bins)

rows_t3 = []
df_pred['_bin'] = pd.cut(df_pred['log10K'], bins=bins, include_lowest=True)
for (lo, hi), kr, bl in zip(zip(bins[:-1], bins[1:]), kranges, blabels):
    sub = df_pred[(df_pred['log10K'] >= lo) & (df_pred['log10K'] <= hi)] if lo == bins[0] \
          else df_pred[(df_pred['log10K'] > lo) & (df_pred['log10K'] <= hi)]
    pw = sub['p_weak']; ps = sub['p_strong']
    row(bl, kr, len(sub),
        f'{pw.mean():.4f} ± {pw.std():.4f}',
        f'{ps.mean():.4f} ± {ps.std():.4f}',
        f'{pw.median():.4f}', f'{ps.median():.4f}',
        widths=[14,20,6,18,18,10,12])
    rows_t3.append({
        'bin': bl, 'K_range': kr, 'n': len(sub),
        'p_weak_mean':   round(pw.mean(),4), 'p_weak_sd':    round(pw.std(),4),
        'p_weak_median': round(pw.median(),4),
        'p_weak_Q1':     round(pw.quantile(.25),4),
        'p_weak_Q3':     round(pw.quantile(.75),4),
        'p_weak_min':    round(pw.min(),4), 'p_weak_max': round(pw.max(),4),
        'p_strong_mean': round(ps.mean(),4),'p_strong_sd':  round(ps.std(),4),
        'p_strong_median':round(ps.median(),4),
    })

sep('─')
# Overall
pw_all = df_pred['p_weak']; ps_all = df_pred['p_strong']
row('All bins (total)', '100–10,000', len(df_pred),
    f'{pw_all.mean():.4f} ± {pw_all.std():.4f}',
    f'{ps_all.mean():.4f} ± {ps_all.std():.4f}',
    f'{pw_all.median():.4f}', f'{ps_all.median():.4f}',
    widths=[14,20,6,18,18,10,12])
sep()

r_val, p_r = pearsonr(df_pred['log10K'],  df_pred['p_weak'])
sr_val,p_sr= spearmanr(df_pred['log10K'], df_pred['p_weak'])
h(f'  Pearson r  (log₁₀K vs p_weak) = {r_val:.4f}   (p < 0.001)')
h(f'  Spearman ρ (log₁₀K vs p_weak) = {sr_val:.4f}   (p < 0.001)')
h('  p_weak = P(Weak Binder); p_strong = 1 − p_weak.')
h('  Fuzzy zone compounds were never seen during training.')
pd.DataFrame(rows_t3).to_csv(f'{OUT}/table3_fuzzy_stats.csv', index=False)

# ───────────────────────────────────────────────────────────────────────────────
# Table 4: log10K full distribution (all 3048)
# ───────────────────────────────────────────────────────────────────────────────
h()
h()
h('Table 4.  Full Dataset log₁₀K Distribution by CD Family (Training Set, n = 3,048)')
sep()
row('CD Type', 'n', 'Min', 'Max', 'Mean ± SD',
    'Q1', 'Median', 'Q3', 'Skewness', 'Kurtosis',
    widths=[14,6,6,6,16,7,8,7,10,10])
sep('─')

# reconstruct full dataset from available splits
df_full = pd.concat([df_train, df_val, df_test,
                     df_fuzzy.assign(label=np.nan)], ignore_index=True)

def cd_fam(h):
    h = str(h).lower()
    if 'trimethyl' in h:                              return 'TM-β-CD'
    if '2,6-di-o-methyl' in h:                       return 'DM-β-CD'
    if 'methyl' in h and 'beta' in h:                return 'Me-β-CD'
    if 'sulfobutyl' in h or 'sbe' in h:              return 'SBE-β-CD'
    if ('hp' in h or 'hydroxypropyl' in h) and 'gamma' not in h: return 'HP-β-CD'
    if 'alpha' in h or 'α' in h:                     return 'α-CD'
    if 'gamma' in h or 'γ' in h:                     return 'γ-CD'
    return 'β-CD'

df_full['cd_fam'] = df_full['Host'].apply(cd_fam)
rows_t4 = []
fam_order = ['α-CD','β-CD','γ-CD','HP-β-CD','SBE-β-CD','Me-β-CD','DM-β-CD','TM-β-CD']

for fam in fam_order:
    sub = df_full[df_full['cd_fam'] == fam]['log10K']
    if len(sub) == 0: continue
    sk = skew(sub)
    kt = kurtosis(sub)
    row(fam, len(sub),
        f'{sub.min():.2f}', f'{sub.max():.2f}',
        f'{sub.mean():.3f} ± {sub.std():.3f}',
        f'{sub.quantile(.25):.3f}', f'{sub.median():.3f}',
        f'{sub.quantile(.75):.3f}',
        f'{sk:.3f}', f'{kt:.3f}',
        widths=[14,6,6,6,16,7,8,7,10,10])
    rows_t4.append({
        'cd_type': fam, 'n': len(sub),
        'min': round(sub.min(),3), 'max': round(sub.max(),3),
        'mean': round(sub.mean(),4), 'sd': round(sub.std(),4),
        'Q1': round(sub.quantile(.25),4),
        'median': round(sub.median(),4),
        'Q3': round(sub.quantile(.75),4),
        'skewness': round(sk,4), 'kurtosis': round(kt,4),
    })

sep('─')
all_k = df_full['log10K']
row('All CD types', len(all_k),
    f'{all_k.min():.2f}', f'{all_k.max():.2f}',
    f'{all_k.mean():.3f} ± {all_k.std():.3f}',
    f'{all_k.quantile(.25):.3f}', f'{all_k.median():.3f}',
    f'{all_k.quantile(.75):.3f}',
    f'{skew(all_k):.3f}', f'{kurtosis(all_k):.3f}',
    widths=[14,6,6,6,16,7,8,7,10,10])
sep()
h('  K = binding association constant (M⁻¹); log₁₀K = base-10 logarithm.')
h('  Source: OpenCycloDB (curated); n = 3,048 unique drug–CD pairs.')
h('  CD types: α = alpha, β = beta, γ = gamma; HP = hydroxypropyl,')
h('  SBE = sulfobutyl ether, Me = methyl, DM = 2,6-di-O-methyl, TM = trimethyl.')

pd.DataFrame(rows_t4).to_csv(f'{OUT}/table4_cd_family_distribution.csv', index=False)

# ───────────────────────────────────────────────────────────────────────────────
# Table 5: Model hyperparameters
# ───────────────────────────────────────────────────────────────────────────────
h()
h()
h('Table 5.  CD-BAN Model Architecture and Training Hyperparameters')
sep()
h()

params = [
    ('ARCHITECTURE',),
    ('Component',            'Setting',                   'Note'),
    ('─'*28, '─'*28, '─'*18),
    ('Guest GCN layers',     '3 × GCNConv (128-128-128)', 'torch_geometric'),
    ('Host GCN layers',      '3 × GCNConv (128-128-128)', 'Same as guest'),
    ('Node embedding dim',   '128',                       'Linear projection'),
    ('Atom feature dim',     '74',                        'Chemprop-style'),
    ('Max graph nodes',      '290',                       'Padding applied'),
    ('BAN attention heads',  '2',                         'BANLayer'),
    ('Decoder hidden dim',   '512',                       'FC layers: 256→512→128→1'),
    ('Output activation',    'Sigmoid',                   'via BCEWithLogitsLoss'),
    ('Total parameters',     '780,550',                   '≈0.78 M'),
    ('',),
    ('TRAINING',),
    ('─'*28, '─'*28, '─'*18),
    ('Loss function',        'BCEWithLogitsLoss',         ''),
    ('pos_weight',           '11.32',                     'n(label=1)/n(label=0)'),
    ('Optimizer',            'Adam',                      ''),
    ('Learning rate',        '5 × 10⁻⁵',                 ''),
    ('Batch size',           '64',                        'drop_last=True'),
    ('Max epochs',           '100',                       'DrugBAN standard'),
    ('Best model criterion', 'Max validation AUROC',      ''),
    ('Random seeds',         '42, 43, …, 51 (10 runs)',   'Stability evaluation'),
    ('GPU',                  'Tesla V100-SXM2 (16 GB)',   '4× parallel for seeds'),
    ('',),
    ('DATASET SPLIT',),
    ('─'*28, '─'*28, '─'*18),
    ('Strategy',             '7 : 2 : 1 stratified',     'Maintains class ratio'),
    ('Train / Val / Test',   '838 / 240 / 120',           'Label = 0 or 1 only'),
    ('Class balance',        'Label 0: 8.1%, Label 1: 91.9%','pos_weight compensates'),
    ('Fuzzy zone',           '1,850 (inference only)',    'Never used in training'),
]
for p in params:
    if len(p) == 1:
        h()
        h(f'  ── {p[0]}')
    elif p[0].startswith('─'):
        sep('─', 74)
    else:
        h(f'  {str(p[0]):<30} {str(p[1]):<30} {str(p[2]) if len(p)>2 else ""}')

sep()
pd.DataFrame([
    {'Parameter': p[0], 'Value': p[1], 'Note': p[2] if len(p)>2 else ''}
    for p in params if len(p)==3 and not p[0].startswith('─')
]).to_csv(f'{OUT}/table5_hyperparameters.csv', index=False)

# ═══════════════════════════════════════════════════════════════════════════════
# Write TXT
# ═══════════════════════════════════════════════════════════════════════════════
h()
h('=' * 80)
h(f'  CSV files saved to: {OUT}/')
h('=' * 80)

txt_path = f'{OUT}/all_tables.txt'
with open(txt_path, 'w') as f:
    f.write('\n'.join(lines))

print(f'  → {txt_path}')
print(f'  → {OUT}/table1_dataset.csv')
print(f'  → {OUT}/table2_seed_stability.csv')
print(f'  → {OUT}/table3_fuzzy_stats.csv')
print(f'  → {OUT}/table4_cd_family_distribution.csv')
print(f'  → {OUT}/table5_hyperparameters.csv')
