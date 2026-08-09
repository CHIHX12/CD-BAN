"""
Publication-quality figures for CD-BAN  v3
- 600 DPI, SVG + PNG, all English, journal-ready
- Fixed: crowding, text overlap, clipping
- v3: figure order updated to match paper structure
    Fig 1 = Classification performance (all metrics, all seeds)
    Fig 2 = Seed stability (line plots)
    Fig 3 = Probability gradient bar chart
    Fig 4 = Scatter hexbin (p vs log10K)
    Fig 5 = KDE distributions per bin
    Fig 6 = Violin boundary validation
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, FancyArrowPatch
from scipy.stats import pearsonr, spearmanr, gaussian_kde
import warnings
warnings.filterwarnings('ignore')
import os

import torch
sys.path.insert(0, '.')
from models import CDBAN
from dataloader import CDBinaryDataset, cd_collate_func
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import yaml

# ── Global style ──────────────────────────────────────────
plt.rcParams.update({
    'font.family':          'DejaVu Sans',
    'font.size':            16,
    'axes.titlesize':       18,
    'axes.labelsize':       17,
    'xtick.labelsize':      14,
    'ytick.labelsize':      14,
    'legend.fontsize':      13,
    'legend.title_fontsize':14,
    'axes.linewidth':       1.5,
    'xtick.major.width':    1.5,
    'ytick.major.width':    1.5,
    'xtick.major.size':     6,
    'ytick.major.size':     6,
    'axes.spines.top':      False,
    'axes.spines.right':    False,
    'figure.dpi':           600,
    'savefig.dpi':          600,
    'savefig.bbox':         'tight',
    'savefig.pad_inches':   0.15,
})

DPI    = 600
OUTDIR = 'results/figures'
os.makedirs(OUTDIR, exist_ok=True)

C_WEAK   = '#D95F02'
C_STRONG = '#1B7AC4'
C_MID    = '#4DAF4A'
C_MID2   = '#9B6BA5'
GREY     = '#AAAAAA'

def save(fig, name):
    for ext in ('svg', 'png'):
        fig.savefig(f'{OUTDIR}/{name}.{ext}', dpi=DPI, format=ext)
    print(f'  saved → {OUTDIR}/{name}.[svg|png]')

# ── Load CSV data ──────────────────────────────────────────
df_fuzzy = pd.read_csv('results/seed_stability/fuzzy_predictions.csv')
df_seeds = pd.read_csv('results/seed_stability/summary.csv')

# ── Load model weights for Fig 1 (all-seed inference) ─────
with open('configs/CDBAN.yaml') as f:
    cfg1 = yaml.safe_load(f)

device1 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df_test1 = pd.read_csv('data/binary/test.csv')
ds_test1 = CDBinaryDataset(df_test1.index.values, df_test1)
ld_test1 = DataLoader(ds_test1, batch_size=128, shuffle=False,
                      num_workers=0, collate_fn=cd_collate_func)

df_sum   = pd.read_csv('results/seed_stability/summary.csv')
seed_epochs = dict(zip(df_sum['seed'], df_sum['best_epoch']))

fpr_grid = np.linspace(0, 1, 500)
rec_grid = np.linspace(0, 1, 500)

all_tpr, all_prec = [], []
all_auroc, all_auprc = [], []
seed_roc, seed_pr = {}, {}

print('  Pre-loading 10 seed models for Fig 1...')
for seed in range(42, 52):
    ep  = int(seed_epochs[seed])
    pth = f'results/seed_{seed}/best_model_epoch_{ep}.pth'
    model1 = CDBAN(**cfg1).to(device1)
    model1.load_state_dict(torch.load(pth, map_location=device1))
    model1.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for bg, bh, yl in ld_test1:
            bg, bh = bg.to(device1), bh.to(device1)
            _, _, score, _ = model1(bg, bh, mode='eval')
            y_pred.extend(torch.sigmoid(score.squeeze(1)).cpu().tolist())
            y_true.extend(yl.tolist())
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    pre, rec, _ = precision_recall_curve(y_true, y_pred)
    tpr_i  = np.interp(fpr_grid, fpr, tpr)
    prec_i = np.interp(rec_grid, rec[::-1], pre[::-1])
    all_tpr.append(tpr_i)
    all_prec.append(prec_i)
    all_auroc.append(auc(fpr, tpr))
    all_auprc.append(auc(rec, pre))
    seed_roc[seed] = (fpr, tpr)
    seed_pr[seed]  = (rec, pre)
    print(f'    seed {seed}: AUROC={all_auroc[-1]:.4f}  AUPRC={all_auprc[-1]:.4f}')

tpr_arr  = np.array(all_tpr)
prec_arr = np.array(all_prec)
mean_tpr  = tpr_arr.mean(0);  std_tpr  = tpr_arr.std(0)
mean_prec = prec_arr.mean(0); std_prec = prec_arr.std(0)
m_auroc = np.mean(all_auroc); s_auroc = np.std(all_auroc)
m_auprc = np.mean(all_auprc); s_auprc = np.std(all_auprc)

# ═══════════════════════════════════════════════════════════
# Fig 1 — Classification performance: all metrics, all seeds
# ═══════════════════════════════════════════════════════════
print('[Fig 1] Classification performance (strip plot)')

METRICS = {
    'auroc':       ('AUROC',       all_auroc,                     C_STRONG),
    'auprc':       ('AUPRC',       all_auprc,                     C_MID),
    'f1':          ('F1 Score',    list(df_sum['f1']),             '#E8A840'),
    'sensitivity': ('Sensitivity', list(df_sum['sensitivity']),   '#9B6BA5'),
    'specificity': ('Specificity', list(df_sum['specificity']),   C_WEAK),
    'accuracy':    ('Accuracy',    list(df_sum['accuracy']),      '#555555'),
}

fig, ax = plt.subplots(figsize=(12, 6.5))
rng = np.random.default_rng(0)

x_pos = np.arange(len(METRICS))
for xi, (key, (label, vals, color)) in enumerate(METRICS.items()):
    arr = np.array(vals)
    jitter = rng.uniform(-0.18, 0.18, len(arr))
    ax.scatter(xi + jitter, arr, color=color, s=80, alpha=0.75,
               zorder=4, edgecolors='white', linewidths=0.8)
    mean_v = arr.mean(); std_v = arr.std(ddof=1)
    ax.hlines(mean_v, xi-0.28, xi+0.28, color=color,
              linewidth=3.5, zorder=5)
    ax.vlines(xi, mean_v-std_v, mean_v+std_v,
              color=color, linewidth=2.0, zorder=4)
    ax.hlines([mean_v-std_v, mean_v+std_v], xi-0.12, xi+0.12,
              color=color, linewidth=1.8, zorder=4)
    ax.text(xi, mean_v + std_v + 0.015,
            f'{mean_v:.3f}\n±{std_v:.3f}',
            ha='center', va='bottom', fontsize=11.5,
            color=color, fontweight='bold')

ax.set_xticks(x_pos)
ax.set_xticklabels([v[0] for v in METRICS.values()], fontsize=15)
ax.set_ylabel('Score', fontsize=17, labelpad=10)
ax.set_ylim(0.42, 1.12)
ax.set_xlim(-0.6, len(METRICS)-0.4)
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
ax.grid(axis='y', alpha=0.25, linewidth=0.8, zorder=0)
ax.set_title('CD-BAN Seed Stability — Performance Across 10 Independent Runs\n'
             'Test Set  (n = 120)  |  Seeds 42–51',
             fontsize=17, pad=14)

legend_elements = [
    Line2D([0], [0], color='#555555', linewidth=3.5,
           label='Mean (horizontal bar)'),
    Line2D([0], [0], color='#555555', linewidth=1.8,
           marker='_', markersize=8, linestyle='-',
           label='±1 SD  (whisker)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#888888',
           markersize=9, markeredgecolor='white',
           label='Individual seed  (n = 10)'),
]
ax.legend(handles=legend_elements,
          loc='lower right',
          framealpha=0.95, edgecolor=GREY,
          fontsize=12.5, borderpad=0.8, handlelength=2.0)

fig.tight_layout()
save(fig, 'fig1_classification_performance')
plt.close(fig)

# ═══════════════════════════════════════════════════════════
# Fig 2 — Seed stability (2 panels)
# ═══════════════════════════════════════════════════════════
print('[Fig 2] Seed stability')

seeds  = df_seeds['seed'].values
aurocs = df_seeds['auroc'].values
auprcs = df_seeds['auprc'].values
f1s    = df_seeds['f1'].values
accs   = df_seeds['accuracy'].values

fig, axes = plt.subplots(1, 2, figsize=(14, 6.0))
x = np.arange(len(seeds))

panels = [
    (aurocs, auprcs, axes[0], 'AUROC  &  AUPRC',
     [C_STRONG, C_WEAK], ['AUROC', 'AUPRC']),
    (f1s, accs,    axes[1], 'F1 Score  &  Accuracy',
     [C_MID, C_MID2], ['F1 Score', 'Accuracy']),
]

for m1, m2, ax, title, colors, mlabels in panels:
    ax.plot(x, m1, 'o-', color=colors[0], linewidth=2.2, markersize=9,
            markerfacecolor='white', markeredgewidth=2.2,
            label=f'{mlabels[0]}', zorder=4)
    ax.plot(x, m2, 's--', color=colors[1], linewidth=2.2, markersize=9,
            markerfacecolor='white', markeredgewidth=2.2,
            label=f'{mlabels[1]}', zorder=4)

    for m, c in [(m1, colors[0]), (m2, colors[1])]:
        ax.axhline(m.mean(), color=c, linewidth=1.0,
                   linestyle=':', alpha=0.6, zorder=2)
        ax.axhspan(m.mean() - m.std(ddof=1),
                   m.mean() + m.std(ddof=1),
                   color=c, alpha=0.07, zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels(seeds, fontsize=13)
    ax.set_xlabel('Random Seed', fontsize=16, labelpad=8)
    ax.set_ylabel('Score', fontsize=17, labelpad=8)
    ax.set_title(title, fontsize=17, pad=10)
    ax.set_ylim(0.45, 1.08)
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.025))
    ax.grid(axis='y', alpha=0.25, linewidth=0.8, zorder=0)

    stat_txt = (f'{mlabels[0]}: {m1.mean():.3f} ± {m1.std(ddof=1):.3f}\n'
                f'{mlabels[1]}: {m2.mean():.3f} ± {m2.std(ddof=1):.3f}')
    ax.text(0.02, 0.03, stat_txt,
            transform=ax.transAxes,
            fontsize=12, va='bottom', ha='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor=GREY, alpha=0.9))

    ax.legend(loc='lower right', framealpha=0.92,
              edgecolor=GREY, fontsize=13, borderpad=0.8)

fig.suptitle('CD-BAN Seed Stability  (10 Independent Runs, Seeds 42–51)',
             fontsize=18, fontweight='bold', y=1.02)
fig.tight_layout(w_pad=3.5)
save(fig, 'fig2_seed_stability')
plt.close(fig)

# ═══════════════════════════════════════════════════════════
# Fig 3 — Probability gradient bar chart
# ═══════════════════════════════════════════════════════════
print('[Fig 3] Probability gradient')

bins   = [2.0, 2.5, 3.0, 3.5, 4.0]
blabels = ['2.0–2.5\n(K=100–316)',
           '2.5–3.0\n(K=316–1,000)',
           '3.0–3.5\n(K=1,000–3,162)',
           '3.5–4.0\n(K=3,162–10,000)']
df_fuzzy['bin4'] = pd.cut(df_fuzzy['log10K'], bins=bins, include_lowest=True)
grp = df_fuzzy.groupby('bin4', observed=True)[['p_weak','p_strong']].mean()
ns  = df_fuzzy.groupby('bin4', observed=True).size().values

fig, ax = plt.subplots(figsize=(10, 6.5))
x = np.arange(len(blabels))
w = 0.36

b1 = ax.bar(x - w/2, grp['p_weak'].values,  w, color=C_WEAK,
            label='P(Weak Binder, K<100)',
            edgecolor='white', linewidth=0.8, zorder=3)
b2 = ax.bar(x + w/2, grp['p_strong'].values, w, color=C_STRONG,
            label='P(Strong Binder, K>10,000)',
            edgecolor='white', linewidth=0.8, zorder=3)

for bar in list(b1) + list(b2):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.015,
            f'{h:.3f}', ha='center', va='bottom',
            fontsize=12.5, fontweight='bold', color='#222222')

for i, (bar, n) in enumerate(zip(b1, ns)):
    ax.text(bar.get_x() + bar.get_width()/2, 0.015,
            f'n={n}', ha='center', va='bottom',
            fontsize=11, color='white', fontweight='bold')

ax.axhline(0.5, color='#555555', linewidth=1.5, linestyle='--',
           alpha=0.7, zorder=2, label='Decision boundary (p = 0.5)')

ax.set_xticks(x)
ax.set_xticklabels(blabels, fontsize=13.5)
ax.set_ylabel('Mean Predicted Probability', fontsize=17, labelpad=10)
ax.set_xlabel('log₁₀K Bin  (Fuzzy Zone: 100 ≤ K ≤ 10,000 M⁻¹)',
              fontsize=15, labelpad=10)
ax.set_title('CD-BAN Predicts a Continuous Affinity Gradient\n'
             'across the Fuzzy Zone (n = 1,850 compounds)', fontsize=18, pad=14)
ax.set_ylim(0, 1.10)
ax.set_xlim(-0.65, 3.65)
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
ax.legend(loc='upper right', framealpha=0.92,
          edgecolor=GREY, fontsize=13, borderpad=0.8)
ax.grid(axis='y', alpha=0.25, linewidth=0.8, zorder=0)

fig.tight_layout(rect=[0, 0, 1, 1])
save(fig, 'fig3_probability_gradient')
plt.close(fig)

# ═══════════════════════════════════════════════════════════
# Fig 4 — Scatter hex density  (redesigned: clear color separation)
# ═══════════════════════════════════════════════════════════
print('[Fig 4] Scatter p_weak vs log10K')

r, _  = pearsonr(df_fuzzy['log10K'], df_fuzzy['p_weak'])
sr, _ = spearmanr(df_fuzzy['log10K'], df_fuzzy['p_weak'])

fig, ax = plt.subplots(figsize=(7.5, 6.5))

hb = ax.hexbin(df_fuzzy['log10K'], df_fuzzy['p_weak'],
               gridsize=28, cmap='YlOrRd', mincnt=1,
               linewidths=0.2, edgecolors='white', zorder=2)
cb = fig.colorbar(hb, ax=ax, shrink=0.80, pad=0.02, aspect=22)
cb.set_label('Count', fontsize=13)
cb.ax.tick_params(labelsize=11)

m, b = np.polyfit(df_fuzzy['log10K'], df_fuzzy['p_weak'], 1)
xr   = np.linspace(2.0, 4.0, 200)
ax.plot(xr, m*xr + b, color='black', linewidth=2.5, zorder=4,
        label='OLS regression')

ax.axhline(0.5, color='crimson', linewidth=2.0, linestyle='--',
           zorder=3, label='Decision boundary  (p = 0.5)')

ax.axvline(2.0, color='#888888', linewidth=1.2, linestyle=':', alpha=0.7, zorder=1)
ax.axvline(4.0, color='#888888', linewidth=1.2, linestyle=':', alpha=0.7, zorder=1)
ax.text(2.12, 1.04, 'K = 100\n(label = 1)', color='#555555',
        fontsize=10.5, ha='left', va='top', style='italic')
ax.text(3.88, 1.04, 'K = 10,000\n(label = 0)', color='#555555',
        fontsize=10.5, ha='right', va='top', style='italic')

ax.legend(loc='upper center',
          bbox_to_anchor=(0.5, -0.18),
          ncol=2,
          framealpha=0.97, edgecolor=GREY,
          fontsize=13, borderpad=0.7,
          handlelength=1.8, handletextpad=0.5,
          columnspacing=1.2)

ax.set_xlabel('log₁₀K  (Experimental Binding Constant)', fontsize=17, labelpad=10)
ax.set_ylabel('P(Weak Binder)  — CD-BAN Output', fontsize=17, labelpad=10)
ax.set_title(f'CD-BAN Output vs Experimental Affinity\n'
             f'Fuzzy Zone  (n = 1,850)    r = {r:.3f},  ρ = {sr:.3f}',
             fontsize=16, pad=12)
ax.set_xlim(1.95, 4.05)
ax.set_ylim(-0.08, 1.08)
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

fig.tight_layout()
fig.subplots_adjust(bottom=0.23)
save(fig, 'fig4_scatter_p_vs_logK')
plt.close(fig)

# ═══════════════════════════════════════════════════════════
# Fig 5 — KDE distributions per bin
# ═══════════════════════════════════════════════════════════
print('[Fig 5] KDE distribution')

bin_defs = [
    (2.0, 2.5, C_WEAK,   'K = 100–316     (log₁₀K 2.0–2.5)'),
    (2.5, 3.0, '#E8A840', 'K = 316–1,000   (log₁₀K 2.5–3.0)'),
    (3.0, 3.5, C_MID,    'K = 1,000–3,162  (log₁₀K 3.0–3.5)'),
    (3.5, 4.0, C_STRONG, 'K = 3,162–10,000 (log₁₀K 3.5–4.0)'),
]

fig, ax = plt.subplots(figsize=(9, 6.5))
x_kde = np.linspace(0, 1, 300)
y_max = 0

for lo, hi, color, lbl in bin_defs:
    sub = df_fuzzy[(df_fuzzy['log10K'] >= lo) &
                   (df_fuzzy['log10K'] < hi)]['p_weak'].values
    if len(sub) < 10:
        continue
    kde = gaussian_kde(sub, bw_method=0.12)
    y   = kde(x_kde)
    y_max = max(y_max, y.max())
    ax.plot(x_kde, y, color=color, linewidth=2.5, label=lbl, zorder=3)
    ax.fill_between(x_kde, y, alpha=0.13, color=color, zorder=2)

ax.axvline(0.5, color='#444444', linewidth=1.8, linestyle='--',
           alpha=0.75, zorder=4)
ax.text(0.513, y_max * 0.50,
        'p = 0.5',
        fontsize=12, color='#444444',
        va='center', ha='left',
        rotation=90,
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                  edgecolor='none', alpha=0.85))

ax.set_xlabel('P(Weak Binder)', fontsize=17, labelpad=10)
ax.set_ylabel('Kernel Density Estimate', fontsize=17, labelpad=10)
ax.set_title('Predicted Probability Distributions\n'
             'by Binding Affinity Bin (Fuzzy Zone, n = 1,850)',
             fontsize=17, pad=14)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(0, y_max * 1.18)
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

ax.legend(title='log₁₀K Range', loc='upper left',
          framealpha=0.92, edgecolor=GREY,
          fontsize=12.5, borderpad=0.9,
          title_fontsize=13.5)

fig.tight_layout()
save(fig, 'fig5_kde_distribution')
plt.close(fig)

# ═══════════════════════════════════════════════════════════
# Fig 6 — Violin boundary validation
# ═══════════════════════════════════════════════════════════
print('[Fig 6] Boundary validation violin')

df_fuzzy['group'] = 'Intermediate'
df_fuzzy.loc[df_fuzzy['log10K'] < 2.5,  'group'] = 'Near Weak'
df_fuzzy.loc[df_fuzzy['log10K'] >= 3.5, 'group'] = 'Near Strong'

group_order  = ['Near Weak', 'Intermediate', 'Near Strong']
group_colors = [C_WEAK, C_MID, C_STRONG]
positions    = [1, 2, 3]

xticklabels  = ['Near Weak\nBoundary\n(K=100–316)',
                'Intermediate\n(K=316–3,162)',
                'Near Strong\nBoundary\n(K=3,162–10,000)']

fig, ax = plt.subplots(figsize=(10, 7.0))

for pos, grp_name, color in zip(positions, group_order, group_colors):
    data = df_fuzzy[df_fuzzy['group'] == grp_name]['p_weak'].values
    vp = ax.violinplot(data, positions=[pos], widths=0.65,
                       showmedians=True, showextrema=False)
    for pc in vp['bodies']:
        pc.set_facecolor(color); pc.set_alpha(0.5)
        pc.set_edgecolor(color); pc.set_linewidth(1.5)
    vp['cmedians'].set_color('#111111')
    vp['cmedians'].set_linewidth(2.8)
    ax.scatter([pos], [data.mean()], color=color, s=110, zorder=5,
               edgecolors='#111111', linewidths=1.5,
               label=f'Mean = {data.mean():.3f}')
    ax.text(pos, -0.27,
            f'n = {len(data)}\nmean = {data.mean():.3f}',
            ha='center', va='top', fontsize=13, color='#333333')

ax.axhline(0.5, color='#444444', linewidth=1.8, linestyle='--',
           alpha=0.7, zorder=3)
ax.text(3.60, 0.52, 'p = 0.5',
        fontsize=12, color='#444444', va='bottom', ha='right')

ax.annotate('Trained:\nlabel = 1\n(K < 100 M⁻¹)',
            xy=(1.32, 0.90), xytext=(1.6, 1.06),
            fontsize=11.5, color=C_WEAK, fontweight='bold', ha='left',
            arrowprops=dict(arrowstyle='->', color=C_WEAK, lw=1.5))
ax.annotate('Trained:\nlabel = 0\n(K > 10,000 M⁻¹)',
            xy=(2.68, 0.12), xytext=(2.1, -0.14),
            fontsize=11.5, color=C_STRONG, fontweight='bold', ha='left',
            arrowprops=dict(arrowstyle='->', color=C_STRONG, lw=1.5))

ax.set_xticks(positions)
ax.set_xticklabels(xticklabels, fontsize=13.5)
ax.set_ylabel('P(Weak Binder)', fontsize=17, labelpad=10)
ax.set_title('CD-BAN Generalizes Beyond Training Boundaries\n'
             'Predicted Probability at Decision Boundaries (Fuzzy Zone)',
             fontsize=17, pad=14)
ax.set_ylim(-0.42, 1.22)
ax.set_xlim(0.35, 3.65)
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax.grid(axis='y', alpha=0.2, linewidth=0.8, zorder=0)

fig.tight_layout()
save(fig, 'fig6_boundary_validation')
plt.close(fig)

print(f'\n  All 6 figures → {OUTDIR}/')
print(f'  Format: SVG + PNG @ {DPI} DPI')
