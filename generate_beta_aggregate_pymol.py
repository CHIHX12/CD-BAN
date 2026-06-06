"""
generate_beta_aggregate_pymol.py
=================================
Aggregate BANLayer attention over ALL label=0 beta-CD samples,
then map to:
  1. PyMOL .pml  (aggregate mean attention → B-factor)
  2. Per-atom-type bar chart (11 glucopyranose positions)

Beta-CD atom-role mapping (77 heavy atoms, 7 units × 11):
  Each glucopyranose unit: C1, C2, C3, C4, C5, O5, C6, O6, O2, O3, O-bridge
"""

import os, sys, warnings, torch, yaml
import numpy as np
import pandas as pd
from collections import Counter
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from models import CDBAN
from dataloader import smiles_to_pyg
from torch_geometric.data import Batch

# ── Config ───────────────────────────────────────────────────────────────────
OUT_DIR  = 'results/aggregate_attention'
DATA_DIR = 'data/binary'
PDB_PATH = 'results/pymol/beta_CD.pdb'
os.makedirs(OUT_DIR, exist_ok=True)

BETA_CD_SMI = (
    'OC[C@H]1O[C@@H]2O[C@@H]3[C@@H](CO)O[C@H](O[C@@H]4[C@@H](CO)O[C@H]('
    'O[C@@H]5[C@@H](CO)O[C@H](O[C@@H]6[C@@H](CO)O[C@H](O[C@@H]7[C@@H](CO)'
    'O[C@H](O[C@@H]8[C@@H](CO)O[C@H](O[C@H]1[C@H](O)[C@H]2O)[C@H](O)[C@H]'
    '8O)[C@H](O)[C@H]7O)[C@H](O)[C@H]6O)[C@H](O)[C@H]5O)[C@H](O)[C@H]4O)'
    '[C@H](O)[C@H]3O'
)

# ── Beta-CD atom-role mapping (0-indexed) ────────────────────────────────────
# Derived from RDKit ring analysis: 7 pyranose rings × 11 atoms each
# Roles: C1(anomeric), C2, C3, C4, C5, O5(ring-O), C6, O6(primary-OH),
#        O2(secondary-OH), O3(secondary-OH), Ob(glycosidic bridge)
ROLE_MAP = {
    # Unit 1  (pyranose ring: 2,3,4,48,49,51)
    0: ('O6', 1),  1: ('C6', 1),  2: ('C5', 1),  3: ('O5', 1),
    4: ('C1', 1),  5: ('Ob',  1), 48: ('C4', 1), 49: ('C3', 1),
    50: ('O3', 1), 51: ('C2', 1), 52: ('O2', 1),
    # Unit 2  (ring: 6,7,10,11,73,75)
    6: ('C4', 2),  7: ('C5', 2),  8: ('C6', 2),  9: ('O6', 2),
    10: ('O5', 2), 11: ('C1', 2), 12: ('Ob',  2), 73: ('C2', 2),
    74: ('O2', 2), 75: ('C3', 2), 76: ('O3', 2),
    # Unit 3  (ring: 13,14,17,18,69,71)
    13: ('C4', 3), 14: ('C5', 3), 15: ('C6', 3), 16: ('O6', 3),
    17: ('O5', 3), 18: ('C1', 3), 19: ('Ob',  3), 69: ('C2', 3),
    70: ('O2', 3), 71: ('C3', 3), 72: ('O3', 3),
    # Unit 4  (ring: 20,21,24,25,65,67)
    20: ('C4', 4), 21: ('C5', 4), 22: ('C6', 4), 23: ('O6', 4),
    24: ('O5', 4), 25: ('C1', 4), 26: ('Ob',  4), 65: ('C2', 4),
    66: ('O2', 4), 67: ('C3', 4), 68: ('O3', 4),
    # Unit 5  (ring: 27,28,31,32,61,63)
    27: ('C4', 5), 28: ('C5', 5), 29: ('C6', 5), 30: ('O6', 5),
    31: ('O5', 5), 32: ('C1', 5), 33: ('Ob',  5), 61: ('C2', 5),
    62: ('O2', 5), 63: ('C3', 5), 64: ('O3', 5),
    # Unit 6  (ring: 34,35,38,39,57,59)
    34: ('C4', 6), 35: ('C5', 6), 36: ('C6', 6), 37: ('O6', 6),
    38: ('O5', 6), 39: ('C1', 6), 40: ('Ob',  6), 57: ('C2', 6),
    58: ('O2', 6), 59: ('C3', 6), 60: ('O3', 6),
    # Unit 7  (ring: 41,42,45,46,53,55)
    41: ('C4', 7), 42: ('C5', 7), 43: ('C6', 7), 44: ('O6', 7),
    45: ('O5', 7), 46: ('C1', 7), 47: ('Ob',  7), 53: ('C2', 7),
    54: ('O2', 7), 55: ('C3', 7), 56: ('O3', 7),
}

# Display order and region colors for bar chart
ROLE_ORDER  = ['C1', 'C2', 'C3', 'C4', 'C5', 'O5', 'C6', 'O6', 'O2', 'O3', 'Ob']
ROLE_LABELS = ['C1\n(anomeric)', 'C2', 'C3', 'C4', 'C5', 'O5\n(ring)', 'C6',
               'O6\n(1° OH)', 'O2\n(2° OH)', 'O3\n(2° OH)', 'O-bridge']

REGION_COLOR = {
    'C1': '#888888',   # backbone
    'C4': '#888888',
    'O5': '#888888',
    'Ob': '#888888',
    'C5': '#E07B39',   # cavity-facing
    'C3': '#E07B39',
    'C6': '#4CAF50',   # primary rim
    'O6': '#4CAF50',
    'C2': '#5B9BD5',   # secondary rim
    'O2': '#5B9BD5',
    'O3': '#5B9BD5',
}

# ── Load model ────────────────────────────────────────────────────────────────
print('Loading model (seed 49)...')
with open('configs/CDBAN.yaml') as f:
    cfg = yaml.safe_load(f)
model = CDBAN(**cfg)
pth = sorted([p for p in os.listdir('results/seed_49') if p.startswith('best_model')])[-1]
model.load_state_dict(
    torch.load(f'results/seed_49/{pth}', map_location='cpu', weights_only=False))
model.eval()


def get_host_attention(guest_smi, host_smi):
    g = smiles_to_pyg(guest_smi)
    h = smiles_to_pyg(host_smi)
    n_g, n_h = g.x.shape[0], h.x.shape[0]
    with torch.no_grad():
        _, _, _, att = model(Batch.from_data_list([g]),
                             Batch.from_data_list([h]), mode='eval')
    arr = att[0].numpy()[:, :n_g, :n_h]   # [2, n_g, n_h]
    host_att = arr.mean(axis=(0, 1))        # [n_h]
    mn, mx = host_att.min(), host_att.max()
    if mx > mn:
        host_att = (host_att - mn) / (mx - mn)
    return host_att, n_h


# ── Load all label=0 beta-CD data ─────────────────────────────────────────────
print('\nLoading label=0 beta-CD samples...')
frames = []
for split in ('train', 'val', 'test'):
    df = pd.read_csv(f'{DATA_DIR}/{split}.csv')
    frames.append(df)
all_df = pd.concat(frames, ignore_index=True)

def is_beta(name):
    n = name.lower()
    return 'beta' in n and 'hp' not in n and 'hydroxypropyl' not in n

subset = all_df[(all_df['label'] == 0) &
                all_df['Host'].apply(is_beta)].reset_index(drop=True)
print(f'  {len(subset)} label=0 beta-CD rows')


# ── Run inference ─────────────────────────────────────────────────────────────
print('\nRunning inference...')
rows = []
for i, row in subset.iterrows():
    try:
        att, n_h = get_host_attention(row['SMILES_Guest'], row['SMILES_Host'])
        rows.append((att, n_h))
        if (i + 1) % 10 == 0:
            print(f'  {i+1}/{len(subset)}')
    except Exception as e:
        print(f'  skip #{i}: {e}')

modal_nh = Counter(r[1] for r in rows).most_common(1)[0][0]
kept = [r[0] for r in rows if r[1] == modal_nh]
print(f'  Kept {len(kept)}/{len(rows)} samples (n_h={modal_nh})')

mat  = np.stack(kept, axis=0)    # [n_samples, 77]
mean = mat.mean(axis=0)          # [77]
std  = mat.std(axis=0)


# ── Region assignment ─────────────────────────────────────────────────────────
ROLE_TO_REGION = {
    'C1': 'backbone', 'C4': 'backbone', 'O5': 'backbone', 'Ob': 'backbone',
    'C3': 'cavity',   'C5': 'cavity',
    'C6': 'primary',  'O6': 'primary',
    'C2': 'secondary','O2': 'secondary', 'O3': 'secondary',
}
# PyMOL RGB (0–1 floats), same hex as bar chart
REGION_RGB = {
    'backbone': [0.533, 0.533, 0.533],   # #888888
    'cavity':   [0.878, 0.482, 0.224],   # #E07B39
    'primary':  [0.298, 0.686, 0.314],   # #4CAF50
    'secondary':[0.357, 0.608, 0.835],   # #5B9BD5
}

def write_region_pymol(pdb_path, out_pml):
    pdb_abs = os.path.abspath(pdb_path).replace('\\', '/')
    lines = [
        '# Beta-CD - Region-colour attention map',
        '# Colors match bar chart regions:',
        '#   grey   = Backbone  (C1, C4, O5, O-bridge)',
        '#   orange = Cavity-facing  (C3, C5)',
        '#   green  = Primary rim - narrow  (C6, O6)',
        '#   blue   = Secondary rim - wide  (C2, O2, O3)',
        '',
        f'load {pdb_abs}, cd_host',
        '',
        '# Define region colours',
    ]
    for name, rgb in REGION_RGB.items():
        lines.append(f'set_color col_{name}, [{rgb[0]:.3f}, {rgb[1]:.3f}, {rgb[2]:.3f}]')
    lines += ['', '# Colour backbone first (default)', 'color col_backbone, cd_host', '']

    # Group atoms by region for compact selection strings
    region_atoms = {r: [] for r in REGION_RGB}
    for atom_idx, (role, unit) in ROLE_MAP.items():
        region = ROLE_TO_REGION.get(role)
        if region:
            region_atoms[region].append(atom_idx + 1)   # PyMOL is 1-indexed

    for region, indices in region_atoms.items():
        if region == 'backbone':
            continue   # already set above
        sel = '+'.join(str(i) for i in sorted(indices))
        lines.append(f'color col_{region}, (cd_host and index {sel})')

    lines += [
        '',
        '# Display',
        'hide everything',
        'show surface, cd_host',
        'show sticks, cd_host',
        'set transparency, 0.30, cd_host',
        'orient',
        'zoom cd_host, 5',
        'bg_color white',
    ]
    with open(out_pml, 'w') as f:
        f.write('\n'.join(lines))

# Write both WSL and Windows path versions
pml_path     = os.path.join(OUT_DIR, 'beta_CD_region_color.pml')
pml_path_win = os.path.join(OUT_DIR, 'beta_CD_region_color_win.pml')
write_region_pymol(PDB_PATH, pml_path)

# Windows version: swap /mnt/d/ → D:/
with open(pml_path) as f:
    content = f.read()
with open(pml_path_win, 'w') as f:
    f.write(content.replace('/mnt/d/', 'D:/'))

print(f'\n-> PyMOL region-color script: {pml_path_win}')


# ── Per-atom-type aggregation ─────────────────────────────────────────────────
role_vals = {r: [] for r in ROLE_ORDER}
for atom_idx, (role, unit) in ROLE_MAP.items():
    if atom_idx < len(mean):
        role_vals[role].append(mean[atom_idx])

role_mean = {r: np.mean(v) for r, v in role_vals.items()}
role_std  = {r: np.std(v)  for r, v in role_vals.items()}

# Also compute per-unit (average of all 11 atoms in each unit)
unit_atoms = {u: [] for u in range(1, 8)}
for atom_idx, (role, unit) in ROLE_MAP.items():
    if atom_idx < len(mean):
        unit_atoms[unit].append(mean[atom_idx])
unit_mean = {u: np.mean(v) for u, v in unit_atoms.items()}
unit_std  = {u: np.std(v)  for u, v in unit_atoms.items()}


# ── Build collapsed heatmap matrix [n_samples, 11 roles] ─────────────────────
# Each cell = mean attention of that role across all 7 units, for that sample
role_mat = np.zeros((len(kept), len(ROLE_ORDER)))
for j, role in enumerate(ROLE_ORDER):
    atom_indices = [idx for idx, (r, u) in ROLE_MAP.items() if r == role and idx < modal_nh]
    role_mat[:, j] = mat[:, atom_indices].mean(axis=1)

# ── Sort positions by mean attention (high → low) ─────────────────────────────
sorted_roles = sorted(ROLE_ORDER, key=lambda r: role_mean[r], reverse=True)
sorted_labels = [ROLE_LABELS[ROLE_ORDER.index(r)] for r in sorted_roles]

# ── Plot: sorted bar + strip ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                         gridspec_kw={'width_ratios': [2, 1]})
fig.suptitle(
    'Beta-CD Aggregate Attention — All Label=0 (Strong-binding) Guests\n'
    f'N={len(kept)} samples · 11 glucopyranose positions · sorted by mean attention',
    fontsize=11, fontweight='bold'
)

# ── Left: horizontal bar chart (sorted) ───────────────────────────────────────
ax = axes[0]
y = np.arange(len(sorted_roles))[::-1]   # top = highest
means = [role_mean[r] for r in sorted_roles]
stds  = [role_std[r]  for r in sorted_roles]
colors = [REGION_COLOR[r] for r in sorted_roles]

bars = ax.barh(y, means, xerr=stds,
               color=colors, edgecolor='black', linewidth=0.7,
               capsize=4, error_kw=dict(elinewidth=1.1, capthick=1.1),
               height=0.6)

# Jitter individual guest points
rng = np.random.default_rng(42)
for j, role in enumerate(sorted_roles):
    col_idx = ROLE_ORDER.index(role)
    vals = role_mat[:, col_idx]
    jitter = rng.uniform(-0.22, 0.22, size=len(vals))
    ax.scatter(vals, y[j] + jitter,
               color='black', alpha=0.25, s=8, zorder=3, linewidths=0)

# Value labels
for j, (m, s) in enumerate(zip(means, stds)):
    ax.text(m + s + 0.015, y[j], f'{m:.2f}',
            va='center', ha='left', fontsize=9, fontweight='bold')

ax.set_yticks(y)
ax.set_yticklabels(sorted_labels, fontsize=10)
ax.set_xlabel('Mean Normalised Attention  (dots = individual guests)', fontsize=10)
ax.set_xlim(0, 1.18)
ax.set_title('Mean ± Std per position', fontsize=10)
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

legend_patches = [
    mpatches.Patch(color='#888888', label='Backbone (C1, C4, O5, O-bridge)'),
    mpatches.Patch(color='#E07B39', label='Cavity-facing (C3, C5)'),
    mpatches.Patch(color='#4CAF50', label='Primary rim — narrow (C6, O6)'),
    mpatches.Patch(color='#5B9BD5', label='Secondary rim — wide (C2, O2, O3)'),
]
ax.legend(handles=legend_patches, fontsize=8.5, loc='lower right',
          framealpha=0.9, edgecolor='gray')

# ── Right: region summary (grouped average) ───────────────────────────────────
ax2 = axes[1]
region_groups = {
    'Backbone\n(C1,C4,O5,Ob)': ['C1', 'C4', 'O5', 'Ob'],
    'Cavity\n(C3, C5)':        ['C3', 'C5'],
    'Primary rim\n(C6, O6)':   ['C6', 'O6'],
    'Secondary rim\n(O2,O3)':  ['O2', 'O3'],
}
region_colors_list = ['#888888', '#E07B39', '#4CAF50', '#5B9BD5']

reg_labels, reg_means, reg_stds = [], [], []
for label, roles in region_groups.items():
    vals = np.concatenate([role_mat[:, ROLE_ORDER.index(r)] for r in roles])
    reg_labels.append(label)
    reg_means.append(vals.mean())
    reg_stds.append(vals.std())

xr = np.arange(len(reg_labels))
ax2.bar(xr, reg_means, yerr=reg_stds,
        color=region_colors_list, edgecolor='black', linewidth=0.8,
        capsize=6, error_kw=dict(elinewidth=1.2, capthick=1.2),
        width=0.55)
ax2.set_xticks(xr)
ax2.set_xticklabels(reg_labels, fontsize=9)
ax2.set_ylabel('Mean Normalised Attention', fontsize=10)
ax2.set_ylim(0, 1.15)
ax2.set_title('By chemical region', fontsize=10)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
for i, (m, s) in enumerate(zip(reg_means, reg_stds)):
    ax2.text(i, m + s + 0.02, f'{m:.2f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
chart_path = os.path.join(OUT_DIR, 'beta_CD_aggregate_barplot.png')
plt.savefig(chart_path, dpi=300, bbox_inches='tight')
plt.savefig(chart_path.replace('.png', '.svg'), bbox_inches='tight')
plt.close()
print(f'-> Bar chart: {chart_path}')

# ── Save per-role CSV ─────────────────────────────────────────────────────────
df_role = pd.DataFrame({
    'position': ROLE_ORDER,
    'label': ROLE_LABELS,
    'mean_attention': [role_mean[r] for r in ROLE_ORDER],
    'std_attention':  [role_std[r]  for r in ROLE_ORDER],
    'n_atoms': [len(role_vals[r]) for r in ROLE_ORDER],
})
df_role.to_csv(os.path.join(OUT_DIR, 'beta_CD_per_role_attention.csv'), index=False)

df_unit = pd.DataFrame({
    'unit': list(range(1, 8)),
    'mean_attention': [unit_mean[u] for u in range(1, 8)],
    'std_attention':  [unit_std[u]  for u in range(1, 8)],
})
df_unit.to_csv(os.path.join(OUT_DIR, 'beta_CD_per_unit_attention.csv'), index=False)

print('\nPer-role summary:')
print(df_role[['position', 'mean_attention', 'std_attention']].to_string(index=False))
print('\nDone.')
