"""
generate_all_cd_aggregate.py
=============================
For each CD type (alpha / beta / gamma / HP-beta):
  1. Aggregate BANLayer attention over all label=0 samples
  2. Auto-derive atom-role mapping via RDKit ring analysis
  3. Collapse to 11 glucopyranose positions (+ HP group for HP-beta-CD)
  4. Save bar chart PNG/SVG
  5. Save PyMOL region-color .pml (Windows path, ASCII only)
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

from rdkit import Chem
from models import CDBAN
from dataloader import smiles_to_pyg
from torch_geometric.data import Batch

# ── Paths ────────────────────────────────────────────────────────────────────
OUT_DIR   = 'results/aggregate_attention'
DATA_DIR  = 'data/binary'
PDB_DIR   = 'results/pymol'
os.makedirs(OUT_DIR, exist_ok=True)

# ── CD definitions ────────────────────────────────────────────────────────────
CD_DEFS = {
    'alpha-CD': {
        'n_units': 6,
        'pdb': 'alpha_CD.pdb',
        'filter': lambda h: 'alpha' in h.lower(),
    },
    'beta-CD': {
        'n_units': 7,
        'pdb': 'beta_CD.pdb',
        'filter': lambda h: 'beta' in h.lower() and 'hp' not in h.lower() and 'hydroxypropyl' not in h.lower(),
    },
    'gamma-CD': {
        'n_units': 8,
        'pdb': 'gamma_CD.pdb',
        'filter': lambda h: 'gamma' in h.lower(),
    },
    'HP-beta-CD': {
        'n_units': 7,
        'pdb': 'HP_beta_CD.pdb',
        'filter': lambda h: 'hp' in h.lower() or 'hydroxypropyl' in h.lower(),
    },
}

# ── Region colors (same as bar chart) ────────────────────────────────────────
REGION_COLOR = {
    'backbone':   '#888888',
    'cavity':     '#E07B39',
    'primary':    '#4CAF50',
    'secondary':  '#5B9BD5',
    'HP':         '#9C6B9E',
}
REGION_RGB = {
    'backbone':  [0.533, 0.533, 0.533],
    'cavity':    [0.878, 0.482, 0.224],
    'primary':   [0.298, 0.686, 0.314],
    'secondary': [0.357, 0.608, 0.835],
    'HP':        [0.612, 0.420, 0.620],
}

ROLE_TO_REGION = {
    'C1': 'backbone', 'C4': 'backbone', 'O5': 'backbone', 'Ob': 'backbone',
    'C3': 'cavity',   'C5': 'cavity',
    'C6': 'primary',  'O6': 'primary',
    'C2': 'secondary','O2': 'secondary', 'O3': 'secondary',
    'HP': 'HP',
}

ROLE_ORDER  = ['C1', 'C2', 'C3', 'C4', 'C5', 'O5', 'C6', 'O6', 'O2', 'O3', 'Ob']
ROLE_LABELS = ['C1\n(anomeric)', 'C2', 'C3', 'C4', 'C5', 'O5\n(ring)',
               'C6', 'O6\n(1 OH)', 'O2\n(2 OH)', 'O3\n(2 OH)', 'O-bridge']


# ── Auto atom-role mapping ────────────────────────────────────────────────────
def build_role_map(smi):
    """
    Return dict: atom_idx (0-based) -> role string
    Roles: C1, C2, C3, C4, C5, O5, C6, O6, O2, O3, Ob, HP
    """
    mol = Chem.MolFromSmiles(smi)
    ring_info = mol.GetRingInfo()
    pyranose = [set(r) for r in ring_info.AtomRings() if len(r) == 6]

    role = {}

    for i, atom in enumerate(mol.GetAtoms()):
        sym = atom.GetSymbol()
        deg = atom.GetDegree()
        in_ring = atom.IsInRing()
        in_pyr = any(i in r for r in pyranose)
        nbs = list(atom.GetNeighbors())
        nb_syms = [n.GetSymbol() for n in nbs]
        nb_in_pyr = [any(n.GetIdx() in r for r in pyranose) for n in nbs]

        # --- exocyclic atoms ---
        if not in_ring:
            if sym == 'O' and deg == 1:
                # terminal OH: O6 if neighbour is CH2 (degree 2), else O2/O3
                nb = nbs[0]
                if nb.GetDegree() == 2 and not nb.IsInRing():
                    role[i] = 'O6'
                elif any(nb_in_pyr):
                    role[i] = 'O2_or_O3'   # resolved below
                else:
                    role[i] = 'HP'
            elif sym == 'C' and deg == 2 and not in_ring:
                # check if neighbour is exocyclic O6 (degree-1 O)
                has_terminal_o = any(n.GetSymbol()=='O' and n.GetDegree()==1 for n in nbs)
                has_ring_c = any(n.IsInRing() for n in nbs)
                if has_terminal_o and has_ring_c:
                    role[i] = 'C6'
                else:
                    role[i] = 'HP'
            else:
                role[i] = 'HP'
            continue

        # --- in pyranose ring ---
        if in_pyr:
            if sym == 'O':
                role[i] = 'O5'
                continue
            # C in pyranose: classify by neighbors
            o_nbs = [n for n in nbs if n.GetSymbol() == 'O']
            c_nbs = [n for n in nbs if n.GetSymbol() == 'C']
            pyr_o = [n for n in o_nbs if any(n.GetIdx() in r for r in pyranose)]   # O5
            bridge_o = [n for n in o_nbs if n.IsInRing() and not any(n.GetIdx() in r for r in pyranose)]
            exo_c = [n for n in c_nbs if not n.IsInRing()]  # C6 branch
            term_o = [n for n in o_nbs if not n.IsInRing() and n.GetDegree()==1]  # OH

            if pyr_o and bridge_o:
                role[i] = 'C1'
            elif exo_c:
                role[i] = 'C5'
            elif bridge_o and not pyr_o:
                role[i] = 'C4'
            else:
                role[i] = 'C2_C3'   # resolved below
            continue

        # --- in macrocyclic ring but NOT pyranose (glycosidic bridge O) ---
        if in_ring and not in_pyr and sym == 'O':
            role[i] = 'Ob'
            continue

        role[i] = 'HP'

    # ── Resolve O2/O3 and C2/C3 ──────────────────────────────────────────────
    # For each pyranose ring, find C1 and C4, then assign C2/C3 and O2/O3 by
    # walking the ring: C4-C3-C2-C1
    for pyr in pyranose:
        pyr = list(pyr)
        # Find C1 and C4 in this ring
        c1 = next((i for i in pyr if role.get(i) == 'C1'), None)
        c4 = next((i for i in pyr if role.get(i) == 'C4'), None)
        if c1 is None or c4 is None:
            continue

        # Walk ring from C4 toward C1 (not through O5):
        # C4 - C3 - C2 - C1
        atom_c4 = mol.GetAtomWithIdx(c4)
        ring_nbs_c4 = [n.GetIdx() for n in atom_c4.GetNeighbors()
                       if n.GetIdx() in pyr and role.get(n.GetIdx()) not in ('O5', 'C5')]
        # pick neighbour that is NOT C1
        c3_candidates = [n for n in ring_nbs_c4 if n != c1]
        if not c3_candidates:
            continue
        c3 = c3_candidates[0]
        role[c3] = 'C3'

        atom_c3 = mol.GetAtomWithIdx(c3)
        ring_nbs_c3 = [n.GetIdx() for n in atom_c3.GetNeighbors()
                       if n.GetIdx() in pyr and n.GetIdx() != c4 and role.get(n.GetIdx()) != 'O5']
        c2_candidates = [n for n in ring_nbs_c3 if n != c4]
        if not c2_candidates:
            continue
        c2 = c2_candidates[0]
        role[c2] = 'C2'

        # Assign O3 (neighbour of C3, terminal) and O2 (neighbour of C2, terminal)
        for cidx, oname in [(c3, 'O3'), (c2, 'O2')]:
            catom = mol.GetAtomWithIdx(cidx)
            for nb in catom.GetNeighbors():
                if nb.GetSymbol() == 'O' and not nb.IsInRing() and nb.GetDegree() == 1:
                    role[nb.GetIdx()] = oname

    # clean up unresolved
    for i in list(role.keys()):
        if role[i] in ('C2_C3', 'O2_or_O3'):
            role[i] = 'HP'

    return role, mol


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
    arr = att[0].numpy()[:, :n_g, :n_h]
    host_att = arr.mean(axis=(0, 1))
    mn, mx = host_att.min(), host_att.max()
    if mx > mn:
        host_att = (host_att - mn) / (mx - mn)
    return host_att, n_h


# ── Load data ─────────────────────────────────────────────────────────────────
frames = [pd.read_csv(f'{DATA_DIR}/{s}.csv') for s in ('train', 'val', 'test')]
all_df = pd.concat(frames, ignore_index=True)
label0 = all_df[all_df['label'] == 0].copy()


def write_pml(pdb_path, role_map, out_pml):
    # Use path relative to repo root so PML works cross-platform
    # Run PyMOL from the repo root: pymol -c results/aggregate_attention/xxx.pml
    pdb_abs = pdb_path.replace('\\', '/')
    # group by region
    region_atoms = {r: [] for r in REGION_RGB}
    for atom_idx, r in role_map.items():
        region = ROLE_TO_REGION.get(r, 'HP')
        region_atoms[region].append(atom_idx + 1)   # PyMOL 1-indexed

    lines = [
        f'# CD Region-colour attention map',
        f'# grey=Backbone, orange=Cavity, green=Primary rim, blue=Secondary rim',
        '',
        f'load {pdb_abs}, cd_host',
        '',
        '# Define region colours',
    ]
    for name, rgb in REGION_RGB.items():
        if region_atoms.get(name):
            lines.append(f'set_color col_{name}, [{rgb[0]:.3f}, {rgb[1]:.3f}, {rgb[2]:.3f}]')
    lines += ['', 'color col_backbone, cd_host', '']

    for region, indices in region_atoms.items():
        if region == 'backbone' or not indices:
            continue
        sel = '+'.join(str(i) for i in sorted(indices))
        lines.append(f'color col_{region}, (cd_host and index {sel})')

    lines += [
        '',
        'hide everything',
        'show surface, cd_host',
        'show sticks, cd_host',
        'set transparency, 0.30, cd_host',
        'orient',
        'zoom cd_host, 5',
        'bg_color white',
    ]
    with open(out_pml, 'w', encoding='ascii') as f:
        f.write('\n'.join(lines))


def plot_bar(cd_name, role_mat, role_mean, role_std, n_kept, has_hp, out_path):
    roles_to_plot = ROLE_ORDER + (['HP'] if has_hp else [])
    labels_to_plot = ROLE_LABELS + (['HP\ngroup'] if has_hp else [])
    colors_to_plot = [REGION_COLOR[ROLE_TO_REGION.get(r, 'HP')] for r in roles_to_plot]

    valid = [r for r in roles_to_plot if r in role_mean]
    sorted_roles = sorted(valid, key=lambda r: role_mean[r], reverse=True)
    sorted_labels = [labels_to_plot[roles_to_plot.index(r)] for r in sorted_roles]
    sorted_colors = [REGION_COLOR[ROLE_TO_REGION.get(r, 'HP')] for r in sorted_roles]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                             gridspec_kw={'width_ratios': [2, 1]})
    fig.suptitle(
        f'{cd_name} Aggregate Attention - All Label=0 (Strong-binding) Guests\n'
        f'N={n_kept} samples - glucopyranose positions - sorted by mean attention',
        fontsize=11, fontweight='bold'
    )

    # Left: sorted horizontal bar + jitter
    ax = axes[0]
    y = np.arange(len(sorted_roles))[::-1]
    means = [role_mean[r] for r in sorted_roles]
    stds  = [role_std[r]  for r in sorted_roles]

    ax.barh(y, means, xerr=stds, color=sorted_colors,
            edgecolor='black', linewidth=0.7,
            capsize=4, error_kw=dict(elinewidth=1.1, capthick=1.1), height=0.6)

    rng = np.random.default_rng(42)
    col_indices = {r: i for i, r in enumerate(roles_to_plot)}
    for j, role in enumerate(sorted_roles):
        if role in col_indices and role_mat is not None:
            ci = col_indices[role]
            if ci < role_mat.shape[1]:
                vals = role_mat[:, ci]
                jitter = rng.uniform(-0.22, 0.22, size=len(vals))
                ax.scatter(vals, y[j] + jitter,
                           color='black', alpha=0.25, s=8, zorder=3, linewidths=0)

    for j, (m, s) in enumerate(zip(means, stds)):
        ax.text(m + s + 0.015, y[j], f'{m:.2f}',
                va='center', ha='left', fontsize=9, fontweight='bold')

    ax.set_yticks(y)
    ax.set_yticklabels(sorted_labels, fontsize=10)
    ax.set_xlabel('Mean Normalised Attention  (dots = individual guests)', fontsize=10)
    ax.set_xlim(0, 1.18)
    ax.set_title('Mean +/- Std per position', fontsize=10)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    legend_patches = [
        mpatches.Patch(color=REGION_COLOR['backbone'],  label='Backbone (C1, C4, O5, O-bridge)'),
        mpatches.Patch(color=REGION_COLOR['cavity'],    label='Cavity-facing (C3, C5)'),
        mpatches.Patch(color=REGION_COLOR['primary'],   label='Primary rim - narrow (C6, O6)'),
        mpatches.Patch(color=REGION_COLOR['secondary'], label='Secondary rim - wide (C2, O2, O3)'),
    ]
    if has_hp:
        legend_patches.append(mpatches.Patch(color=REGION_COLOR['HP'], label='HP substituent'))
    ax.legend(handles=legend_patches, fontsize=8.5, loc='lower right',
              framealpha=0.9, edgecolor='gray')

    # Right: region summary
    ax2 = axes[1]
    region_groups = {
        'Backbone\n(C1,C4,O5,Ob)': ['C1', 'C4', 'O5', 'Ob'],
        'Cavity\n(C3, C5)':        ['C3', 'C5'],
        'Primary rim\n(C6, O6)':   ['C6', 'O6'],
        'Secondary rim\n(O2,O3)':  ['O2', 'O3'],
    }
    reg_colors = [REGION_COLOR['backbone'], REGION_COLOR['cavity'],
                  REGION_COLOR['primary'],  REGION_COLOR['secondary']]
    if has_hp:
        region_groups['HP group'] = ['HP']
        reg_colors.append(REGION_COLOR['HP'])

    reg_labels, reg_means, reg_stds = [], [], []
    for label, roles in region_groups.items():
        vals = []
        for r in roles:
            if r in col_indices and role_mat is not None:
                ci = col_indices[r]
                if ci < role_mat.shape[1]:
                    vals.extend(role_mat[:, ci].tolist())
        if vals:
            reg_labels.append(label)
            reg_means.append(np.mean(vals))
            reg_stds.append(np.std(vals))

    xr = np.arange(len(reg_labels))
    ax2.bar(xr, reg_means, yerr=reg_stds,
            color=reg_colors[:len(reg_labels)], edgecolor='black', linewidth=0.8,
            capsize=6, error_kw=dict(elinewidth=1.2, capthick=1.2), width=0.55)
    ax2.set_xticks(xr)
    ax2.set_xticklabels(reg_labels, fontsize=8.5)
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
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.savefig(out_path.replace('.png', '.svg'), bbox_inches='tight')
    plt.close()


# ── Main loop ─────────────────────────────────────────────────────────────────
for cd_name, cfg_cd in CD_DEFS.items():
    print(f'\n{"="*55}')
    print(f'  {cd_name}')
    print(f'{"="*55}')

    subset = label0[label0['Host'].apply(cfg_cd['filter'])].reset_index(drop=True)
    if len(subset) == 0:
        print('  No label=0 samples, skip.')
        continue
    print(f'  {len(subset)} label=0 samples')

    # Get one SMILES to build role map
    host_smi_sample = subset.iloc[0]['SMILES_Host']
    role_map, mol = build_role_map(host_smi_sample)
    n_atoms = mol.GetNumAtoms()
    print(f'  Atoms: {n_atoms}  |  Role breakdown: '
          + str({r: sum(1 for v in role_map.values() if v==r) for r in set(role_map.values())}))

    # Build roles_to_plot list (columns of role_mat)
    has_hp = any(v == 'HP' for v in role_map.values())
    roles_for_mat = ROLE_ORDER + (['HP'] if has_hp else [])

    # Inference
    rows = []
    for i, row in subset.iterrows():
        try:
            att, n_h = get_host_attention(row['SMILES_Guest'], row['SMILES_Host'])
            rows.append((att, n_h))
            if (len(rows)) % 10 == 0:
                print(f'  {len(rows)}/{len(subset)}')
        except Exception as e:
            print(f'  skip #{i}: {e}')

    modal_nh = Counter(r[1] for r in rows).most_common(1)[0][0]
    kept = [r[0] for r in rows if r[1] == modal_nh]
    print(f'  Kept {len(kept)}/{len(rows)} (n_h={modal_nh})')

    mat  = np.stack(kept, axis=0)
    mean_all = mat.mean(axis=0)

    # Collapse to role_mat
    role_mat  = np.zeros((len(kept), len(roles_for_mat)))
    role_mean = {}
    role_std  = {}
    for j, role in enumerate(roles_for_mat):
        atom_indices = [idx for idx, r in role_map.items() if r == role and idx < modal_nh]
        if atom_indices:
            role_mat[:, j] = mat[:, atom_indices].mean(axis=1)
            role_mean[role] = role_mat[:, j].mean()
            role_std[role]  = role_mat[:, j].std()

    # Print summary
    print('  Role mean attention:')
    for r in sorted(role_mean, key=role_mean.get, reverse=True):
        print(f'    {r:12s}: {role_mean[r]:.3f} +/- {role_std[r]:.3f}')

    # PyMOL script
    safe = cd_name.replace('-', '_').replace(' ', '_')
    pdb_path = os.path.join(PDB_DIR, cfg_cd['pdb'])
    pml_path = os.path.join(OUT_DIR, f'{safe}_region_color.pml')
    if os.path.exists(pdb_path):
        write_pml(pdb_path, role_map, pml_path)
        print(f'  -> PyMOL: {pml_path}')
    else:
        print(f'  [WARN] PDB not found: {pdb_path}')

    # Bar chart
    chart_path = os.path.join(OUT_DIR, f'{safe}_aggregate_barplot.png')
    plot_bar(cd_name, role_mat, role_mean, role_std, len(kept), has_hp, chart_path)
    print(f'  -> Chart: {chart_path}')

print('\nAll done.')
