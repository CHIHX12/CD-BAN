"""
generate_aggregate_attention.py
================================
Aggregate BANLayer attention over ALL label=0 (strong-binding) samples
for the 4 main cyclodextrin types (alpha / beta / gamma / HP-beta).

For each CD type:
  - Load every label=0 row from train + val + test
  - Run model forward pass  → attention [2 heads, n_guest, n_host]
  - Average over heads + guest atoms  → [n_host]  (per sample)
  - Normalise each sample to [0, 1]
  - Stack into matrix [n_samples, n_host]

Heatmap design:
  - Rows  = individual label-0 guest molecules  (Y-axis labels hidden)
  - Cols  = CD host heavy atoms
  - X-axis: ONLY the top-K CD atoms are labelled  (with mean ± std)
  - Cyan dashed vertical lines mark those top-K positions

Usage:
  python generate_aggregate_attention.py

Output:
  results/aggregate_attention/aggregate_attention_heatmap.png / .svg
  results/aggregate_attention/<cd_type>_top_atoms.csv
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
import matplotlib.gridspec as gridspec

from models import CDBAN
from dataloader import smiles_to_pyg
from torch_geometric.data import Batch

# ── Config ──────────────────────────────────────────────────────────────────
OUT_DIR  = 'results/aggregate_attention'
DATA_DIR = 'data/binary'
TOP_K    = 5          # top-K CD atoms to highlight on x-axis
os.makedirs(OUT_DIR, exist_ok=True)

CD_ORDER = ['alpha-CD', 'beta-CD', 'gamma-CD', 'HP-beta-CD']

# ── Load model ───────────────────────────────────────────────────────────────
print('Loading model (seed 49)...')
with open('configs/CDBAN.yaml') as f:
    cfg = yaml.safe_load(f)
model = CDBAN(**cfg)
pth = sorted([p for p in os.listdir('results/seed_49')
              if p.startswith('best_model')])[-1]
model.load_state_dict(
    torch.load(f'results/seed_49/{pth}', map_location='cpu', weights_only=False))
model.eval()
print(f'  Model: {pth}')


# ── Helpers ──────────────────────────────────────────────────────────────────
def get_host_attention(guest_smi: str, host_smi: str):
    """
    Single forward pass → normalised attention vector per host atom [n_h].
    Averaged over 2 heads AND all guest atoms.
    """
    g  = smiles_to_pyg(guest_smi)
    h  = smiles_to_pyg(host_smi)
    n_g, n_h = g.x.shape[0], h.x.shape[0]
    with torch.no_grad():
        _, _, _, att = model(Batch.from_data_list([g]),
                             Batch.from_data_list([h]),
                             mode='eval')
    arr = att[0].numpy()              # [2, 290, 290]
    arr = arr[:, :n_g, :n_h]          # [2, n_g, n_h]
    host_att = arr.mean(axis=(0, 1))  # [n_h]  heads + guest average
    mn, mx = host_att.min(), host_att.max()
    if mx > mn:
        host_att = (host_att - mn) / (mx - mn)
    return host_att, n_h


def classify_host(name: str):
    """Map Host column string → one of 4 canonical CD type keys."""
    h = name.lower()
    if 'hp' in h or 'hydroxypropyl' in h:
        return 'HP-beta-CD'
    if 'alpha' in h:
        return 'alpha-CD'
    if 'gamma' in h:
        return 'gamma-CD'
    if 'beta' in h:
        return 'beta-CD'
    return None


# ── Load all label=0 data ────────────────────────────────────────────────────
print('\nLoading label=0 samples from train / val / test ...')
frames = []
for split in ('train', 'val', 'test'):
    path = f'{DATA_DIR}/{split}.csv'
    df   = pd.read_csv(path)
    df['split'] = split
    frames.append(df)

all_df  = pd.concat(frames, ignore_index=True)
label0  = all_df[all_df['label'] == 0].copy()
label0['cd_type'] = label0['Host'].apply(classify_host)

print(f'  Total label=0 rows : {len(label0)}')
print(f'  CD-type breakdown  :\n{label0["cd_type"].value_counts().to_string()}')


# ── Run inference & aggregate ─────────────────────────────────────────────────
print('\nRunning model inference ...')
results = {}

for cd_type in CD_ORDER:
    subset = label0[label0['cd_type'] == cd_type].reset_index(drop=True)
    n_tot  = len(subset)
    print(f'\n  [{cd_type}]  {n_tot} label-0 samples')
    if n_tot == 0:
        continue

    rows = []
    for i, row in subset.iterrows():
        try:
            att, n_h = get_host_attention(row['SMILES_Guest'], row['SMILES_Host'])
            rows.append((att, n_h, row['SMILES_Guest']))
            if (i + 1) % 10 == 0:
                print(f'    {i+1}/{n_tot}')
        except Exception as e:
            print(f'    skip #{i}: {e}')

    if not rows:
        continue

    # Keep only samples whose n_h matches the most common length
    modal_nh = Counter(r[1] for r in rows).most_common(1)[0][0]
    kept     = [(r[0], r[2]) for r in rows if r[1] == modal_nh]
    skipped  = n_tot - len(kept)
    if skipped:
        print(f'    Skipped {skipped} rows with mismatched n_h')

    mat    = np.stack([k[0] for k in kept], axis=0)   # [n_samples, n_h]
    guests = [k[1] for k in kept]

    results[cd_type] = {
        'mat'      : mat,
        'mean'     : mat.mean(axis=0),
        'std'      : mat.std(axis=0),
        'n_h'      : modal_nh,
        'n_samples': len(kept),
        'guests'   : guests,
    }

    # Save top-K CSV
    mean = results[cd_type]['mean']
    std  = results[cd_type]['std']
    top_k_idx = np.argsort(mean)[-TOP_K:][::-1]
    df_top = pd.DataFrame({
        'rank'         : range(1, TOP_K + 1),
        'atom_index'   : top_k_idx + 1,
        'mean_attention': mean[top_k_idx].round(4),
        'std_attention' : std[top_k_idx].round(4),
    })
    safe = cd_type.replace('-', '_')
    df_top.to_csv(f'{OUT_DIR}/{safe}_top{TOP_K}_atoms.csv', index=False)
    print(f'    -> top-{TOP_K} atoms: {list(top_k_idx + 1)}')


# ── Plot ──────────────────────────────────────────────────────────────────────
available = [k for k in CD_ORDER if k in results]
n_cd      = len(available)

fig = plt.figure(figsize=(18, 4 * n_cd + 1.5))
fig.suptitle(
    'CD-BAN Aggregated Attention — label=0 (strong-binding) samples\n'
    f'(2 heads + guest-atom averaged; normalised per sample; top-{TOP_K} host atoms highlighted)',
    fontsize=12, fontweight='bold'
)
gs = gridspec.GridSpec(n_cd, 1, figure=fig, hspace=0.65)

for i, cd_type in enumerate(available):
    data   = results[cd_type]
    mat    = data['mat']        # [n_samples, n_h]
    mean   = data['mean']
    std    = data['std']
    n_h    = data['n_h']
    n_samp = data['n_samples']

    top_k = sorted(np.argsort(mean)[-TOP_K:])

    ax = fig.add_subplot(gs[i])
    im = ax.imshow(mat, cmap='YlOrRd', aspect='auto',
                   vmin=0, vmax=1, interpolation='nearest')

    ax.set_title(
        f'{cd_type}  —  N={n_samp} label-0 guests,  {n_h} host heavy atoms',
        fontsize=10, fontweight='bold'
    )

    # Y-axis: hidden (many different guest molecules)
    ax.set_ylabel(f'Label-0 guests\n(N={n_samp}, sorted by input order)',
                  fontsize=8)
    ax.set_yticks([])

    # X-axis: ONLY top-K positions with atom index + mean±std
    ax.set_xticks(top_k)
    ax.set_xticklabels(
        [f'#{idx + 1}\n{mean[idx]:.2f}±{std[idx]:.2f}' for idx in top_k],
        fontsize=7.5, fontweight='bold', color='cyan'
    )
    ax.tick_params(axis='x', colors='cyan', length=5, width=1.2)
    ax.set_xlabel(
        f'Host CD heavy-atom index  (only top-{TOP_K} labelled)',
        fontsize=8, labelpad=18
    )

    # Cyan dashed vertical lines at top-K
    for idx in top_k:
        ax.axvline(x=idx, color='cyan', linewidth=2.0,
                   linestyle='--', alpha=0.9)

    # Legend box
    ax.text(
        1.01, 1.0,
        f'— top-{TOP_K} host atoms\n'
        f'  (highest mean attention\n'
        f'   across all label-0 guests)\n\n'
        f'label format:\n'
        f'  #index\n'
        f'  mean ± std',
        transform=ax.transAxes, fontsize=6.5, color='cyan',
        va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#111111', alpha=0.8)
    )

    cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.12)
    cb.set_label('Normalised Attention', fontsize=8)

out_base = f'{OUT_DIR}/aggregate_attention_heatmap'
plt.savefig(out_base + '.png', dpi=300, bbox_inches='tight')
plt.savefig(out_base + '.svg', bbox_inches='tight')
print(f'\n-> Saved: {out_base}.png / .svg')
plt.close()
print('Done.')
