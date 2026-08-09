"""
plot_attention.py — BANLayer Attention Heatmap + PyMOL Coloring Script
=======================================================================
Extracts BANLayer attention weights from a trained CD-BAN model and
produces a Guest × Host attention heatmap and a PyMOL coloring script.

Usage:
  python plot_attention.py \\
    --guest "CC(c1ccc2cccc(OC)c2c1)C(=O)O" \\
    --host  "<beta-CD SMILES>" \\
    --guest_name "Naproxen" \\
    --host_name  "beta-CD" \\
    --out   results/figures/fig_attention

Outputs:
  fig_attention_heatmap.png/svg  — attention heatmap
  fig_attention_pymol.pml        — PyMOL coloring script
"""

import argparse, sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

import torch
import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd

from models import CDBAN
from dataloader import smiles_to_pyg
from torch_geometric.data import Batch
from rdkit import Chem
from rdkit.Chem import Descriptors

DEFAULT_SEED   = 49
COLORMAP_ATTN  = 'YlOrRd'
COLORMAP_PYMOL = 'hot'


def load_model(seed: int = DEFAULT_SEED) -> CDBAN:
    with open('configs/CDBAN.yaml') as f:
        cfg = yaml.safe_load(f)
    model = CDBAN(**cfg)
    seed_dir = f'results/seed_{seed}'
    pth = sorted([p for p in os.listdir(seed_dir) if p.startswith('best_model')])[-1]
    model.load_state_dict(torch.load(
        os.path.join(seed_dir, pth), map_location='cpu', weights_only=False))
    model.eval()
    return model


def get_atom_symbols(smiles: str) -> list:
    """Return atom symbol + index labels for heatmap axes."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [f'A{i}' for i in range(100)]
    return [f'{a.GetSymbol()}{a.GetIdx()}' for a in mol.GetAtoms()]


def extract_attention(model: CDBAN, smiles_guest: str, smiles_host: str):
    """
    Returns:
      z_bin   : model logit
      att_avg : mean attention matrix [n_guest, n_host] (averaged over heads)
      n_guest : number of guest atoms
      n_host  : number of host atoms
    """
    g = smiles_to_pyg(smiles_guest)
    h = smiles_to_pyg(smiles_host)
    bg = Batch.from_data_list([g])
    bh = Batch.from_data_list([h])
    n_guest = g.x.shape[0]
    n_host  = h.x.shape[0]

    with torch.no_grad():
        v_g, v_h, score, att = model(bg, bh, mode='eval')

    z = score.item()
    att_np    = att[0].numpy()                        # [n_heads, 290, 290]
    att_valid = att_np[:, :n_guest, :n_host]           # [n_heads, n_guest, n_host]
    att_avg   = att_valid.mean(axis=0)                 # [n_guest, n_host]
    mn, mx = att_avg.min(), att_avg.max()
    if mx > mn:
        att_avg = (att_avg - mn) / (mx - mn)
    return z, att_avg, n_guest, n_host


def plot_heatmap(att_avg, guest_labels, host_labels,
                 guest_name, host_name, out_prefix, z_bin):
    """Plot Guest × Host attention heatmap."""
    n_g, n_h = att_avg.shape
    fig_w = max(10, n_h * 0.22)
    fig_h = max(6,  n_g * 0.35)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(att_avg, cmap=COLORMAP_ATTN, aspect='auto',
                   vmin=0, vmax=1, interpolation='nearest')

    ax.set_xticks(range(n_h))
    ax.set_yticks(range(n_g))
    ax.set_xticklabels(host_labels[:n_h], rotation=90, fontsize=6)
    ax.set_yticklabels(guest_labels[:n_g], fontsize=7)
    ax.set_xlabel(f'Host ({host_name}, {n_h} atoms)', fontsize=11)
    ax.set_ylabel(f'Guest ({guest_name}, {n_g} atoms)', fontsize=11)

    p_weak = 1 / (1 + np.exp(-z_bin))
    title  = (f'CD-BAN Bilinear Attention Map  |  {guest_name} / {host_name}\n'
              f'z_bin = {z_bin:+.4f}   P(Weak) = {p_weak:.4f}   '
              f'(Heads = 2, averaged)')
    ax.set_title(title, fontsize=11, pad=10)

    cb = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label('Normalised Attention Weight', fontsize=9)

    plt.tight_layout()
    for ext in ('png', 'svg'):
        path = f'{out_prefix}_heatmap.{ext}'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f'-> Saved: {path}')
    plt.close()


def write_pymol_script(att_avg, host_labels, host_name, out_prefix,
                       host_smiles):
    """
    Generate a PyMOL coloring script.
    Maps per-atom host attention (max over guest atoms) to B-factor,
    then colors by spectrum. Assumes the CD structure is loaded in PyMOL
    as object 'cd_host'.
    """
    host_attn = att_avg.max(axis=0)   # [n_host]

    lines = [
        f'# CD-BAN Attention Coloring Script',
        f'# Host: {host_name}',
        f'# Guest attention mapped to B-factor, then colored by spectrum',
        f'# Usage: run this .pml file in PyMOL after loading your CD structure',
        f'#   load your_cd_structure.pdb, cd_host',
        f'#   @{out_prefix}_pymol.pml',
        '',
        '# Reset B-factors to 0',
        'alter cd_host, b=0',
        '',
        '# Set B-factor per atom index (0-based)',
    ]

    for idx, score in enumerate(host_attn):
        lines.append(f'alter (cd_host and index {idx+1}), b={score:.6f}')

    lines += [
        '',
        '# Color by B-factor (attention score)',
        'spectrum b, blue_white_red, cd_host, minimum=0, maximum=1',
        '',
        '# Optional: show surface + sticks',
        'show surface, cd_host',
        'show sticks, cd_host',
        'set transparency, 0.3, cd_host',
        '',
        '# Save session',
        f'# save {out_prefix}_pymol_session.pse',
    ]

    path = f'{out_prefix}_pymol.pml'
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'-> Saved: {path}')

    df = pd.DataFrame({
        'atom_index_1based': range(1, len(host_attn)+1),
        'atom_symbol':       host_labels[:len(host_attn)],
        'attention_max':     host_attn,
        'attention_norm':    host_attn,
    })
    csv_path = f'{out_prefix}_host_attention.csv'
    df.to_csv(csv_path, index=False)
    print(f'-> Saved: {csv_path}')


def main():
    ap = argparse.ArgumentParser(
        description='CD-BAN BANLayer Attention Heatmap + PyMOL Script')
    ap.add_argument('--guest',       type=str, required=True,  help='Guest SMILES')
    ap.add_argument('--host',        type=str, required=True,  help='Host SMILES')
    ap.add_argument('--guest_name',  type=str, default='Drug', help='Guest name')
    ap.add_argument('--host_name',   type=str, default='CD',   help='Host name')
    ap.add_argument('--out',         type=str, default='results/figures/fig_attention',
                                     help='Output path prefix')
    ap.add_argument('--seed',        type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print(f'[CD-BAN Attention Analysis]')
    print(f'  Guest: {args.guest_name}')
    print(f'  Host : {args.host_name}')
    print(f'  Seed : {args.seed}')

    model       = load_model(args.seed)
    guest_labels = get_atom_symbols(args.guest)
    host_labels  = get_atom_symbols(args.host)

    z_bin, att_avg, n_g, n_h = extract_attention(
        model, args.guest, args.host)

    print(f'  z_bin = {z_bin:+.4f}  |  P(Weak) = {1/(1+np.exp(-z_bin)):.4f}')
    print(f'  Attention map: {n_g} guest atoms × {n_h} host atoms')

    plot_heatmap(att_avg, guest_labels[:n_g], host_labels[:n_h],
                 args.guest_name, args.host_name, args.out, z_bin)

    write_pymol_script(att_avg, host_labels[:n_h],
                       args.host_name, args.out, args.host)

    print('\nOutputs:')
    print(f'  Heatmap : {args.out}_heatmap.png/.svg')
    print(f'  PyMOL   : load CD PDB as cd_host, then run: @{args.out}_pymol.pml')
    print(f'  CSV     : {args.out}_host_attention.csv (per-atom attention scores)')


if __name__ == '__main__':
    main()
