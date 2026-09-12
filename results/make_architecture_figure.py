"""
make_architecture_figure.py
==========================
Reviewer R2-minor9: "An additional figure demonstrating the model architecture
and the encoding would be helpful."

Produces a clean schematic of CD-BAN: the 2D SMILES -> atom-encoding -> dual-branch
GCN -> bilinear attention -> decoder flow, plus the multi-task regression head.
Saved at 600 dpi to manuscript_figures/Fig_1_architecture_encoding.png/.svg.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT = os.path.join(ROOT, 'manuscript_figures')
os.makedirs(OUT, exist_ok=True)

fig, ax = plt.subplots(figsize=(15, 8.5))
ax.set_xlim(0, 150); ax.set_ylim(0, 85); ax.axis('off')

C_GUEST = '#2a7f9e'; C_HOST = '#c0392b'; C_ATT = '#8e44ad'; C_DEC = '#27ae60'; C_REG = '#e67e22'

def box(x, y, w, h, text, fc, ec, fs=10, tc='white', bold=True, alpha=1.0):
    p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.6,rounding_size=1.5',
                       fc=fc, ec=ec, lw=1.6, alpha=alpha)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fs,
            color=tc, fontweight='bold' if bold else 'normal', wrap=True)

def arrow(x1, y1, x2, y2, color='grey', style='-|>', lw=1.8, ls='-'):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=18,
                        color=color, lw=lw, linestyle=ls)
    ax.add_patch(a)

# Title
ax.text(75, 82, 'CD-BAN: architecture and 2D atom encoding', ha='center', va='center',
        fontsize=15, fontweight='bold')

# --- Inputs ---
box(4, 60, 26, 12, 'Guest SMILES\n(drug)', C_GUEST, C_GUEST, fs=11)
box(4, 28, 26, 12, 'Host SMILES\n(cyclodextrin)', C_HOST, C_HOST, fs=11)

# --- Encoding (shared) ---
box(38, 42, 30, 18, 'Atom encoding\n74-dim per atom:\nelement · degree · valence\ncharge · hybridisation\naromaticity · H-count', '#5d6d7e', '#5d6d7e', fs=9.5)
arrow(30, 66, 38, 54, C_GUEST)
arrow(30, 34, 38, 46, C_HOST)

# --- Dual-branch GCN ---
box(76, 60, 30, 12, 'Guest GCN\n3× GCNConv 128→128→128\n+ mean pool → v_g (128)', C_GUEST, C_GUEST, fs=9)
box(76, 28, 30, 12, 'Host GCN\n3× GCNConv 128→128→128\n+ mean pool → v_h (128)', C_HOST, C_HOST, fs=9)
arrow(68, 56, 76, 66, '#5d6d7e')
arrow(68, 48, 76, 34, '#5d6d7e')

# --- Bilinear attention ---
box(114, 42, 30, 18, 'Bilinear\nAttention\n(2 heads, k=3)\n→ fused f (256)', C_ATT, C_ATT, fs=10)
arrow(106, 66, 114, 54, C_ATT)
arrow(106, 34, 114, 46, C_ATT)

# --- Decoder (classification) ---
box(114, 68, 30, 12, 'MLP decoder\n256→512→512→128→1\n→ z_bin → P(Weak)', C_DEC, C_DEC, fs=9)
arrow(129, 60, 129, 68, C_DEC)

# --- Multi-task regression head ---
box(114, 12, 30, 12, 'Regression head\n(same MLP dims)\n→ log₁₀K  [multi-task]', C_REG, C_REG, fs=9)
arrow(129, 42, 129, 24, C_REG, ls='--')

# --- Legend / notes ---
ax.text(75, 6, 'Dual-branch graph encoder (no 3D coordinates) → explicit guest–host interaction tensor (bilinear attention) → '
               'classification logit; an optional parallel regression head yields a formula-free log₁₀K estimate.',
        ha='center', va='center', fontsize=9.5, style='italic', color='#333333')

fig.tight_layout()
fig.savefig(os.path.join(OUT, 'Fig_6-1_architecture_encoding.png'), dpi=600)
fig.savefig(os.path.join(OUT, 'Fig_6-1_architecture_encoding.svg'))
print('Saved Fig_6-1_architecture_encoding.png/.svg (600 dpi)')
