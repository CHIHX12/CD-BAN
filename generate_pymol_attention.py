"""
generate_pymol_attention.py
============================
Generates for α/β/γ/HP-β-CD + Naproxen:
  1. 3D PDB structures (RDKit ETKDGv3)
  2. BANLayer attention scores per CD atom
  3. PyMOL scripts (load PDB + attention coloring + guest placement)
  4. Comparison heatmap (4 CD types side by side)

Usage:
  python generate_pymol_attention.py

Output directory: results/pymol/
"""

import os, sys, warnings, torch, yaml, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

from models import CDBAN
from dataloader import smiles_to_pyg
from torch_geometric.data import Batch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.rdchem import Conformer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors

OUT_DIR = 'results/pymol'
os.makedirs(OUT_DIR, exist_ok=True)

CDS = {
    'alpha-CD': 'OC[C@H]1O[C@@H]2O[C@@H]3[C@@H](CO)O[C@H](O[C@@H]4[C@@H](CO)O[C@H](O[C@@H]5[C@@H](CO)O[C@H](O[C@@H]6[C@@H](CO)O[C@H](O[C@H]1[C@H](O)[C@H]2O)[C@H](O)[C@H]6O)[C@H](O)[C@H]5O)[C@H](O)[C@H]4O)[C@H](O)[C@H]3O',
    'beta-CD':  'OC[C@H]1O[C@@H]2O[C@@H]3[C@@H](CO)O[C@H](O[C@@H]4[C@@H](CO)O[C@H](O[C@@H]5[C@@H](CO)O[C@H](O[C@@H]6[C@@H](CO)O[C@H](O[C@@H]7[C@@H](CO)O[C@H](O[C@@H]8[C@@H](CO)O[C@H](O[C@H]1[C@H](O)[C@H]2O)[C@H](O)[C@H]8O)[C@H](O)[C@H]7O)[C@H](O)[C@H]6O)[C@H](O)[C@H]5O)[C@H](O)[C@H]4O)[C@H](O)[C@H]3O',
    'gamma-CD': 'OC[C@H]1O[C@@H]2O[C@@H]3[C@@H](CO)O[C@H](O[C@@H]4[C@@H](CO)O[C@H](O[C@@H]5[C@@H](CO)O[C@H](O[C@@H]6[C@@H](CO)O[C@H](O[C@@H]7[C@@H](CO)O[C@H](O[C@@H]8[C@@H](CO)O[C@H](O[C@@H]9[C@@H](CO)O[C@H](O[C@H]1[C@H](O)[C@H]2O)[C@H](O)[C@H]9O)[C@H](O)[C@H]8O)[C@H](O)[C@H]7O)[C@H](O)[C@H]6O)[C@H](O)[C@H]5O)[C@H](O)[C@H]4O)[C@H](O)[C@H]3O',
    'HP-beta-CD': 'CC(O)COC[C@H]1O[C@@H]2O[C@H]3[C@@H](O)[C@H](O)[C@@H](O[C@H]4[C@@H](O)[C@H](O)[C@@H](O[C@H]5[C@@H](O)[C@H](O)[C@@H](O[C@H]6[C@@H](O)[C@H](O)[C@@H](O[C@H]7[C@@H](O)[C@H](O)[C@@H](O[C@H]8[C@@H](O)[C@H](O)[C@@H](O[C@H]1[C@@H](O)[C@@H]2O)O[C@@H]8COCC(C)O)O[C@@H]7COCC(C)O)O[C@@H]6COCC(C)O)O[C@@H]5COCC(C)O)O[C@@H]4COCC(C)O)O[C@@H]3COCC(C)O',
}

GUEST_SMILES = 'CC(c1ccc2cccc(OC)c2c1)C(=O)O'
GUEST_NAME   = 'Naproxen'

print('Loading model (seed 49)...')
with open('configs/CDBAN.yaml') as f:
    cfg = yaml.safe_load(f)
model = CDBAN(**cfg)
pth = sorted([p for p in os.listdir('results/seed_49') if p.startswith('best_model')])[-1]
model.load_state_dict(torch.load(f'results/seed_49/{pth}', map_location='cpu', weights_only=False))
model.eval()


def smiles_to_pdb(smiles: str, name: str, out_path: str, optimize=True) -> bool:
    """Convert SMILES to 3D conformer and write PDB file."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f'  Invalid SMILES: {name}')
        return False
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    r = AllChem.EmbedMolecule(mol, params)
    if r == -1:
        r = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    if r == -1:
        print(f'  3D generation failed: {name}')
        return False
    if optimize:
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        except Exception:
            pass
    Chem.MolToPDBFile(mol, out_path)
    return True


def get_attention(guest_smi: str, host_smi: str):
    """Return (z_bin, att_avg[n_guest, n_host], n_guest, n_host)."""
    g = smiles_to_pyg(guest_smi)
    h = smiles_to_pyg(host_smi)
    bg = Batch.from_data_list([g])
    bh = Batch.from_data_list([h])
    n_g, n_h = g.x.shape[0], h.x.shape[0]
    with torch.no_grad():
        _, _, score, att = model(bg, bh, mode='eval')
    z = score.item()
    att_np   = att[0].numpy()                  # [2, 290, 290]
    att_crop = att_np[:, :n_g, :n_h]           # [2, n_g, n_h]
    att_avg  = att_crop.mean(axis=0)            # [n_g, n_h]
    mn, mx   = att_avg.min(), att_avg.max()
    if mx > mn:
        att_avg = (att_avg - mn) / (mx - mn)
    return z, att_avg, n_g, n_h


def write_pymol(cd_name: str, guest_name: str, host_attn: np.ndarray,
                cd_pdb: str, guest_pdb: str, out_pml: str):
    """Generate PyMOL script: load PDB, map attention to B-factor, color."""
    lines = [
        f'# CD-BAN Attention Visualization',
        f'# {guest_name} inside {cd_name}',
        f'# Color = normalised BANLayer attention (blue=low, red=high)',
        '',
        '# Load structures',
        f'load {os.path.abspath(cd_pdb)}, cd_host',
        f'load {os.path.abspath(guest_pdb)}, guest',
        '',
        '# Reset B-factor to 0',
        'alter cd_host, b=0.0',
        '',
        '# Attention score -> B-factor (per atom)',
    ]
    for idx, score in enumerate(host_attn):
        lines.append(f'alter (cd_host and index {idx+1}), b={score:.6f}')
    lines += [
        '',
        '# Color by B-factor (blue=low attention, red=high attention)',
        'spectrum b, blue_white_red, cd_host, minimum=0, maximum=1',
        '',
        '# Display style',
        'hide everything',
        'show surface, cd_host',
        'show sticks, cd_host',
        'set transparency, 0.35, cd_host',
        '',
        '# Guest (drug) display',
        'show sticks, guest',
        'color yellow, guest',
        'set stick_radius, 0.15, guest',
        '',
        '# Camera',
        'orient',
        'zoom cd_host, 5',
        'set bg_color, white',
        '',
        f'# blue  = low attention (CD atoms less attended by model)',
        f'# red   = high attention (CD atoms model associates with binding)',
        f'# yellow = {guest_name} (drug guest)',
        '',
        '# Optional: save image',
        f'# png {out_pml.replace(".pml", ".png")}, width=1200, height=900, dpi=300',
    ]
    with open(out_pml, 'w') as f:
        f.write('\n'.join(lines))


def plot_comparison(results: dict, out_path: str):
    """Side-by-side attention heatmaps for 4 CD types."""
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f'CD-BAN Attention Heatmaps — {GUEST_NAME} vs 4 Cyclodextrin Types\n'
                 f'(BANLayer, seed 49, 2 heads averaged)',
                 fontsize=13, fontweight='bold')

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.3)

    for i, (cd_name, data) in enumerate(results.items()):
        ax = fig.add_subplot(gs[i//2, i%2])
        att  = data['att']
        z    = data['z']
        n_g, n_h = att.shape
        im = ax.imshow(att, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        ax.set_title(f'{cd_name}  (z={z:+.3f}, P(Weak)={1/(1+np.exp(-z)):.3f})',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel(f'Host ({n_h} heavy atoms)', fontsize=8)
        ax.set_ylabel(f'Guest ({n_g} atoms)', fontsize=8)

        g_mol = Chem.MolFromSmiles(GUEST_SMILES)
        g_labels = [f'{a.GetSymbol()}{a.GetIdx()}' for a in g_mol.GetAtoms()]
        ax.set_yticks(range(n_g))
        ax.set_yticklabels(g_labels[:n_g], fontsize=6)
        ax.set_xticks([])

        host_max = att.max(axis=0)
        top5_idx = np.argsort(host_max)[-5:]
        for idx in top5_idx:
            ax.axvline(x=idx, color='cyan', linewidth=0.8, alpha=0.6)

        plt.colorbar(im, ax=ax, shrink=0.7, label='Attention')

    plt.savefig(out_path + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(out_path + '.svg', bbox_inches='tight')
    print(f'-> Comparison heatmap: {out_path}.png')
    plt.close()


print(f'\nGenerating {GUEST_NAME} 3D PDB...')
guest_pdb = os.path.join(OUT_DIR, f'{GUEST_NAME.lower()}.pdb')
ok = smiles_to_pdb(GUEST_SMILES, GUEST_NAME, guest_pdb)
print(f'  {"ok" if ok else "FAILED"} {guest_pdb}')

results = {}
print(f'\nComputing attention for 4 CD types + generating 3D PDBs...')
for cd_name, cd_smi in CDS.items():
    print(f'\n  [{cd_name}]')
    safe_name = cd_name.replace('-', '_').replace(' ', '_')

    cd_pdb = os.path.join(OUT_DIR, f'{safe_name}.pdb')
    ok_pdb = smiles_to_pdb(cd_smi, cd_name, cd_pdb)
    print(f'    PDB: {"ok" if ok_pdb else "FAILED"} {cd_pdb}')

    z, att, n_g, n_h = get_attention(GUEST_SMILES, cd_smi)
    host_attn = att.max(axis=0)   # [n_h] max attention per CD atom across all guest atoms
    print(f'    z={z:+.4f}  P(Weak)={1/(1+np.exp(-z)):.4f}  '
          f'Guest {n_g} atoms x Host {n_h} atoms')

    csv_path = os.path.join(OUT_DIR, f'{safe_name}_attention.csv')
    pd.DataFrame({
        'atom_index': range(1, n_h+1),
        'attention_max': host_attn,
        'attention_mean': att.mean(axis=0),
    }).to_csv(csv_path, index=False)

    pml_path = os.path.join(OUT_DIR, f'{safe_name}_{GUEST_NAME.lower()}.pml')
    write_pymol(cd_name, GUEST_NAME, host_attn, cd_pdb, guest_pdb, pml_path)
    print(f'    PyMOL: {pml_path}')

    results[cd_name] = {'att': att, 'z': z}

print('\nGenerating comparison heatmap...')
plot_comparison(results, os.path.join(OUT_DIR, 'comparison_heatmap'))

readme = f"""CD-BAN PyMOL Visualization Files
==================================

Guest molecule: {GUEST_NAME}  ({GUEST_SMILES})
Four cyclodextrin types: alpha-CD, beta-CD, gamma-CD, HP-beta-CD

File listing
------------
naproxen.pdb           - drug 3D structure (RDKit ETKDGv3)
alpha_CD.pdb           - alpha-CD 3D structure
beta_CD.pdb            - beta-CD 3D structure
gamma_CD.pdb           - gamma-CD 3D structure
HP_beta_CD.pdb         - HP-beta-CD 3D structure

*_attention.csv        - per-atom attention scores for each CD

alpha_CD_naproxen.pml  - PyMOL script (alpha-CD + Naproxen)
beta_CD_naproxen.pml   - PyMOL script (beta-CD + Naproxen)
gamma_CD_naproxen.pml  - PyMOL script (gamma-CD + Naproxen)
HP_beta_CD_naproxen.pml- PyMOL script (HP-beta-CD + Naproxen)

comparison_heatmap.png/svg - side-by-side attention heatmaps for 4 CD types

PyMOL usage
-----------
1. Install PyMOL:
   conda install -c conda-forge pymol-open-source

2. Open PyMOL and run:
   @<full_path>/beta_CD_naproxen.pml

3. Color legend:
   blue   = low attention (CD atoms less attended by the model)
   white  = intermediate attention
   red    = high attention (CD atoms associated with binding by the model)
   yellow = Naproxen (drug guest)

Note:
   PDB structures are generated from SMILES by RDKit (ETKDGv3 force-field
   optimization). The drug position is not a real docking pose; it is
   illustrative only. For precise docking positions, use AutoDock Vina or
   similar tools.
"""

with open(os.path.join(OUT_DIR, 'README_pymol.txt'), 'w') as f:
    f.write(readme)

print(f'\nAll files saved to: {OUT_DIR}/')
print('Done.')
