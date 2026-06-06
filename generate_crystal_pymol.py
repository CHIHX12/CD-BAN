"""
generate_crystal_pymol.py
==========================
Visualises BANLayer attention on CRYSTAL STRUCTURE PDB files for
alpha / beta / gamma-CD, and on a RDKit-generated PDB for HP-beta-CD.

Crystal sources (downloaded automatically from RCSB if missing):
  alpha-CD : 4FEM  chain B  (SusE complex, 6 GLC residues = 66 atoms)
  beta-CD  : 1DMB  chain B  (maltodextrin binding protein, 7 GLC = 77 atoms)
  gamma-CD : 2ZYK  chain E  (maltodextrin-binding protein, 8 GLC = 88 atoms)
  HP-beta-CD: RDKit ETKDGv3 (no clean crystal structure available)

Atom-role mapping (PDB name -> glucopyranose role):
  C1->C1  C2->C2  C3->C3  C4->C4  C5->C5  C6->C6
  O2->O2  O3->O3  O4->Ob  O5->O5  O6->O6

Attention: averaged over all label=0 samples, then averaged per role.
Color coded by B-factor in PyMOL (blue=low, white=mid, red=high attention).

Usage:
  python generate_crystal_pymol.py
  # Then from results/crystal_pymol/:
  #   pymol beta_CD_crystal.pml
"""

import os, sys, warnings, urllib.request, torch, yaml
import numpy as np
import pandas as pd
from collections import Counter
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

from rdkit import Chem
from rdkit.Chem import AllChem
from models import CDBAN
from dataloader import smiles_to_pyg
from torch_geometric.data import Batch

OUT_DIR  = 'results/crystal_pymol'
DATA_DIR = 'data/binary'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Crystal structure definitions ────────────────────────────────────────────
CRYSTAL = {
    'alpha-CD': {'pdb_id': '4FEM', 'chain': 'B', 'n_units': 6},
    'beta-CD':  {'pdb_id': '1DMB', 'chain': 'B', 'n_units': 7},
    'gamma-CD': {'pdb_id': '2ZYK', 'chain': 'E', 'n_units': 8},
}

# PDB atom name -> chemical role (applies to GLC HETATM records)
PDB_NAME_TO_ROLE = {
    'C1': 'C1', 'C2': 'C2', 'C3': 'C3', 'C4': 'C4', 'C5': 'C5', 'C6': 'C6',
    'O2': 'O2', 'O3': 'O3', 'O4': 'Ob', 'O5': 'O5', 'O6': 'O6',
}

# CD SMILES (from dataset; needed for HP-beta-CD PDB generation + model inference)
CD_SMILES = {
    'alpha-CD':   'OC[C@H]1O[C@@H]2O[C@@H]3[C@@H](CO)O[C@H](O[C@@H]4[C@@H](CO)O[C@H](O[C@@H]5[C@@H](CO)O[C@H](O[C@@H]6[C@@H](CO)O[C@H](O[C@@H]7[C@@H](CO)O[C@H](O[C@H]1[C@H](O)[C@H]2O)[C@H](O)[C@H]7O)[C@H](O)[C@H]6O)[C@H](O)[C@H]5O)[C@H](O)[C@H]4O)[C@H](O)[C@H]3O',
    'beta-CD':    'OC[C@H]1O[C@@H]2O[C@@H]3[C@@H](CO)O[C@H](O[C@@H]4[C@@H](CO)O[C@H](O[C@@H]5[C@@H](CO)O[C@H](O[C@@H]6[C@@H](CO)O[C@H](O[C@@H]7[C@@H](CO)O[C@H](O[C@@H]8[C@@H](CO)O[C@H](O[C@H]1[C@H](O)[C@H]2O)[C@H](O)[C@H]8O)[C@H](O)[C@H]7O)[C@H](O)[C@H]6O)[C@H](O)[C@H]5O)[C@H](O)[C@H]4O)[C@H](O)[C@H]3O',
    'gamma-CD':   'OC[C@H]1O[C@@H]2O[C@@H]3[C@@H](CO)O[C@H](O[C@@H]4[C@@H](CO)O[C@H](O[C@@H]5[C@@H](CO)O[C@H](O[C@@H]6[C@@H](CO)O[C@H](O[C@@H]7[C@@H](CO)O[C@H](O[C@@H]8[C@@H](CO)O[C@H](O[C@@H]9[C@@H](CO)O[C@H](O[C@H]1[C@H](O)[C@H]2O)[C@H](O)[C@H]9O)[C@H](O)[C@H]8O)[C@H](O)[C@H]7O)[C@H](O)[C@H]6O)[C@H](O)[C@H]5O)[C@H](O)[C@H]4O)[C@H](O)[C@H]3O',
    'HP-beta-CD': 'CC(O)COC[C@H]1O[C@@H]2O[C@H]3[C@@H](O)[C@H](O)[C@@H](O[C@H]4[C@@H](O)[C@H](O)[C@@H](O[C@H]5[C@@H](O)[C@H](O)[C@@H](O[C@H]6[C@@H](O)[C@H](O)[C@@H](O[C@H]7[C@@H](O)[C@H](O)[C@@H](O[C@H]8[C@@H](O)[C@H](O)[C@@H](O[C@H]1[C@@H](O)[C@@H]2O)O[C@@H]8COCC(C)O)O[C@@H]7COCC(C)O)O[C@@H]6COCC(C)O)O[C@@H]5COCC(C)O)O[C@@H]4COCC(C)O)O[C@@H]3COCC(C)O',
}

CD_ORDER = ['alpha-CD', 'beta-CD', 'gamma-CD', 'HP-beta-CD']


# ── Load model ───────────────────────────────────────────────────────────────
print('Loading model (seed 49)...')
with open('configs/CDBAN.yaml') as f:
    cfg = yaml.safe_load(f)
model = CDBAN(**cfg)
pth = sorted([p for p in os.listdir('results/seed_49') if p.startswith('best_model')])[-1]
model.load_state_dict(
    torch.load(f'results/seed_49/{pth}', map_location='cpu', weights_only=False))
model.eval()
print(f'  Model: {pth}')


# ── Helpers ──────────────────────────────────────────────────────────────────
def get_attention_vec(guest_smi, host_smi):
    """Run forward pass, return normalised [n_host] attention (avg over heads+guest)."""
    g = smiles_to_pyg(guest_smi)
    h = smiles_to_pyg(host_smi)
    n_g, n_h = g.x.shape[0], h.x.shape[0]
    with torch.no_grad():
        _, _, _, att = model(Batch.from_data_list([g]),
                             Batch.from_data_list([h]), mode='eval')
    arr = att[0].numpy()[:, :n_g, :n_h].mean(axis=(0, 1))
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn) if mx > mn else arr, n_h


def build_role_map(smi):
    """Return ({atom_idx: role_str}, mol) using the same logic as generate_all_cd_aggregate.py."""
    mol = Chem.MolFromSmiles(smi)
    ring_info = mol.GetRingInfo()
    pyranose = [set(r) for r in ring_info.AtomRings() if len(r) == 6]
    role = {}

    for i, atom in enumerate(mol.GetAtoms()):
        sym = atom.GetSymbol()
        in_ring = atom.IsInRing()
        in_pyr = any(i in r for r in pyranose)
        nbs = list(atom.GetNeighbors())
        nb_in_pyr = [any(n.GetIdx() in r for r in pyranose) for n in nbs]

        if not in_ring:
            if sym == 'O' and atom.GetDegree() == 1:
                nb = nbs[0]
                if nb.GetDegree() == 2 and not nb.IsInRing():
                    role[i] = 'O6'
                elif any(nb_in_pyr):
                    role[i] = 'O2_or_O3'
                else:
                    role[i] = 'HP'
            elif sym == 'C' and atom.GetDegree() == 2 and not in_ring:
                has_terminal_o = any(n.GetSymbol() == 'O' and n.GetDegree() == 1 for n in nbs)
                has_ring_c = any(n.IsInRing() for n in nbs)
                if has_terminal_o and has_ring_c:
                    role[i] = 'C6'
                else:
                    role[i] = 'HP'
            else:
                role[i] = 'HP'
            continue

        if in_pyr:
            if sym == 'O':
                role[i] = 'O5'
                continue
            o_nbs = [n for n in nbs if n.GetSymbol() == 'O']
            c_nbs = [n for n in nbs if n.GetSymbol() == 'C']
            pyr_o = [n for n in o_nbs if any(n.GetIdx() in r for r in pyranose)]
            bridge_o = [n for n in o_nbs if n.IsInRing() and not any(n.GetIdx() in r for r in pyranose)]
            exo_c = [n for n in c_nbs if not n.IsInRing()]
            if pyr_o and bridge_o:
                role[i] = 'C1'
            elif exo_c:
                role[i] = 'C5'
            elif bridge_o and not pyr_o:
                role[i] = 'C4'
            else:
                role[i] = 'C2_C3'
            continue

        if in_ring and not in_pyr and sym == 'O':
            role[i] = 'Ob'
            continue

        role[i] = 'HP'

    # Resolve C2/C3 and O2/O3 by ring walk: C4-C3-C2-C1
    for pyr in pyranose:
        pyr = list(pyr)
        c1 = next((i for i in pyr if role.get(i) == 'C1'), None)
        c4 = next((i for i in pyr if role.get(i) == 'C4'), None)
        if c1 is None or c4 is None:
            continue
        atom_c4 = mol.GetAtomWithIdx(c4)
        ring_nbs_c4 = [n.GetIdx() for n in atom_c4.GetNeighbors()
                       if n.GetIdx() in pyr and role.get(n.GetIdx()) not in ('O5', 'C5')]
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
        for cidx, oname in [(c3, 'O3'), (c2, 'O2')]:
            for nb in mol.GetAtomWithIdx(cidx).GetNeighbors():
                if nb.GetSymbol() == 'O' and not nb.IsInRing() and nb.GetDegree() == 1:
                    role[nb.GetIdx()] = oname

    for i in list(role.keys()):
        if role[i] in ('C2_C3', 'O2_or_O3'):
            role[i] = 'HP'

    return role, mol


def classify_host(name):
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


def download_pdb(pdb_id, dest):
    """Download PDB from RCSB if not present."""
    if not os.path.exists(dest):
        url = f'https://files.rcsb.org/download/{pdb_id}.pdb'
        print(f'  Downloading {pdb_id} from RCSB...')
        urllib.request.urlretrieve(url, dest)
    return dest


def extract_cd_pdb(src_pdb, chain, out_pdb):
    """Extract only GLC residues of a given chain and write a clean PDB."""
    lines = []
    with open(src_pdb) as f:
        for line in f:
            if line.startswith(('HETATM', 'ATOM')):
                res = line[17:20].strip()
                ch  = line[21]
                if res == 'GLC' and ch == chain:
                    lines.append(line)
    lines.append('END\n')
    with open(out_pdb, 'w') as f:
        f.writelines(lines)
    return out_pdb


def smiles_to_pdb_rdkit(smi, out_pdb):
    """Generate 3D conformer from SMILES via RDKit MMFF and write PDB."""
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol, params) == -1:
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass
    mol = Chem.RemoveHs(mol)
    Chem.MolToPDBFile(mol, out_pdb)


# ── Load data and run inference ──────────────────────────────────────────────
print('\nLoading label=0 samples...')
frames = [pd.read_csv(f'{DATA_DIR}/{s}.csv').assign(split=s)
          for s in ('train', 'val', 'test')]
all_df = pd.concat(frames, ignore_index=True)
label0 = all_df[all_df['label'] == 0].copy()
label0['cd_type'] = label0['Host'].apply(classify_host)
print(f'  Total label=0: {len(label0)}')

print('\nRunning model inference...')
role_means = {}   # {cd_type: {role: mean_attention}}

for cd_type in CD_ORDER:
    smi = CD_SMILES[cd_type]
    subset = label0[label0['cd_type'] == cd_type].reset_index(drop=True)
    n_tot = len(subset)
    print(f'\n  [{cd_type}]  {n_tot} label-0 samples')
    if n_tot == 0:
        continue

    role_map, _ = build_role_map(smi)

    rows_att = []
    for i, row in subset.iterrows():
        try:
            att, n_h = get_attention_vec(row['SMILES_Guest'], row['SMILES_Host'])
            rows_att.append((att, n_h))
        except Exception as e:
            pass

    if not rows_att:
        continue

    modal_nh = Counter(r[1] for r in rows_att).most_common(1)[0][0]
    kept = [r[0] for r in rows_att if r[1] == modal_nh]
    mat = np.stack(kept)
    mean_att = mat.mean(axis=0)

    by_role = {}
    for idx, r in role_map.items():
        if idx < len(mean_att):
            by_role.setdefault(r, []).append(mean_att[idx])
    role_means[cd_type] = {r: float(np.mean(v)) for r, v in by_role.items()}
    print('  Role means:', {r: round(v, 3) for r, v in
                            sorted(role_means[cd_type].items(), key=lambda x: -x[1])})


# ── Generate crystal PDB + PML for alpha / beta / gamma ──────────────────────
print('\nGenerating crystal structure PML files...')

for cd_type, info in CRYSTAL.items():
    pdb_id = info['pdb_id']
    chain  = info['chain']
    rmeans = role_means.get(cd_type)
    if rmeans is None:
        print(f'  [{cd_type}] No attention data, skipping')
        continue

    safe = cd_type.replace('-', '_').replace(' ', '_')

    # Download full crystal PDB if needed
    full_pdb = f'{OUT_DIR}/{pdb_id}.pdb'
    download_pdb(pdb_id, full_pdb)

    # Extract CD-only PDB
    cd_pdb = f'{OUT_DIR}/{safe}_crystal.pdb'
    extract_cd_pdb(full_pdb, chain, cd_pdb)
    print(f'  [{cd_type}]  Crystal PDB: {cd_pdb}')

    # Count atoms in extracted PDB
    n_atoms = sum(1 for ln in open(cd_pdb)
                  if ln.startswith(('HETATM', 'ATOM')))
    print(f'    {n_atoms} atoms  (expected {info["n_units"] * 11})')

    # Write PML
    pml_path = f'{OUT_DIR}/{safe}_crystal.pml'
    png_path = f'{safe}_crystal_render.png'

    lines = [
        f'# CD-BAN attention on crystal structure: {cd_type}',
        f'# Source: PDB {pdb_id} chain {chain}',
        f'# Color: blue=low attention, white=mid, red=high attention',
        '',
        f'load {os.path.basename(cd_pdb)}, cd_host',
        '',
        '# Set all B-factors to 0 first',
        'alter cd_host, b=0.0',
        '',
        '# Map attention score to B-factor by atom name (role)',
    ]

    for pdb_name, role in PDB_NAME_TO_ROLE.items():
        val = rmeans.get(role, 0.0)
        lines.append(
            f'alter (cd_host and name {pdb_name}), b={val:.6f}')

    lines += [
        '',
        '# Color spectrum: blue(low) -> white(mid) -> red(high)',
        'spectrum b, blue_white_red, cd_host, minimum=0, maximum=1',
        '',
        '# Display settings',
        'hide everything',
        'show sticks, cd_host',
        'show surface, cd_host',
        'set transparency, 0.30, cd_host',
        'set stick_radius, 0.15',
        '',
        '# View',
        'orient cd_host',
        'zoom cd_host, 3',
        'set bg_color, white',
        '',
        '# Render 1920x1080',
        f'ray 1920, 1080',
        f'png {png_path}, dpi=300',
        '',
        '# Legend:',
        '# blue  = low attention (C2, C3 backbone carbons)',
        '# white = intermediate (C1, C4, C5, C6)',
        '# red   = high attention (O5 ring-O, O2/O3 secondary-OH, Ob bridge-O)',
    ]

    with open(pml_path, 'w', encoding='ascii') as f:
        f.write('\n'.join(lines))
    print(f'    PML: {pml_path}')

    # Save role-attention CSV
    csv_path = f'{OUT_DIR}/{safe}_role_attention.csv'
    pd.DataFrame([{'role': r, 'mean_attention': round(v, 4)}
                  for r, v in sorted(rmeans.items(), key=lambda x: -x[1])]
                 ).to_csv(csv_path, index=False)


# ── HP-beta-CD: RDKit PDB + PML ──────────────────────────────────────────────
print('\n  [HP-beta-CD]  Using RDKit-generated PDB (no clean crystal available)')
hp_rmeans = role_means.get('HP-beta-CD')
if hp_rmeans:
    hp_pdb = f'{OUT_DIR}/HP_beta_CD_rdkit.pdb'
    smiles_to_pdb_rdkit(CD_SMILES['HP-beta-CD'], hp_pdb)
    print(f'    RDKit PDB: {hp_pdb}')

    # For HP-beta-CD we use RDKit-generated PDB with build_role_map
    # and write B-factor via index
    role_map_hp, _ = build_role_map(CD_SMILES['HP-beta-CD'])

    pml_lines = [
        '# CD-BAN attention on HP-beta-CD (RDKit ETKDGv3+MMFF structure)',
        '# Color: blue=low, white=mid, red=high attention',
        '',
        'load HP_beta_CD_rdkit.pdb, cd_host',
        '',
        'alter cd_host, b=0.0',
        '',
        '# Map attention by atom index',
    ]
    for idx, role in role_map_hp.items():
        val = hp_rmeans.get(role, 0.0)
        pml_lines.append(f'alter (cd_host and index {idx+1}), b={val:.6f}')

    pml_lines += [
        '',
        'spectrum b, blue_white_red, cd_host, minimum=0, maximum=1',
        'hide everything',
        'show sticks, cd_host',
        'show surface, cd_host',
        'set transparency, 0.30, cd_host',
        'set stick_radius, 0.15',
        'orient cd_host',
        'zoom cd_host, 3',
        'set bg_color, white',
        'ray 1920, 1080',
        'png HP_beta_CD_crystal_render.png, dpi=300',
        '',
        '# blue=low(C2,C3)  white=mid  red=high(O5,O2,O3,Ob)',
    ]
    hp_pml = f'{OUT_DIR}/HP_beta_CD_crystal.pml'
    with open(hp_pml, 'w', encoding='ascii') as f:
        f.write('\n'.join(pml_lines))
    print(f'    PML: {hp_pml}')

    csv_path = f'{OUT_DIR}/HP_beta_CD_role_attention.csv'
    pd.DataFrame([{'role': r, 'mean_attention': round(v, 4)}
                  for r, v in sorted(hp_rmeans.items(), key=lambda x: -x[1])]
                 ).to_csv(csv_path, index=False)

print(f'\nAll files in: {OUT_DIR}/')
print("""
Usage:
  cd results/crystal_pymol
  pymol beta_CD_crystal.pml
  pymol alpha_CD_crystal.pml
  pymol gamma_CD_crystal.pml
  pymol HP_beta_CD_crystal.pml
""")
print('Done.')
