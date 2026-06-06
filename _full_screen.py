"""
_full_screen.py — Full ternary screening
=========================================
Drug × CD × Coformer three-way combination in a single run.

Steps:
  Step 1: model inference -> z_bin  (for each drug-CD pair)
  Step 2: ternary formula -> K_ternary, K_ratio, ΔΔG  (for each z_bin × coformer)

Outputs:
  results/tables/full_ternary_screen.csv   complete raw data
  results/tables/full_ternary_screen.txt   human-readable summary
"""
import sys, os, warnings, math
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

import torch, yaml, pandas as pd, numpy as np
from torch_geometric.data import Batch
from models import CDBAN
from dataloader import smiles_to_pyg
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

W_LOGP, CONST_COEF = -0.0378, -0.7361
DZ_CONST            = -0.690
SLOPE, INTCPT       = -1.128, 3.714
RT_298              = 2.479
C2, C1              = 1.458, -0.799

DRUGS = {
    'Naproxen':      'CC(c1ccc2cccc(OC)c2c1)C(=O)O',
    'Indomethacin':  'CC1=C(CC(=O)O)c2cc(OC)ccc2N1C(=O)c1ccc(Cl)cc1',
    'Ibuprofen':     'CC(C)Cc1ccc(cc1)C(C)C(=O)O',
    'Diclofenac':    'OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl',
    'Ketoprofen':    'OC(=O)C(C)c1cccc(c1)C(=O)c1ccccc1',
    'Aspirin':       'CC(=O)Oc1ccccc1C(=O)O',
    'Paracetamol':   'CC(=O)Nc1ccc(O)cc1',
}

def get_cd_smiles():
    dfs = [pd.read_csv(f) for f in
           ['data/binary/train.csv','data/binary/val.csv',
            'data/binary/test.csv','data/binary/fuzzy.csv']]
    df = pd.concat(dfs)
    hosts = {
        'α-CD':      'alpha-cyclodextrin',
        'β-CD':      'beta-cyclodextrin',
        'γ-CD':      'gamma-cyclodextrin',
        'HP-β-CD':   'hp-beta-cd',
    }
    return {k: df[df['Host']==v]['SMILES_Host'].iloc[0] for k, v in hosts.items()}

COFORMERS_SM = [
    ('Nicotinamide',     'NC(=O)c1ccncc1',                       'hydrotrope'),
    ('Succinic acid',    'OC(=O)CCC(=O)O',                       'organic acid'),
    ('L-Proline',        'OC(=O)[C@@H]1CCCN1',                   'amino acid'),
    ('L-Lysine',         'N[C@@H](CCCCN)C(=O)O',                 'amino acid'),
    ('L-Histidine',      'N[C@@H](Cc1cnc[nH]1)C(=O)O',           'amino acid'),
    ('Glycine',          'NCC(=O)O',                              'amino acid'),
    ('Urea',             'NC(N)=O',                               'hydrotrope'),
    ('Caffeine',         'Cn1cnc2c1c(=O)n(c(=O)n2C)C',           'hydrotrope'),
    ('L-Arginine',       'N[C@@H](CCCNC(=N)N)C(=O)O',            'amino acid'),
    ('L-Glutamine',      'N[C@@H](CCC(N)=O)C(=O)O',              'amino acid'),
    ('Triethanolamine',  'OCCN(CCO)CCO',                          'amine'),
    ('Citric acid',      'OC(CC(O)(CC(=O)O)C(=O)O)C(=O)O',       'organic acid'),
    ('Tartaric acid',    'OC(C(O)C(=O)O)C(=O)O',                 'organic acid'),
    ('Tromethamine',     'OCC(N)(CO)CO',                          'buffer amine'),
    ('Meglumine',        'OCC(O)C(O)C(O)C(O)CN',                 'sugar amine'),
]
COFORMERS_POLY = [
    ('HPMC E5',       'polymer'),
    ('HPMC K15M',     'polymer'),
    ('PVP K30',       'polymer'),
    ('PEG 4000',      'polymer'),
    ('Poloxamer 188', 'polymer'),
]

def coformer_descriptors():
    desc = {}
    for name, smi, cat in COFORMERS_SM:
        mol = Chem.MolFromSmiles(smi)
        desc[name] = {
            'logP':   Descriptors.MolLogP(mol),
            'TPSA':   rdMolDescriptors.CalcTPSA(mol),
            'cat':    cat,
            'poly':   False,
        }
    for name, cat in COFORMERS_POLY:
        desc[name] = {'logP': None, 'TPSA': None, 'cat': cat, 'poly': True}
    return desc

def ternary(z_bin, logP, is_poly):
    dz = DZ_CONST if is_poly else W_LOGP * logP + CONST_COEF
    z_t = z_bin + dz
    logKb = max(-1, min((INTCPT - z_bin) / (-SLOPE), 8))
    logKt = max(-1, min((INTCPT - z_t)  / (-SLOPE), 8))
    Kb = 10 ** logKb
    Kt = 10 ** logKt
    ratio = 10 ** (-dz / abs(SLOPE))
    ddG = -RT_298 * math.log(ratio)
    valid = C1 <= z_t <= C2
    return dict(dz=round(dz,4), z_tern=round(z_t,4),
                K_binary=round(Kb,1), K_ternary=round(Kt,1),
                K_ratio=round(ratio,2), ddG=round(ddG,2),
                calib_valid=valid)

def main():
    os.makedirs('results/tables', exist_ok=True)

    print('Loading model (seed 49)...')
    with open('configs/CDBAN.yaml') as f:
        cfg = yaml.safe_load(f)
    model = CDBAN(**cfg)
    sd = sorted([p for p in os.listdir('results/seed_49') if p.startswith('best_model')])[-1]
    model.load_state_dict(torch.load(f'results/seed_49/{sd}', map_location='cpu', weights_only=False))
    model.eval()

    cd_smiles = get_cd_smiles()
    desc = coformer_descriptors()

    # Step 1: compute z_bin for all drug-CD pairs
    print(f'\n[Step 1] {len(DRUGS)} drugs × {len(cd_smiles)} CDs = {len(DRUGS)*len(cd_smiles)} pairs...')
    zbin_table = []
    for drug, dsmi in DRUGS.items():
        for cd, csmi in cd_smiles.items():
            g = smiles_to_pyg(dsmi)
            h = smiles_to_pyg(csmi)
            bg = Batch.from_data_list([g])
            bh = Batch.from_data_list([h])
            with torch.no_grad():
                _, _, score, _ = model(bg, bh, mode='eval')
            z = score.item()
            p = torch.sigmoid(score).item()
            label = ('Weak (High conf)' if z > C2 else
                     'Strong (High conf)' if z < C1 else
                     'Fuzzy-Weak' if z > 0 else 'Fuzzy-Strong')
            zbin_table.append({'Drug': drug, 'CD': cd, 'z_bin': round(z,4),
                                'P_weak': round(p,4), 'DL_label': label})
            print(f'  {drug:15} + {cd:10} -> z={z:+.4f}  [{label}]')

    df_z = pd.DataFrame(zbin_table)

    # Step 2: ternary screening
    n_coformers = len(COFORMERS_SM) + len(COFORMERS_POLY)
    print(f'\n[Step 2] {len(df_z)} pairs × {n_coformers} coformers = {len(df_z)*n_coformers} ternary predictions...')
    rows = []
    for _, zrow in df_z.iterrows():
        for cname, cinfo in desc.items():
            t = ternary(zrow['z_bin'], cinfo['logP'], cinfo['poly'])
            usable = 'Fuzzy' in zrow['DL_label']
            rows.append({
                'Drug':        zrow['Drug'],
                'CD':          zrow['CD'],
                'DL_label':    zrow['DL_label'],
                'z_bin':       zrow['z_bin'],
                'Coformer':    cname,
                'Category':    cinfo['cat'],
                'logP':        round(cinfo['logP'], 3) if cinfo['logP'] else 'polymer',
                'K_binary':    t['K_binary'],
                'K_ternary':   t['K_ternary'],
                'K_ratio':     t['K_ratio'],
                'ddG_kJmol':   t['ddG'],
                'calib_valid': t['calib_valid'],
                'model':       'constant' if cinfo['poly'] else 'descriptor',
                'usable':      usable,
                'usable_note': ('OK — Fuzzy Zone, ternary estimate valid' if usable else
                                'SKIP — Strong (High conf): already K>>10,000, extrapolation only' if 'Strong (High conf)' in zrow['DL_label'] else
                                'SKIP — Weak (High conf): K<100, drug-CD mismatch, coformer ineffective'),
            })

    df = pd.DataFrame(rows)
    out_csv = 'results/tables/full_ternary_screen.csv'
    df.to_csv(out_csv, index=False)
    print(f'\n-> Saved: {out_csv}  ({len(df)} rows)')

    out_txt = 'results/tables/full_ternary_screen.txt'
    lines = []
    lines.append('='*78)
    lines.append('  CD-BAN Full Ternary Screening Report')
    lines.append(f'  {len(DRUGS)} Drugs × {len(cd_smiles)} CDs × {n_coformers} Coformers = {len(df)} predictions')
    lines.append('='*78)

    for drug in DRUGS:
        lines.append(f'\n{"─"*78}')
        lines.append(f'  DRUG: {drug}')
        lines.append(f'{"─"*78}')
        sub = df[df['Drug'] == drug]
        for cd in cd_smiles:
            z_row = df_z[(df_z['Drug']==drug) & (df_z['CD']==cd)].iloc[0]
            lines.append(f'\n  + {cd}  (z_bin={z_row["z_bin"]:+.4f}, {z_row["DL_label"]},'
                         f'  K_binary≈{10**max(-1,min((INTCPT-z_row["z_bin"])/(-SLOPE),8)):.0f} M⁻¹)')
            sub_cd = sub[sub['CD']==cd].sort_values('K_ratio', ascending=False)
            lines.append(f'  {"Rank":<5} {"Coformer":<20} {"Cat":<15} {"K_ratio":>8} {"K_tern":>10} {"ΔΔG":>10}')
            lines.append(f'  {"----":<5} {"--------":<20} {"---":<15} {"-------":>8} {"------":>10} {"---":>10}')
            for rank, (_, r) in enumerate(sub_cd.iterrows(), 1):
                lines.append(f'  {rank:<5} {r["Coformer"]:<20} {r["Category"]:<15}'
                              f' {r["K_ratio"]:>8.2f}× {r["K_ternary"]:>9.0f} M⁻¹'
                              f' {r["ddG_kJmol"]:>8.2f} kJ/mol')

    lines.append('\n' + '='*78)
    lines.append('  ⚠️  Ternary formula based on n=4 data points.')
    lines.append('  K_ratio and ΔΔG trends are directionally reliable;')
    lines.append('  absolute K values require experimental validation.')
    lines.append('='*78)

    with open(out_txt, 'w') as f:
        f.write('\n'.join(lines))
    print(f'-> Saved: {out_txt}')
    print('\nDone.')

if __name__ == '__main__':
    main()
