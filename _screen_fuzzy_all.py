"""
_screen_fuzzy_all.py — Full fuzzy-zone ternary screening
=========================================================
Recovers z_bin from fuzzy_predictions.csv (1,850 compounds) via logit(p_weak),
runs the ternary formula against 20 coformers, and outputs a 37,000-row matrix.

No model re-inference required (pure arithmetic).

Outputs:
  results/tables/fuzzy_ternary_all.csv    complete matrix (37,000 rows)
  results/tables/fuzzy_ternary_top.csv    best coformer per drug-CD pair (1,850 rows)
  results/tables/fuzzy_ternary_top100.csv top 100 by K_ternary
"""
import math, os, warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

os.makedirs('results/tables', exist_ok=True)

W_LOGP, CONST_COEF = -0.0378, -0.7361
DZ_CONST            = -0.690
SLOPE, INTCPT       = -1.128, 3.714
RT_298              = 2.479
C2, C1              = 1.458, -0.799

COFORMERS_SM = [
    ('Nicotinamide',    'NC(=O)c1ccncc1',                 'hydrotrope'),
    ('Succinic acid',   'OC(=O)CCC(=O)O',                 'organic acid'),
    ('L-Proline',       'OC(=O)[C@@H]1CCCN1',             'amino acid'),
    ('L-Lysine',        'N[C@@H](CCCCN)C(=O)O',           'amino acid'),
    ('L-Histidine',     'N[C@@H](Cc1cnc[nH]1)C(=O)O',     'amino acid'),
    ('Glycine',         'NCC(=O)O',                        'amino acid'),
    ('Urea',            'NC(N)=O',                         'hydrotrope'),
    ('Caffeine',        'Cn1cnc2c1c(=O)n(c(=O)n2C)C',     'hydrotrope'),
    ('L-Arginine',      'N[C@@H](CCCNC(=N)N)C(=O)O',      'amino acid'),
    ('L-Glutamine',     'N[C@@H](CCC(N)=O)C(=O)O',        'amino acid'),
    ('Triethanolamine', 'OCCN(CCO)CCO',                    'amine'),
    ('Citric acid',     'OC(CC(O)(CC(=O)O)C(=O)O)C(=O)O', 'organic acid'),
    ('Tartaric acid',   'OC(C(O)C(=O)O)C(=O)O',           'organic acid'),
    ('Tromethamine',    'OCC(N)(CO)CO',                    'buffer amine'),
    ('Meglumine',       'OCC(O)C(O)C(O)C(O)CN',           'sugar amine'),
]
COFORMERS_POLY = [
    ('HPMC E5',       'polymer'),
    ('HPMC K15M',     'polymer'),
    ('PVP K30',       'polymer'),
    ('PEG 4000',      'polymer'),
    ('Poloxamer 188', 'polymer'),
]

def build_coformer_desc():
    desc = {}
    for name, smi, cat in COFORMERS_SM:
        mol = Chem.MolFromSmiles(smi)
        desc[name] = {'logP': Descriptors.MolLogP(mol), 'cat': cat, 'poly': False}
    for name, cat in COFORMERS_POLY:
        desc[name] = {'logP': None, 'cat': cat, 'poly': True}
    return desc

def ternary(z_bin, logP, is_poly):
    dz      = DZ_CONST if is_poly else W_LOGP * logP + CONST_COEF
    z_t     = z_bin + dz
    logKb   = max(-1.0, min((INTCPT - z_bin) / (-SLOPE), 8.0))
    logKt   = max(-1.0, min((INTCPT - z_t)  / (-SLOPE), 8.0))
    ratio   = 10 ** (-dz / abs(SLOPE))
    ddG     = -RT_298 * math.log(ratio)
    usable  = C1 <= z_t <= C2
    return {
        'dz':        round(dz,    4),
        'z_tern':    round(z_t,   4),
        'K_binary':  round(10**logKb, 1),
        'K_ternary': round(10**logKt, 1),
        'K_ratio':   round(ratio,  2),
        'ddG':       round(ddG,    2),
        'z_tern_in_range': usable,
    }

print('Loading fuzzy_predictions.csv ...')
fz = pd.read_csv('results/seed_stability/fuzzy_predictions.csv')

# Recover z_bin = logit(p_weak)
fz['z_bin'] = np.log(fz['p_weak'] / (1 - fz['p_weak']))

def dl_label(z):
    if   z > C2:  return 'Weak (High conf)'
    elif z < C1:  return 'Strong (High conf)'
    elif z > 0:   return 'Fuzzy-Weak'
    else:         return 'Fuzzy-Strong'

fz['DL_label'] = fz['z_bin'].apply(dl_label)
fz['usable']   = fz['DL_label'].str.startswith('Fuzzy')

print(f'  Total rows: {len(fz)}')
print(f'  Fuzzy Zone (usable): {fz["usable"].sum()}')
print(f'  High-confidence Weak/Strong (skipped): {(~fz["usable"]).sum()}')
print()

desc = build_coformer_desc()
n_coformers = len(desc)

print(f'[Ternary formula] {len(fz)} × {n_coformers} = {len(fz)*n_coformers} predictions...')

rows = []
for _, r in fz.iterrows():
    z    = r['z_bin']
    for cname, cinfo in desc.items():
        t = ternary(z, cinfo['logP'], cinfo['poly'])
        if not r['usable']:
            usable_note = ('SKIP — Strong (High conf): extrapolation only'
                           if 'Strong' in r['DL_label'] else
                           'SKIP — Weak (High conf): drug-CD mismatch')
        else:
            usable_note = 'OK — Fuzzy Zone'

        rows.append({
            'SMILES_Guest':  r['SMILES_Guest'],
            'Host':          r['Host'],
            'log10K_true':   round(r['log10K'], 4),
            'K_true':        round(10**r['log10K'], 1),
            'z_bin':         round(z, 4),
            'DL_label':      r['DL_label'],
            'usable':        r['usable'],
            'usable_note':   usable_note,
            'Coformer':      cname,
            'Category':      cinfo['cat'],
            'K_binary':      t['K_binary'],
            'K_ternary':     t['K_ternary'],
            'K_ratio':       t['K_ratio'],
            'ddG_kJmol':     t['ddG'],
            'z_tern':        t['z_tern'],
            'z_tern_valid':  t['z_tern_in_range'],
        })

df = pd.DataFrame(rows)

out_all = 'results/tables/fuzzy_ternary_all.csv'
df.to_csv(out_all, index=False)
print(f'-> Complete: {out_all}  ({len(df):,} rows)')

df_use = df[df['usable']]
top_per_pair = (df_use.sort_values('K_ternary', ascending=False)
                      .groupby(['SMILES_Guest','Host'])
                      .first()
                      .reset_index())
out_top = 'results/tables/fuzzy_ternary_top.csv'
top_per_pair.to_csv(out_top, index=False)
print(f'-> Best coformer per pair: {out_top}  ({len(top_per_pair):,} rows)')

top100 = df_use.sort_values('K_ternary', ascending=False).head(100)
out_100 = 'results/tables/fuzzy_ternary_top100.csv'
top100.to_csv(out_100, index=False)
print(f'-> Top-100: {out_100}')

print()
print('='*60)
print('  Summary statistics (usable rows)')
print('='*60)
print(f'  Total usable predictions : {len(df_use):,}')
print(f'  K_ternary > 10,000 M⁻¹  : {(df_use["K_ternary"] > 10000).sum():,}')
print(f'  K_ternary >  5,000 M⁻¹  : {(df_use["K_ternary"] > 5000).sum():,}')
print(f'  K_ternary >  1,000 M⁻¹  : {(df_use["K_ternary"] > 1000).sum():,}')
print()
print('  Top-10 K_ternary combinations:')
cols = ['Host','K_true','z_bin','Coformer','K_binary','K_ternary','K_ratio']
print(top100[cols].head(10).to_string(index=False))
print()
print('Done.')
