"""
predict_ternary_formula.py — Ternary descriptor correction formula
===================================================================
Extension of the binary formula: given z_bin from the CD-BAN model and
the SMILES of an auxiliary coformer ligand, estimates the ternary
inclusion complex binding constant K_ternary and free energy change ΔΔG.

Formula
-------
  Binary (from model):
    z_bin = CD-BAN(SMILES_Guest, SMILES_Host)

  Ternary correction (this script):
    Δz     = w_logP × logP(Ligand) + c
    z_tern = z_bin + Δz
    P_tern(Weak) = σ(z_tern)
    K_tern ≈ K_binary × 10^(−Δz / 1.128)
    ΔΔG   = −RT × ln(K_tern / K_binary)  kJ/mol

Descriptors (computed from SMILES_Ligand via RDKit):
    logP   = Wildman-Crippen partition coefficient
    TPSA   = topological polar surface area (Å²)
    charge = net formal charge

Fitted coefficients (n=4 literature ternary data points):
----------------------------------------------------------
  Ligand       logP    Δz_obs
  HPMC E5     -1.26   -0.614
  PEG 4000    -1.03   -0.695
  L-arginine  -1.34   -0.679
  HPMC K15M   -1.26   -0.772

  Fit: Δz = -0.0378 × logP + (-0.7361)
  RMSE = 0.056  (comparable to constant model SD = 0.065)

⚠️  Limitations
---------------
  1. Only 4 training points — do not over-interpret coefficients
  2. Polymer coformers (PVP, HPMC, PEG) approximated by monomer SMILES
  3. ≥30 diverse small-molecule coformers needed for a reliable multi-parameter fit
  4. logP correlation r = -0.078 (weak) — use --constant for polymer coformers

Full derivation: ALGORITHM.md

Usage
-----
  # Predict ternary K from z_bin and ligand SMILES
  python predict_ternary_formula.py \\
    --z 1.2308 \\
    --ligand "N[C@@H](CCCNC(=N)N)C(=O)O"   # L-arginine

  # Chain with predict_model.py
  python predict_model.py --guest "..." --host "..."
  # -> get z_bin, e.g. z_bin = +1.2308
  python predict_ternary_formula.py --z 1.2308 --ligand "SMILES_L"

  # Use constant model (more robust when data are scarce)
  python predict_ternary_formula.py --z 1.2308 --constant
"""

import argparse, math, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

# Fitted coefficients (n=4 literature data points)
W_LOGP     = -0.0378   # Δz = W_LOGP × logP + CONST
CONST      = -0.7361
DZ_CONST   = -0.690    # constant model (fallback, more robust)
DZ_SD      =  0.065    # constant model SD

SLOPE      = -1.128    # calibration: logK = (3.714 - z) / 1.128
INTCPT     =  3.714
RT_298     =  2.479    # kJ/mol
C2, C1     =  1.458, -0.799


def ligand_descriptors(smiles: str) -> dict:
    """Compute ligand descriptors via RDKit."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f'Invalid SMILES: {smiles}')
        return {
            'logP':   round(Descriptors.MolLogP(mol), 3),
            'TPSA':   round(rdMolDescriptors.CalcTPSA(mol), 2),
            'MW':     round(Descriptors.MolWt(mol), 2),
            'charge': int(sum(a.GetFormalCharge() for a in mol.GetAtoms())),
            'HBD':    rdMolDescriptors.CalcNumHBD(mol),
            'HBA':    rdMolDescriptors.CalcNumHBA(mol),
        }
    except ImportError:
        raise ImportError('rdkit required: pip install rdkit')


def predict_ternary(z_bin: float, ligand_smiles: str = None,
                    use_constant: bool = False) -> dict:
    """
    Ternary correction formula.

    descriptor-based:
      Δz = W_LOGP × logP(Ligand) + CONST
      -> requires ligand_smiles, computes logP

    constant (fallback):
      Δz = -0.690  (ligand-independent)
    """
    desc = None

    if use_constant or ligand_smiles is None:
        dz     = DZ_CONST
        method = 'constant (n=4, SD=0.065, ligand SMILES not used)'
    else:
        desc = ligand_descriptors(ligand_smiles)
        dz   = W_LOGP * desc['logP'] + CONST
        method = 'descriptor: Δz = {:.4f}×logP + ({:.4f})'.format(W_LOGP, CONST)

    z_tern   = z_bin + dz
    p_tern   = 1.0 / (1.0 + math.exp(-z_tern))

    logK_bin  = max(-1., min((INTCPT - z_bin)  / (-SLOPE), 8.))
    logK_tern = max(-1., min((INTCPT - z_tern) / (-SLOPE), 8.))
    K_bin     = 10 ** logK_bin
    K_tern    = 10 ** logK_tern
    K_ratio   = 10 ** (-dz / abs(SLOPE))
    ddG       = -RT_298 * math.log(K_ratio)

    valid = (C1 <= z_tern <= C2)

    return {
        'z_bin':       round(z_bin, 4),
        'ligand_logP': desc['logP'] if desc else None,
        'ligand_TPSA': desc['TPSA'] if desc else None,
        'ligand_charge': desc['charge'] if desc else None,
        'delta_z':     round(dz, 4),
        'method':      method,
        'z_tern':      round(z_tern, 4),
        'p_weak_tern': round(p_tern, 4),
        'K_bin_est':   round(K_bin, 1),
        'K_tern_est':  round(K_tern, 1),
        'K_ratio':     round(K_ratio, 2),
        'ddG_kJmol':   round(ddG, 2),
        'calib_valid': valid,
    }


def print_result(r: dict, ligand_name: str = ''):
    sep = '─' * 62
    print(sep)
    if ligand_name:
        print(f'  Coformer: {ligand_name}')
    print(sep)
    print('  [Binary formula result]')
    print(f'    z_bin      = {r["z_bin"]:+.4f}  (from predict_model.py)')
    print(f'    K_binary   ≈ {r["K_bin_est"]:,.1f} M⁻¹')
    print()
    print('  [Ternary extension]')
    if r['ligand_logP'] is not None:
        print(f'    logP(L)    = {r["ligand_logP"]}')
        print(f'    TPSA(L)    = {r["ligand_TPSA"]} Å²')
        print(f'    charge(L)  = {r["ligand_charge"]}')
    print(f'    Δz         = {r["delta_z"]:+.4f}  ({r["method"]})')
    print(f'    z_tern     = z_bin + Δz = {r["z_tern"]:+.4f}')
    print(f'    P(Weak, ternary) = σ(z_tern) = {r["p_weak_tern"]:.4f}')
    print(f'    K_ternary  ≈ {r["K_tern_est"]:,.1f} M⁻¹  (× {r["K_ratio"]:.2f})')
    print(f'    ΔΔG        ≈ {r["ddG_kJmol"]:.2f} kJ/mol')
    if not r['calib_valid']:
        print(f'    ⚠️  z_tern outside calibration range — K is extrapolated')
    print()


def main():
    ap = argparse.ArgumentParser(
        description='Ternary descriptor correction formula — extension of binary formula',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Full derivation: ALGORITHM.md'
    )
    ap.add_argument('--z',        type=float, help='z_bin (from predict_model.py)')
    ap.add_argument('--ligand',   type=str,   help='Coformer SMILES (used to compute logP, TPSA, charge)')
    ap.add_argument('--name',     type=str,   default='', help='Coformer name')
    ap.add_argument('--constant', action='store_true', help='Use constant model (no ligand SMILES needed)')
    ap.add_argument('--csv',      type=str,   help='Batch CSV (must have z_bin, SMILES_Ligand columns)')
    ap.add_argument('--out',      type=str,   help='Output CSV path')
    args = ap.parse_args()

    if args.z is None and not args.csv:
        ap.print_help(); return

    print()
    print('[predict_ternary_formula.py]  Ternary descriptor correction formula')
    print('  Formula: Δz = {:.4f}×logP + ({:.4f})   [n=4, RMSE=0.056]'.format(W_LOGP, CONST))
    print('  Fallback: Δz = {:.3f}  (constant, SD={})'.format(DZ_CONST, DZ_SD))
    print('  ⚠️  n=4 data points — use --constant for polymer coformers')
    print()

    if args.z is not None:
        r = predict_ternary(args.z, args.ligand, args.constant)
        print_result(r, args.name)

    elif args.csv:
        df = pd.read_csv(args.csv)
        rows = []
        for _, row in df.iterrows():
            z   = row['z_bin']
            smi = row.get('SMILES_Ligand', None)
            r   = predict_ternary(z, smi, args.constant)
            r['name'] = row.get('name', '')
            rows.append(r)
            print_result(r, r['name'])
        if args.out:
            pd.DataFrame(rows).to_csv(args.out, index=False)
            print(f'-> Saved: {args.out}')


if __name__ == '__main__':
    main()
