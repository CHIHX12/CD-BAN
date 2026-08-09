"""
predict_model.py — CD-BAN model inference
==========================================
Inputs two SMILES strings and runs the trained deep learning model
to output the inclusion complex affinity logit z_bin.

This is neural network inference, not a formula.
z_bin and P(Weak) are computed by the network forward pass.

Requires: results/seed_*/best_model_epoch_*.pth (model weights)

Usage
-----
  # Single prediction
  python predict_model.py --guest "SMILES" --host "SMILES" --name "name"

  # Batch (CSV must have columns: name, SMILES_Guest, SMILES_Host)
  python predict_model.py --csv input.csv --out output.csv

  # Use a specific seed (default: seed 49, best AUROC = 0.9655)
  python predict_model.py --guest "..." --host "..." --seed 42

Output
------
  z_bin      : model logit (real value; more negative = stronger binder)
  P(Weak)    : P(K < 100 M⁻¹); higher = weaker binder
  label      : Strong / Weak / Fuzzy Zone
  confidence : High (z < -0.799 or z > +1.458) / Low (intermediate)

Model architecture (full derivation in ALGORITHM.md):
  φ(a) ∈ ℝ^74 → GCNConv ×3 → padding(290) → BANLayer(2 heads) → MLP(256→512→512→128→1)
  z_bin = MLP(BAN(GCN(G), GCN(H)))    total parameters: 780,550
  P(Weak) = σ(z_bin)

Validation (full dataset n=1,198):
  AUROC 0.9895 | AUPRC 0.9990 | Accuracy 97.66%
"""

import argparse, sys, os, warnings
warnings.filterwarnings('ignore')

import torch
import yaml
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import CDBAN
from dataloader import smiles_to_pyg
from torch_geometric.data import Batch

# Thresholds from ALGORITHM.md calibration
C2 =  1.458   # z > C2 -> Confident Weak   (K < 100)
C1 = -0.799   # z < C1 -> Confident Strong (K > 10,000)
DEFAULT_SEED = 49


def load_model(seed: int = DEFAULT_SEED) -> CDBAN:
    with open('configs/CDBAN.yaml') as f:
        cfg = yaml.safe_load(f)
    model = CDBAN(**cfg)
    seed_dir = f'results/seed_{seed}'
    pth = sorted([p for p in os.listdir(seed_dir) if p.startswith('best_model')])[-1]
    model.load_state_dict(torch.load(os.path.join(seed_dir, pth), map_location='cpu'))
    model.eval()
    return model


def predict(model: CDBAN, smiles_guest: str, smiles_host: str) -> dict:
    """
    Model forward pass: 2 SMILES -> z_bin, P(Weak)

    Returns
    -------
    dict with keys: z_bin, p_weak, label, confidence
    """
    g = smiles_to_pyg(smiles_guest)
    h = smiles_to_pyg(smiles_host)
    bg = Batch.from_data_list([g])
    bh = Batch.from_data_list([h])
    with torch.no_grad():
        _, _, score, _ = model(bg, bh, mode='eval')

    z   = score.item()
    p   = torch.sigmoid(score).item()

    if z > C2:
        label, conf = 'Weak Binder (K < 100 M⁻¹)', 'High'
    elif z < C1:
        label, conf = 'Strong Binder (K > 10,000 M⁻¹)', 'High'
    elif z > 0:
        label, conf = 'Likely Weak — Fuzzy Zone', 'Low'
    else:
        label, conf = 'Likely Strong — Fuzzy Zone', 'Low'

    return {'z_bin': round(z, 4), 'p_weak': round(p, 4),
            'label': label, 'confidence': conf}


def print_result(name: str, r: dict):
    sep = '─' * 56
    print(sep)
    print(f'  {name}')
    print(sep)
    print(f'  z_bin      = {r["z_bin"]:+.4f}   <- model logit')
    print(f'  P(Weak)    = {r["p_weak"]:.4f}    <- σ(z_bin)')
    print(f'  Label      = {r["label"]}')
    print(f'  Confidence = {r["confidence"]}')
    print(f'  (for K estimate, run predict_calibration.py)')
    print()


def main():
    ap = argparse.ArgumentParser(description='CD-BAN model inference (deep learning only)')
    ap.add_argument('--guest', type=str)
    ap.add_argument('--host',  type=str)
    ap.add_argument('--name',  type=str, default='compound')
    ap.add_argument('--csv',   type=str)
    ap.add_argument('--out',   type=str)
    ap.add_argument('--seed',  type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    if not args.guest and not args.csv:
        ap.print_help(); return

    model = load_model(args.seed)
    print(f'\n[predict_model.py]  seed={args.seed} | model inference (not a formula)\n')

    if args.guest:
        r = predict(model, args.guest, args.host)
        print_result(args.name, r)

    elif args.csv:
        df = pd.read_csv(args.csv)
        rows = []
        for _, row in df.iterrows():
            r = predict(model, row['SMILES_Guest'], row['SMILES_Host'])
            r['name'] = row.get('name', '')
            rows.append(r)
            print_result(r['name'], r)
        if args.out:
            pd.DataFrame(rows).to_csv(args.out, index=False)
            print(f'-> Saved: {args.out}')


if __name__ == '__main__':
    main()
