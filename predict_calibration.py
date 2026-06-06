"""
predict_calibration.py — Calibration formula: z_bin → K estimate
=================================================================
Converts the model logit z_bin into an estimated binding constant K (M⁻¹).

This is a mathematical formula, not a neural network.
No model weights are required — only z_bin from predict_model.py.

Formula derivation
------------------
Derived from boundary conditions (not OLS regression):
  z = +1.458  corresponds to K = 100 M⁻¹    (lower boundary of Fuzzy Zone)
  z = −0.799  corresponds to K = 10,000 M⁻¹ (upper boundary of Fuzzy Zone)
  -> two-point linear fit: z ≈ −1.128 × log₁₀K + 3.714
  -> inverse: log₁₀K ≈ (3.714 − z_bin) / 1.128

Validation (n=1,850 fuzzy-zone compounds)
------------------------------------------
  Pearson r(z_bin, log₁₀K)   = -0.381  (p < 0.001)
  Pearson r(P_weak, log₁₀K)  = -0.468  (p < 0.001)  (reported in Fig. 4)
  Spearman ρ                  = -0.425  (p < 0.001)
  Valid range: z ∈ (−0.799, +1.458)  i.e. K ∈ 100–10,000 M⁻¹
  Outside this range: high-confidence labels are reliable; K values are extrapolated.

Usage
-----
  # Single z_bin value
  python predict_calibration.py --z 1.23

  # From predict_model.py output CSV
  python predict_calibration.py --csv model_output.csv --out k_estimates.csv
"""

import argparse, math, sys
import pandas as pd

# Calibration coefficients (boundary-condition derivation: z=+1.458↔K=100, z=−0.799↔K=10,000)
SLOPE     = -1.128
INTERCEPT =  3.714
K_MIN     =  0.1       # extrapolation lower clamp
K_MAX     =  1e8       # extrapolation upper clamp
C2        =  1.458     # K=100  boundary
C1        = -0.799     # K=10k  boundary


def z_to_K(z: float) -> dict:
    """
    Formula: log₁₀K = (INTERCEPT − z) / (−SLOPE) = (3.714 − z) / 1.128

    Returns
    -------
    dict: log10K_est, K_est_M, valid, note
    """
    log10K = (INTERCEPT - z) / (-SLOPE)
    log10K_clamped = max(math.log10(K_MIN), min(log10K, math.log10(K_MAX)))
    K_est = 10 ** log10K_clamped
    valid = (C1 <= z <= C2)
    note = '(within calibration range)' if valid else '(outside calibration range — extrapolated)'

    return {
        'z_bin':      round(z, 4),
        'log10K_est': round(log10K_clamped, 3),
        'K_est_M':    round(K_est, 1),
        'valid':      valid,
        'note':       note,
    }


def print_result(r: dict):
    print(f'  z_bin     = {r["z_bin"]:+.4f}')
    print(f'  Formula   : log₁₀K = (3.714 − z) / 1.128')
    print(f'  log₁₀K   ≈ {r["log10K_est"]:.3f}')
    print(f'  K_est     ≈ {r["K_est_M"]:,.1f} M⁻¹  {r["note"]}')
    print()


def main():
    ap = argparse.ArgumentParser(description='z_bin -> K estimate (mathematical calibration formula)')
    ap.add_argument('--z',   type=float, help='z_bin value (from predict_model.py)')
    ap.add_argument('--csv', type=str,   help='CSV with z_bin column')
    ap.add_argument('--out', type=str,   help='Output CSV path')
    args = ap.parse_args()

    if args.z is None and not args.csv:
        ap.print_help(); return

    print('\n[predict_calibration.py]  Mathematical calibration formula (no model)')
    print('  Formula: log₁₀K ≈ (3.714 − z) / 1.128')
    print('  Source: boundary-condition derivation (z=+1.458↔K=100, z=−0.799↔K=10,000)'
          '  Pearson r(z,logK)=−0.381\n')

    if args.z is not None:
        r = z_to_K(args.z)
        print_result(r)

    elif args.csv:
        df = pd.read_csv(args.csv)
        if 'z_bin' not in df.columns:
            print('Error: CSV must have a z_bin column'); sys.exit(1)
        results = [z_to_K(z) for z in df['z_bin']]
        out_df = pd.DataFrame(results)
        print(out_df.to_string(index=False))
        if args.out:
            out_df.to_csv(args.out, index=False)
            print(f'\n-> Saved: {args.out}')


if __name__ == '__main__':
    main()
