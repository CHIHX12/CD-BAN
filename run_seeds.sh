#!/bin/bash
# CD-BAN Seed Stability: seeds 42–51 (10 sequential runs)
# Usage: bash run_seeds.sh

set -e
cd "$(dirname "$0")"

SEEDS=$(seq 42 51)
LOG_DIR="results/seed_stability"
mkdir -p "$LOG_DIR"

echo "======================================================"
echo "  CD-BAN Seed Stability  (seed 42~51)"
echo "  $(date)"
echo "======================================================"

for SEED in $SEEDS; do
    echo ""
    echo "[Seed $SEED] Starting training..."

    OUT="results/seed_${SEED}"
    mkdir -p "$OUT"

    python main.py \
        --cfg  configs/CDBAN.yaml \
        --seed "$SEED" \
        --out  "$OUT" \
        > "$LOG_DIR/seed_${SEED}.log" 2>&1

    RESULT=$(grep "Test @" "$LOG_DIR/seed_${SEED}.log" | tail -1)
    echo "  [Seed $SEED] $RESULT"
done

echo ""
echo "======================================================"
echo "  All seeds done  $(date)"
echo "======================================================"

# Aggregate results into summary.csv
python - <<'EOF'
import os, re, pandas as pd

rows = []
for seed in range(42, 52):
    log = f"results/seed_stability/seed_{seed}.log"
    if not os.path.exists(log):
        continue
    with open(log) as f:
        text = f.read()
    m = re.search(r'Test @ Best Epoch (\d+)\].*?AUROC=([\d.]+).*?AUPRC=([\d.]+).*?F1=([\d.]+).*?Sens=([\d.]+).*?Spec=([\d.]+).*?Acc=([\d.]+)', text)
    if m:
        rows.append({
            'seed': seed,
            'best_epoch': int(m.group(1)),
            'auroc': float(m.group(2)),
            'auprc': float(m.group(3)),
            'f1':   float(m.group(4)),
            'sens': float(m.group(5)),
            'spec': float(m.group(6)),
            'acc':  float(m.group(7)),
        })

if rows:
    df = pd.DataFrame(rows)
    print("\n=== Seed Stability Results ===")
    print(df.to_string(index=False))
    print(f"\n  AUROC: {df.auroc.mean():.4f} ± {df.auroc.std():.4f}")
    print(f"  AUPRC: {df.auprc.mean():.4f} ± {df.auprc.std():.4f}")
    print(f"  F1   : {df.f1.mean():.4f} ± {df.f1.std():.4f}")
    df.to_csv("results/seed_stability/summary.csv", index=False)
    print("\n  -> Saved: results/seed_stability/summary.csv")
EOF
