#!/bin/bash
# CD-BAN Seed Stability: 4-GPU parallel, seeds 42–51
# GPU 0 → seeds 42, 43, 44
# GPU 1 → seeds 45, 46, 47
# GPU 2 → seeds 48, 49
# GPU 3 → seeds 50, 51
# Each GPU runs its seeds sequentially; all 4 groups run in parallel.

set -e
cd "$(dirname "$0")"

mkdir -p results/seed_stability

echo "======================================================"
echo "  CD-BAN Seed Stability  (4 GPU parallel)"
echo "  $(date)"
echo "======================================================"

run_on_gpu() {
    local GPU=$1
    shift
    local SEEDS=("$@")
    for SEED in "${SEEDS[@]}"; do
        echo "[GPU $GPU] Seed $SEED starting..."
        CUDA_VISIBLE_DEVICES=$GPU python main.py \
            --cfg  configs/CDBAN.yaml \
            --seed "$SEED" \
            --out  "results/seed_${SEED}" \
            > "results/seed_stability/seed_${SEED}.log" 2>&1
        RESULT=$(grep "Test @" "results/seed_stability/seed_${SEED}.log" | tail -1)
        echo "[GPU $GPU] Seed $SEED done  $RESULT"
    done
}

run_on_gpu 0 42 43 44 &
run_on_gpu 1 45 46 47 &
run_on_gpu 2 48 49    &
run_on_gpu 3 50 51    &

wait
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
    m = re.search(
        r'Test @ Best Epoch (\d+)\].*?AUROC=([\d.]+).*?AUPRC=([\d.]+)'
        r'.*?F1=([\d.]+).*?Sens=([\d.]+).*?Spec=([\d.]+).*?Acc=([\d.]+)', text)
    if m:
        rows.append({
            'seed':       seed,
            'best_epoch': int(m.group(1)),
            'auroc':      float(m.group(2)),
            'auprc':      float(m.group(3)),
            'f1':         float(m.group(4)),
            'sensitivity':float(m.group(5)),
            'specificity':float(m.group(6)),
            'accuracy':   float(m.group(7)),
        })

if rows:
    df = pd.DataFrame(rows).sort_values('seed')
    print("\n=== Seed Stability Summary ===")
    print(df.to_string(index=False))
    print(f"\n  AUROC : {df.auroc.mean():.4f} ± {df.auroc.std():.4f}")
    print(f"  AUPRC : {df.auprc.mean():.4f} ± {df.auprc.std():.4f}")
    print(f"  F1    : {df.f1.mean():.4f} ± {df.f1.std():.4f}")
    print(f"  Acc   : {df.accuracy.mean():.4f} ± {df.accuracy.std():.4f}")
    df.to_csv("results/seed_stability/summary.csv", index=False)
    print("\n  -> Saved: results/seed_stability/summary.csv")
EOF
