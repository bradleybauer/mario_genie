#!/bin/bash
set -e

DATA_DIR="data/human_play/"
COMMON="--overfit-one --epochs 10 --lr 3e-4 --val-interval 500 --seed 42"

for CB_SIZE in 256 1024 4096 8192; do
    echo "=== Codebook size: $CB_SIZE ==="
    python scripts/train_magvit.py \
        --data-dir "$DATA_DIR" \
        --codebook-size "$CB_SIZE" \
        --run-name "codebook_${CB_SIZE}" \
        $COMMON
done

echo "=== Sweep complete ==="
echo "Results in checkpoints/magvit2/codebook_*/"
