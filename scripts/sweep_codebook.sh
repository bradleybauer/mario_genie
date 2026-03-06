#!/bin/bash
set -e

DATA_DIR="data/human_play/"
COMMON="--overfit-n 2 --epochs 25 --lr 3e-4 --val-interval 500 --seed 44"

for CB_SIZE in 256 1024 4096 8192; do
    echo "=== Codebook size: $CB_SIZE ==="
    python scripts/train_magvit.py \
        --data-dir "$DATA_DIR" \
        --codebook-size "$CB_SIZE" \
        --run-name "codebook_${CB_SIZE}" \
        --no-shuffle \
        --batch-size 2 \
        $COMMON
done

echo "=== Sweep complete ==="
echo "Results in checkpoints/magvit2/codebook_*/"
