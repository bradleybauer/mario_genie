#!/bin/bash
# =============================================================================
# Architecture & Capacity Sweep for MAGVIT-2 Video Tokenizer
# =============================================================================
#
# Sweeps three axes that most affect generalisation from small->large data:
#   1. Model capacity   (init_dim)       — controls channel width at every layer
#   2. Codebook size    (codebook_size)   — discrete bottleneck capacity
#   3. Architecture     (layers)          — depth + attention layers
#
# Each configuration is given a unique --run-name and writes to its own
# sub-directory under OUTPUT_DIR.  All runs share the same seed and data
# so results are directly comparable.
#
# Usage:
#   bash scripts/sweep_arch.sh                   # full sweep
#   DRY_RUN=1 bash scripts/sweep_arch.sh         # print commands without running
#   SANITY_CHECK=1 bash scripts/sweep_arch.sh    # validate all configs then exit
#   GPUS=0,1 bash scripts/sweep_arch.sh          # select GPUs
# =============================================================================
set -euo pipefail

DATA_DIR="${DATA_DIR:-data/human_play}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/magvit2}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-1e-4}"
AUTO_BATCH="${AUTO_BATCH:-1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
VAL_INTERVAL="${VAL_INTERVAL:-200}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-48}"
DRY_RUN="${DRY_RUN:-0}"
SANITY_CHECK="${SANITY_CHECK:-0}"
SANITY_TIMEOUT="${SANITY_TIMEOUT:-120}"
GPUS="${GPUS:-0}"

export CUDA_VISIBLE_DEVICES="$GPUS"

if [[ "$AUTO_BATCH" == "1" ]]; then
    BATCH_ARG="--auto-batch-size"
else
    BATCH_ARG="--batch-size $BATCH_SIZE"
fi

COMMON="--data-dir $DATA_DIR --epochs $EPOCHS --lr $LR $BATCH_ARG \
--val-interval $VAL_INTERVAL --seed $SEED --num-workers $NUM_WORKERS \
--output-dir $OUTPUT_DIR"

run_count=0
declare -a RUN_NAMES=()
declare -a RUN_ARGS=()

run() {
    local name="$1"; shift
    run_count=$((run_count + 1))
    RUN_NAMES+=("$name")
    RUN_ARGS+=("--run-name $name $COMMON $*")
    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo "[$run_count] $name"
    echo "══════════════════════════════════════════════════════════════"
    local cmd="python scripts/train_magvit.py --run-name $name $COMMON $*"
    echo "$cmd"
    if [[ "$DRY_RUN" == "1" ]] || [[ "$SANITY_CHECK" == "1" ]]; then
        echo "(deferred — will run after all configs are collected)"
    else
        eval "$cmd"
    fi
}

# ---------------------------------------------------------------------------
# Axis 1: Model capacity (init_dim) — baseline architecture
# ---------------------------------------------------------------------------
BASELINE_LAYERS="residual,compress_space,residual,compress_space,residual,compress_space"
BASELINE_CB=8192

for DIM in 32 64 128; do
    run "dim${DIM}_cb${BASELINE_CB}" \
        --init-dim "$DIM" --codebook-size "$BASELINE_CB" --layers "$BASELINE_LAYERS"
done

# ---------------------------------------------------------------------------
# Axis 2: Codebook size — with a reasonably sized model
# ---------------------------------------------------------------------------
CAP_DIM=64

for CB in 8192 16384; do
    run "dim${CAP_DIM}_cb${CB}" \
        --init-dim "$CAP_DIM" --codebook-size "$CB" --layers "$BASELINE_LAYERS"
done

# ---------------------------------------------------------------------------
# Axis 3: Architecture depth & attention
# ---------------------------------------------------------------------------
ARCH_DIM=64
ARCH_CB=8192

# Deeper: extra residual blocks between each spatial compression
DEEP="residual,residual,compress_space,residual,residual,compress_space,residual,residual,compress_space"
run "dim${ARCH_DIM}_cb${ARCH_CB}_deep" \
    --init-dim "$ARCH_DIM" --codebook-size "$ARCH_CB" --layers "$DEEP"

# Spatial attention after each compression
ATTN_SPACE="residual,compress_space,attend_space,residual,compress_space,attend_space,residual,compress_space,attend_space"
run "dim${ARCH_DIM}_cb${ARCH_CB}_attn_space" \
    --init-dim "$ARCH_DIM" --codebook-size "$ARCH_CB" --layers "$ATTN_SPACE"

# Linear attention (cheaper) after each compression
LIN_ATTN_SPACE="residual,compress_space,residual,compress_space,linear_attend_space,residual,compress_space,linear_attend_space"
run "dim${ARCH_DIM}_cb${ARCH_CB}_lin_attn" \
    --init-dim "$ARCH_DIM" --codebook-size "$ARCH_CB" --layers "$LIN_ATTN_SPACE"

# Temporal attention (causal) — helpful for multi-frame coherence
ATTN_TIME="residual,compress_space,attend_time,residual,compress_space,attend_time,residual,compress_space,attend_time"
run "dim${ARCH_DIM}_cb${ARCH_CB}_attn_time" \
    --init-dim "$ARCH_DIM" --codebook-size "$ARCH_CB" --layers "$ATTN_TIME"

# Full: spatial + temporal attention after each compression
FULL_ATTN="residual,compress_space,residual,compress_space,attend_space,attend_time,residual,compress_space,attend_space,attend_time"
run "dim${ARCH_DIM}_cb${ARCH_CB}_full_attn" \
    --init-dim "$ARCH_DIM" --codebook-size "$ARCH_CB" --layers "$FULL_ATTN"

# # ---------------------------------------------------------------------------
# # "Kitchen sink" — large model with attention + big codebook
# # Attention is deferred past the first compression to fit in VRAM;
# # linear attention at the second level, full attention only at 16×16.
# # ---------------------------------------------------------------------------
# KITCHEN_SINK="residual,compress_space,residual,compress_space,linear_attend_space,attend_time,residual,compress_space,attend_space,attend_time"
# run "dim128_cb4096_kitchen" \
#     --init-dim 128 --codebook-size 4096 --layers "$KITCHEN_SINK"

# ---------------------------------------------------------------------------
# Sanity-check pre-flight: validate every config before any real training
# ---------------------------------------------------------------------------
if [[ "$SANITY_CHECK" == "1" ]]; then
    echo ""
    echo "┌──────────────────────────────────────────────────────────────┐"
    echo "│  SANITY CHECK — validating $run_count configurations               │"
    echo "└──────────────────────────────────────────────────────────────┘"
    passed=0
    failed=0
    failed_names=()
    for i in "${!RUN_ARGS[@]}"; do
        name="${RUN_NAMES[$i]}"
        cmd="python scripts/train_magvit.py --sanity-check ${RUN_ARGS[$i]}"
        echo ""
        echo "--- [$((i+1))/$run_count] $name ---"
        if timeout "$SANITY_TIMEOUT" bash -c "$cmd"; then
            passed=$((passed + 1))
        else
            exit_code=$?
            failed=$((failed + 1))
            failed_names+=("$name")
            if [[ $exit_code -eq 124 ]]; then
                echo "[sanity-check] FAIL  $name  timed out after ${SANITY_TIMEOUT}s"
            fi
        fi
    done
    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo "Sanity check complete:  $passed passed,  $failed failed  (out of $run_count)"
    if [[ $failed -gt 0 ]]; then
        echo "FAILED runs: ${failed_names[*]}"
        exit 1
    fi
    echo "All configurations OK — safe to run the full sweep."
    exit 0
fi

# ---------------------------------------------------------------------------
echo ""
echo "=== Sweep complete ($run_count runs) ==="
echo "Results in: $OUTPUT_DIR/{$(IFS=,; echo "${RUN_NAMES[*]}")}"
echo ""
echo "Compare with:  python scripts/compare_sweeps.py --results-dir $OUTPUT_DIR"
