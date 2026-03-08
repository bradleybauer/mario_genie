#!/bin/bash
# Retrieve training results from remote server
source "$(dirname "$0")/config.sh"

CHECKPOINT_SUBDIR=""

for arg in "$@"; do
    if [[ "$arg" == "--images" ]]; then
        continue
    fi

    if [[ -n "$CHECKPOINT_SUBDIR" ]]; then
        echo "Usage: $0 [checkpoint-subdir] [--images]"
        exit 1
    fi

    CHECKPOINT_SUBDIR="$arg"
done

REMOTE_RESULTS="$REMOTE_DIR/checkpoints/"
LOCAL_RESULTS="./results/"

if [[ -n "$CHECKPOINT_SUBDIR" ]]; then
    REMOTE_RESULTS+="$CHECKPOINT_SUBDIR/"
    LOCAL_RESULTS+="$CHECKPOINT_SUBDIR/"
fi

mkdir -p "$LOCAL_RESULTS"

RSYNC_OPTS=(-avz --progress)

if [[ " $* " == *" --images "* ]]; then
    RSYNC_OPTS+=(--include='*/' --include='*.png' --include='*.json' --exclude='*')
fi

rsync "${RSYNC_OPTS[@]}" -e "ssh -p $REMOTE_PORT" "$REMOTE:$REMOTE_RESULTS" "$LOCAL_RESULTS"
