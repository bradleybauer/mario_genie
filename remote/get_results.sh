#!/bin/bash
# Retrieve training results from remote server
source "$(dirname "$0")/config.sh"

REMOTE_RESULTS="$REMOTE_DIR/checkpoints/"
LOCAL_RESULTS="./results/"

mkdir -p "$LOCAL_RESULTS"

RSYNC_OPTS=(-avz --progress)

if [[ "$1" == "--images" ]]; then
    RSYNC_OPTS+=(--include='*/' --include='*.png' --exclude='*')
fi

rsync "${RSYNC_OPTS[@]}" -e "ssh -p $REMOTE_PORT" "$REMOTE:$REMOTE_RESULTS" "$LOCAL_RESULTS"
