#!/bin/bash
# Sync local data/ to remote instance
set -e
source "$(dirname "$0")/config.sh"

echo "==> Syncing data..."
ssh -p "$REMOTE_PORT" "$REMOTE" "mkdir -p $REMOTE_DIR/data"
rsync -avz --progress -e "ssh -p $REMOTE_PORT" \
    ./data/ \
    "$REMOTE:$REMOTE_DIR/data/"
