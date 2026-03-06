#!/bin/bash
# Sync local code to remote instance (without full setup/install)
set -e
source "$(dirname "$0")/config.sh"

echo "==> Syncing code..."
rsync -avz --progress \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='data' \
    --exclude='checkpoints' \
    --exclude='results' \
    --exclude='*.pyc' \
    -e "ssh -p $REMOTE_PORT" \
    ./src \
    ./scripts \
    ./environment.yml \
    "$REMOTE:$REMOTE_DIR/"

echo "==> Code synced."
