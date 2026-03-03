#!/bin/bash
# Retrieve training results from remote server
source "$(dirname "$0")/config.sh"

REMOTE_RESULTS="$REMOTE_DIR/results/"
LOCAL_RESULTS="./results/"

mkdir -p "$LOCAL_RESULTS"
rsync -avz --progress -e "ssh -p $REMOTE_PORT" "$REMOTE:$REMOTE_RESULTS" "$LOCAL_RESULTS"
