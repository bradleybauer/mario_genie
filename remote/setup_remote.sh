#!/bin/bash
# Full remote setup: sync code + data, install dependencies, create conda env
set -e
source "$(dirname "$0")/config.sh"

echo "==> Syncing code..."
rsync -avz --progress --exclude='__pycache__' -e "ssh -p $REMOTE_PORT" \
    ./src \
    ./scripts \
    ./environment.yml \
    "$REMOTE:$REMOTE_DIR/"

echo "==> Installing build tools and conda environment..."
ssh -p "$REMOTE_PORT" "$REMOTE" "
  set -e
  apt-get install -y build-essential htop vim
  conda init bash 2>/dev/null || true
  # Source conda so it's available in this non-interactive session
  eval \"\$(/opt/miniforge3/bin/conda shell.bash hook)\"
  cd $REMOTE_DIR
  conda env create -f environment.yml || conda env update -f environment.yml
  echo 'Setup complete. Activate with: conda activate mario'
"
