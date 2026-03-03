#!/bin/bash
# Remote instance connection config

# https://docs.vast.ai/documentation/instances/connect/ssh
# https://tmuxcheatsheet.com/
#
# to scroll up in tmux session: Ctrl-b then [ then pgup/pgdn or arrow keys, q to exit scroll mode

REMOTE_HOST="136.59.129.136"
REMOTE_PORT=33814 
REMOTE_USER="root"
REMOTE="${REMOTE_USER}@${REMOTE_HOST}"
REMOTE_HOME=$([ "$REMOTE_USER" = "root" ] && echo "/root" || echo "/home/$REMOTE_USER")
REMOTE_DIR="$REMOTE_HOME/mario"
