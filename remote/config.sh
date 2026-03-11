#!/bin/bash
# Remote instance connection config

# https://docs.vast.ai/documentation/instances/connect/ssh
# https://tmuxcheatsheet.com/
#
# to scroll up in tmux session: Ctrl-b then [ then pgup/pgdn or arrow keys, q to exit scroll mode

#4090
REMOTE_HOST="188.36.196.221"
REMOTE_PORT=5398

# #H100
REMOTE_HOST="69.143.221.78"
REMOTE_PORT=23730

REMOTE_USER="root"
REMOTE="${REMOTE_USER}@${REMOTE_HOST}"
REMOTE_HOME=$([ "$REMOTE_USER" = "root" ] && echo "/root" || echo "/home/$REMOTE_USER")
REMOTE_DIR="$REMOTE_HOME/mario"
