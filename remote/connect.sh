#!/bin/bash
source "$(dirname "$0")/config.sh"

ssh -p "$REMOTE_PORT" "$REMOTE" -L 8080:localhost:8080