"""
Multi-machine worker inventory.

Edit WORKERS to match your current remote instances.
Each entry needs: name, host, port, user.

Example for Vast.ai instances:
    {"name": "h100a", "host": "154.57.34.100", "port": 19805, "user": "root"},

Tips:
    - Use `tmux attach -t sweep` on a worker to see live output
    - Worker names are used as identifiers in logs and result directories
"""

# fmt: off
WORKERS = [
    {"name": "a", "host": "108.255.76.60", "port": 65248 , "user": "root"},
]
# fmt: on

# Project directory name on remote machines (created inside the user's home dir)
PROJECT_NAME = "mario"
