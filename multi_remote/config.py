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
    {"name": "a", "host": "38.117.87.38", "port": 43783, "user": "root"},
    {"name": "b", "host": "86.127.245.129", "port": 22773, "user": "root"},
    {"name": "c", "host": "188.36.196.221", "port": 40857, "user": "root"},
    {"name": "d", "host": "79.112.2.29", "port": 20394 , "user": "root"},
    {"name": "f", "host": "79.112.2.29", "port": 21712 , "user": "root"},
    {"name": "g", "host": "79.112.58.103", "port": 35103, "user": "root"},
    {"name": "h", "host": "38.117.87.38", "port": 45488, "user": "root"},
    {"name": "i", "host": "79.112.58.103", "port": 35247, "user": "root"},

]
# fmt: on

# Project directory name on remote machines (created inside the user's home dir)
PROJECT_NAME = "mario"
