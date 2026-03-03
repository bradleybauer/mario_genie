# Mario World Model Phase 1: Data Generation

This project implements the Phase 1 step for an Action-Conditioned Mario World Model. It focuses on large-scale dataset generation, providing the engine, environments, and chunking storage components required to record gameplay experiences from Super Mario Bros. The collection captures frame observations, actions taken, environment signals, and side information (like coins, scores, and status) into sequenced files for downstream model training.

## Features

- **Dynamic Environment Wrapping**: Seamless integration with the legacy `gym_super_mario_bros` via `Gymnasium` and `shimmy` wrappers.
- **Randomized and Fixed Levels**: Ability to target specific game levels (e.g. World 1, Stage 1) or randomly sample across all 32 standard levels utilizing `RandomLevelMarioEnv`.
- **Complex Action Space**: Automatically synthesizes the `COMPLEX_MOVEMENT` discrete action space containing combinatorially valid Joypad inputs.
- **Optimized Storage**: Asynchronous, chunked sequence writing into compressed or uncompressed NumPy arrays (`.npz`) using a dedicated `ChunkWriter`. Parallel metadata capture into `.meta.json` pairs.
- **Vectorized Data Collection**: High-throughput collection traversing multiple vectorized Gymnasium environments simultaneously.
- **Multiple Agent Modes**:
  - `random`: Samples random actions uniformly.
  - `heuristic`: Action distribution explicitly tuned for basic progression heuristics (favoring moving right, jumping, and sprinting).
  - `human`: Pygame-based GUI capturing keyboard inputs for human demonstration recording.

## Project Structure

```text
├── data/
│   └── phase1/human_play/              # Default output directory for collected chunks
├── remote/
│   ├── config.sh                       # Single source of truth for host, port, and user
│   ├── connect.sh                      # Open SSH tunnel to remote instance
│   ├── setup_remote.sh                 # Sync code, install deps, create conda env
│   ├── send_data.sh                    # Sync local data/ to remote instance
│   └── get_results.sh                  # Retrieve training results from remote instance
├── scripts/
│   └── collect_phase1_vector.py        # Main execution entrypoint for data collection
└── src/
    └── mario_world_model_phase1/
        ├── actions.py                  # Joypad action space definitions (COMPLEX_MOVEMENT)
        ├── envs.py                     # Gym & Shimmy wrappers, RandomLevelMarioEnv
        ├── preprocess.py               # Frame padding logic (pad_to_square_256)
        └── storage.py                  # Asynchronous chunk writing (ChunkWriter)
```

## Remote Instance Management

The `remote/` directory contains helper scripts for working with a remote GPU instance (e.g. [Vast.ai](https://vast.ai)). Connection details (host, port, user) are defined once in `remote/config.sh` — update that file when switching instances.

Using pytorch/pytorch_2.5.1-cuda12.4-cudnn9-runtime/jupyter vastai template

| Script | Description |
|---|---|
| `config.sh` | Defines `REMOTE_HOST`, `REMOTE_PORT`, and `REMOTE_USER`. Edit this when switching instances. |
| `connect.sh` | Opens an SSH session with a local port forward (`8080`) for remote monitoring. |
| `setup_remote.sh` | Syncs code to the remote, installs `build-essential`, and creates the conda env. |
| `send_data.sh` | Rsyncs `./data/` to `/root/data/` on the remote. Run this before launching training. |
| `get_results.sh` | Rsyncs `/root/mario/results/` back to `./results/` locally. Run this after training completes. |

```bash
# Example workflow
nano config.sh                     # Define `REMOTE_HOST`, `REMOTE_PORT`, and `REMOTE_USER`
bash remote/setup_remote.sh        # sync code, install deps, create conda env
bash remote/send_data.sh           # upload dataset
bash remote/connect.sh             # SSH in and launch training
bash remote/get_results.sh         # retrieve checkpoints and logs
```

**Useful references:**
- [Vast.ai SSH connection docs](https://docs.vast.ai/documentation/instances/connect/ssh)
- [tmux cheat sheet](https://tmuxcheatsheet.com/)

## Setup and Usage

### Prerequisites

`nes-py` requires a C++ compiler to build. On a fresh Linux instance (e.g. Vast.ai), install it first:

```bash
apt-get install -y build-essential htop
```

Then create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate mario
```

### Collecting Data

Data is captured utilizing the script deployed in `scripts/collect_phase1_vector.py`. 

**Example: Collecting data via Human Play**
Play the game yourself utilizing a single environment.
```bash
python scripts/collect_phase1_vector.py \
    --mode human \
    --num-envs 1 \
    --total-steps 20000 \
    --sequences-per-chunk 512 \
    --output-dir data/phase1/human_play
```

### Analyzing Data

Verify the quality and diversity of your dataset using the analysis script. This tool reports action distribution, level coverage, and causal events (deaths/flag captures).

```bash
python scripts/analyze_phase1_data.py --data-dir data/phase1/human_play
```

Key metrics to watch:
1. **Action Distribution**: Ensure `NOOP` isn't dominating (aim for <20%) and that `Right`, `Run`, and `Jump` are well-represented.
2. **Level Coverage**: Check that the dataset covers multiple levels (not just 1-1) and that max x-positions indicate deep traversal, not just start-area loitering.
3. **Causal Events**: You want non-zero deaths and flag captures so the model learns consequences and goals.
*Keyboard keys: Arrow keys (or WASD) to move, `o` to jump (A), `p` to sprint (B).*

**Example: Collecting data via Vectorized Heuristic Bot**
Run 16 environments asynchronously capturing data using the heuristic policy. Writes compressed chunk files in the background to avoid stalling ticks.
```bash
python scripts/collect_phase1_vector.py \
    --mode heuristic \
    --level-mode random \
    --num-envs 16 \
    --total-steps 200000 \
    --sequences-per-chunk 512 \
    --async-write \
    --compress-chunks \
    --output-dir data/phase1/heuristic_play
```

## Data Format
Gameplay is stored sequence by sequence, packed into uncompressed or compressed `.npz` files.
Each `.npz` chunk holds:
- `frames`: Observation frames (padded to `256 x 256`, format: `TCHW`).
- `actions`: Selected actions per step.
- `dones`: Terminal done flags.
- Real-time internal state metadata arrays: `coins`, `flag_get`, `life`, `score`, `stage`, `time`, `world`, `x_pos`, `y_pos`, and `status` (encoded as `0: small`, `1: tall`, `2: fireball`).

Each `.npz` file contains an identical `.meta.json` equivalent indicating sequences contained alongside sequence length metrics. A global `run_stats.json` is generated mapping global action meanings and configuration defaults.

### Note on Cross-Episode Sequences
The data collection script writes out fixed-length sequences (e.g. 16 frames). Because environments auto-reset upon death or level completion, **a single sequence can seamlessly cross episode boundaries.** This means a sequence may start in one world/stage and jump to another midway through.
Downstream models utilizing 3D convolutions (like the MAGVIT-2 video tokenizer) will perceive these boundaries as sudden "scene cuts". While standard models typcially learn to compress these jump cuts properly, if you require strictly continuous patches for training, make sure to evaluate the `dones` flag arrays to mask or split the dataset accordingly.

*(Note: The current `MarioVideoDataset` implementation in the training script automatically handles this. It evaluates the `dones` array during data loading and entirely drops any sequences that contain a boundary break prior to the final frame. This ensures perfectly stable datasets for 3D continuous training by sacrificing a small amount of data volume).*

## Video Tokenizer Architecture Notes

### Spatial Token Alignment
The video tokenizer is configured to produce a **16x16 grid of discrete tokens** per frame. This is a deliberate design choice grounded in the underlying structure of the NES hardware.

**Why 16x16?**
Super Mario Bros. constructs its world almost entirely from **16x16 pixel metatiles** (also called macroblocks) — 2x2 arrangements of the base 8x8 hardware sprites. The NES engine uses a single 8-bit index (0–255) to reference each unique metatile, meaning there are at most 256 unique building blocks per level theme. Every pipe, brick, question block, and ground tile is one of these metatiles.

By targeting a 16x16 latent grid, we achieve **1 discrete token = 1 NES metatile**, which provides a strong structural inductive bias that aligns the learned codebook with the actual primitives the game is built from.

**How it is achieved:**
- Input images are downscaled to `128x128` (from the native `256x240`).
- The tokenizer applies 3 `compress_space` layers, each halving the spatial dimension ($2^3 = 8\times$ total reduction).
- $128 / 8 = 16$, yielding the 16x16 latent grid.

A native `256x256` input with 4 `compress_space` layers would achieve identical token alignment with sharper pixel fidelity, but was ruled out due to the increased computational cost.

