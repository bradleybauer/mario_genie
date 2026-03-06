# Mario World Model: Data Generation

This project implements an Action-Conditioned Mario World Model. It focuses on large-scale dataset generation, providing the engine, environments, and chunking storage components required to record gameplay experiences from Super Mario Bros. The collection captures frame observations, actions taken, environment signals, and side information (like coins, scores, and status) into sequenced files for downstream model training.

## Features

- **Randomized and Fixed Levels**: Ability to target specific game levels (e.g. World 1, Stage 1) or randomly sample across all 32 standard levels utilizing `RandomLevelMarioEnv`.
- **Complex Action Space**: Automatically synthesizes the `COMPLEX_MOVEMENT` discrete action space containing combinatorially valid Joypad inputs.
- **Optimized Storage**: Asynchronous, chunked sequence writing into compressed or uncompressed NumPy arrays (`.npz`) using a dedicated `ChunkWriter`. Parallel metadata capture into `.meta.json` pairs.
- **Vectorized Data Collection**: High-throughput collection traversing multiple vectorized Gymnasium environments simultaneously.
- **Multiple Agent Modes**:
  - `random`: Samples random actions uniformly.
  - `heuristic`: Action distribution explicitly tuned for basic progression heuristics (favoring moving right, jumping, and sprinting).
  - `human`: Pygame-based GUI capturing keyboard inputs for human demonstration recording.
- **Three-Dimensional Dataset Balancing**: Opt-in system that equalizes world-stage and progression coverage (action-replay mid-level spawning) and action distribution (dynamic policy reweighting). See [Dataset Balancing System](#dataset-balancing-system).

## Project Structure

```text
├── data/
│   └── human_play/              # Default output directory for collected chunks
│       ├── chunk_*.npz          # Sequence data (frames, actions, dones, metadata)
│       ├── chunk_*.meta.json    # Per-chunk summaries (level, action, progression)
│       └── rollouts.jsonl       # Episode rollouts for replay-based progression balance
├── remote/
│   ├── config.sh                       # Single source of truth for host, port, and user
│   ├── connect.sh                      # Open SSH tunnel to remote instance
│   ├── setup_remote.sh                 # Sync code, install deps, create conda env
│   ├── send_data.sh                    # Sync local data/ to remote instance
│   ├── get_results.sh                  # Retrieve training results from remote instance
│   └── sync_code.sh                    # Sync code
├── scripts/
│   ├── collect_vector.py        # Main execution entrypoint for data collection
│   └── balance_report.py        # Dataset balance inspection tool
└── src/
    └── mario_world_model/
        ├── actions.py                  # Joypad action space definitions (COMPLEX_MOVEMENT)
        ├── config.py                   # Shared constants (SEQUENCE_LENGTH, IMAGE_SIZE, etc.)
        ├── coverage.py                 # Coverage scanning & balance computation (level/progression/action)
        ├── envs.py                     # Gym & Shimmy wrappers, RandomLevelMarioEnv (replay pool)
        ├── preprocess.py               # Frame padding logic (pad_to_square_256)
        ├── rollouts.py                 # Episode tracking, JSONL storage, RolloutIndex
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

Data is captured utilizing the script deployed in `scripts/collect_vector.py`. 

**Example: Collecting data via Human Play**
Play the game yourself utilizing a single environment.
```bash
python scripts/collect_vector.py \
    --mode human \
    --num-envs 1 \
    --total-steps 20000 \
    --sequences-per-chunk 512 \
    --output-dir data/human_play
```

### Analyzing Data

Verify the quality and diversity of your dataset using the analysis script. This tool reports action distribution, level coverage, and causal events (deaths/flag captures).

```bash
python scripts/analyze_data.py --data-dir data/human_play
```

Key metrics to watch:
1. **Action Distribution**: Ensure `NOOP` isn't dominating (aim for <20%) and that `Right`, `Run`, and `Jump` are well-represented.
2. **Level Coverage**: Check that the dataset covers multiple levels (not just 1-1) and that max x-positions indicate deep traversal, not just start-area loitering.
3. **Causal Events**: You want non-zero deaths and flag captures so the model learns consequences and goals.
*Keyboard keys: Arrow keys (or WASD) to move, `o` to jump (A), `p` to sprint (B).*

**Example: Collecting data via Vectorized Heuristic Bot**
Run 16 environments asynchronously capturing data using the heuristic policy. Writes compressed chunk files in the background to avoid stalling ticks.
```bash
python scripts/collect_vector.py \
    --mode heuristic \
    --level-mode random \
    --num-envs 16 \
    --total-steps 200000 \
    --sequences-per-chunk 512 \
    --output-dir data/heuristic_play
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

## Dataset Balancing System

Naïve data collection produces heavily skewed datasets — early levels dominate, most frames hover near spawn points, and certain actions (like NOOP) are massively overrepresented. The balancing system addresses two dimensions:

| Dimension | Problem | Solution |
|---|---|---|
| **Progression** | Frames cluster near level starts (low x_pos) and some levels are overrepresented | Deficit-proportional replay pool spanning all `(world, stage, x_bin)` tuples — naturally balances world-stage coverage as a side effect |
| **Action** | NOOP/Right dominate; rare combos underrepresented | Dynamic reweighting of the agent's action distribution |

Both are opt-in via CLI flags and build on a shared rebalance loop. Progression balancing inherently equalizes world-stage representation because every `(world, stage, x_bin)` triple is treated as an equal target — if a level has fewer bins covered, its existing bins get proportionally more weight.

### How It Works

#### 1. Episode Rollout Tracking

During collection, every completed episode is recorded to `<output_dir>/rollouts.jsonl` — one JSON line per episode containing the `world`, `stage`, full `actions` list, `x_positions` trajectory, `max_x`, and `outcome` (`"death"`, `"flag"`, or `"timeout"`). These rollouts serve as lightweight save-states: to reach a particular x-position within a level, the system replays the recorded action sequence from level start.

Key classes (in `src/mario_world_model/rollouts.py`):
- **`EpisodeTracker`**: Per-environment episode buffers. Call `record_step()` every tick, `finish_episode()` on done → yields a `Rollout`.
- **`RolloutWriter`**: Append-only JSONL output.
- **`RolloutIndex`**: Loads the JSONL into memory with fast lookups. `find_replay_actions(world, stage, target_x_bin)` returns the action sequence and step count to reach a given 256px screen bin.

#### 2. Progression Balance via Replay Pool

The system bins x-positions into 256px screens (one NES screen width). At each rebalance interval (default: every 5 chunks), it:

1. **Scans** all existing chunks to compute `{(world, stage, x_bin): frame_count}`.
2. **Computes** per-bin deficit — bins with fewer frames than the uniform target get higher weight.
3. **Builds a replay pool** — for each deficit bin, finds a rollout in `rollouts.jsonl` that reaches that position and adds a `(level, actions, target_step, weight)` entry. Bin-0 entries (level starts) are included as zero-step replays.
4. **Pushes the pool** to each `RandomLevelMarioEnv` via `update_progression_balance()`.

On every `reset()`, the environment samples a weighted entry from the replay pool, then **fast-forwards** the emulator by replaying the recorded actions inside `reset()` — transparent to gymnasium's auto-reset machinery. The returned observation is from the target mid-level position. If Mario dies during fast-forward, that entry is removed from the pool and one retry is attempted before falling back to a normal level start.

#### 3. Action Balance via Dynamic Reweighting

At each rebalance interval, the system scans chunk metadata for per-action frame counts, computes deficit-proportional weights, and pushes them to the `VectorActionPolicy`:
- **Random mode**: weighted random sampling replaces uniform selection during sticky-action choice.
- **Heuristic mode**: 50% of samples use the balance weights, 50% use the built-in heuristic distribution.
- **Human mode**: action balance is informational only (reported but not enforced).

#### 4. Chunk Metadata Fast-Path

Each `.meta.json` includes `action_summary` and `progression_summary` fields alongside the existing `level_summary`. Rebalance scans read these lightweight JSON files instead of decompressing full `.npz` arrays, keeping the rebalance loop fast even with hundreds of chunks.

### CLI Flags

```bash
python scripts/collect_vector.py \
    --mode human \
    --num-envs 1 \
    --total-steps 200000 \
    --sequences-per-chunk 512 \
    --async-write \
    --compress-chunks \
    --output-dir data/human_play \
    --balance \                  # Enable progression-aware balanced collection
    --balance-actions            # Enable action distribution rebalancing
```

| Flag | Effect |
|---|---|
| `--balance` | Progression-aware balanced collection via replay pool (equalizes world-stage and x-position coverage) |
| `--balance-actions` | Dynamic action distribution reweighting toward underrepresented actions (implies `--balance`) |
| `--rebalance-interval N` | Re-scan and recompute weights every N chunks (default: 5) |

### Balance Report

Inspect dataset balance at any time:

```bash
# World-stage balance only
python scripts/balance_report.py --data-dir data/human_play

# Include progression and action breakdowns
python scripts/balance_report.py --data-dir data/human_play --progression --actions

# Show only the 10 most-deficient entries
python scripts/balance_report.py --data-dir data/human_play --progression --actions --top 10
```

### Architecture

```text
collect_vector.py
  ├── EpisodeTracker ──→ RolloutWriter ──→ rollouts.jsonl
  │
  ├── [rebalance interval]
  │     ├── scan_progression_coverage()  → replay pool     → env.update_progression_balance()
  │     └── scan_action_coverage()       → action weights  → policy.update_action_weights()
  │
  └── RandomLevelMarioEnv.reset()
        ├── Sample from replay pool (weighted by per-bin deficit)
        ├── Bin 0 → level start (no fast-forward needed)
        ├── Bin N → fast-forward: replay actions[:target_step]
        │     └── On death → remove entry, retry once → fallback to level start
        └── Return obs from target position
```

### Design Decisions

- **JSONL rollout format**: Append-friendly, one-line-per-episode, trivially parseable. Loaded into memory for rebalancing — rollout files stay small (action sequences only, no frames).
- **Replay inside `reset()`**: Keeps fast-forward transparent to gymnasium's `SyncVectorEnv`/`AsyncVectorEnv` auto-reset. No modifications needed to the outer collection loop.
- **Discard replay frames**: Fast-forward is overhead only. Fresh data is collected starting from the target position.
- **256px bins**: One NES screen width. Matches the natural granularity of level design (screen-by-screen scrolling).
- **No separate world-stage balancing**: Progression balance treats every `(world, stage, x_bin)` as an equal target. Since bin-0 entries (level starts) are in the pool alongside mid-level bins, world-stage coverage is inherently equalized without a separate weighting layer.
- **Retry on death**: If a replay fails (Mario dies during fast-forward), the faulty entry is pruned from the pool and one retry with a fresh sample is attempted before falling back to a normal reset.

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

### Capacity Planning for Full-Dataset Training

Overfitting 1–2 samples confirms optimization and implementation correctness, but does **not** imply sufficient model capacity for large-scale generalization. As dataset size and diversity increase, reconstruction quality usually drops unless capacity and/or token budget is scaled appropriately.

**Recommended scaling order (highest impact first):**
1. Increase `init_dim` (e.g. 32 → 64).
2. Add residual depth (extra `residual` blocks around `compress_space` stages).
3. Add selective attention (`attend_space`, `attend_time`, or `gateloop_time`) near bottlenecks.
4. Increase codebook size (e.g. 256 → 512/1024) once encoder-decoder capacity is no longer the bottleneck.

**How to interpret training signals:**
- **Train recon high + Val recon high**: likely under-capacity (or optimization constraints) → increase architecture capacity.
- **Train recon low + Val recon high**: likely overfitting or data split/domain issues → improve data/regularization before scaling params.
- **Good PSNR but blurry textures**: codebook likely too small or quantization too coarse → increase codebook size.
- **Temporal artifacts/flicker**: add temporal modeling blocks (`attend_time` / `gateloop_time`) rather than only widening spatial conv stacks.

**Practical staged recipe:**
- **Stage A**: keep layer topology; increase `init_dim` to 64.
- **Stage B**: add one extra `residual` block after each `compress_space`.
- **Stage C**: add one bottleneck `attend_space` and one `attend_time` block.
- Move to the next stage only if both train and validation reconstruction metrics plateau for multiple epochs.


