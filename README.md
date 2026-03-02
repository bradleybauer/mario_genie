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
├── scripts/
│   └── collect_phase1_vector.py        # Main execution entrypoint for data collection
└── src/
    └── mario_world_model_phase1/
        ├── actions.py                  # Joypad action space definitions (COMPLEX_MOVEMENT)
        ├── envs.py                     # Gym & Shimmy wrappers, RandomLevelMarioEnv
        ├── preprocess.py               # Frame padding logic (pad_to_square_256)
        └── storage.py                  # Asynchronous chunk writing (ChunkWriter)
```

## Setup and Usage

### Prerequisites
The codebase relies on Python 3 with dependencies for environment emulation, array manipulation, and GUI playback, notably `gymnasium`, `pygame`, `numpy`, `shimmy`, and `gym_super_mario_bros`.

### Collecting Data

Data is captured utilizing the script deployed in `scripts/collect_phase1_vector.py`. 

**Example: Collecting data via Human Play**
Play the game yourself utilizing a single environment.
```bash
python scripts/collect_phase1_vector.py \
    --mode human \
    --num-envs 1 \
    --total-steps 20000 \
    --sequence-length 16 \
    --sequences-per-chunk 512 \
    --output-dir data/phase1/human_play
```
*Keyboard keys: Arrow keys (or WASD) to move, `o` to jump (A), `p` to sprint (B).*

**Example: Collecting data via Vectorized Heuristic Bot**
Run 16 environments asynchronously capturing data using the heuristic policy. Writes compressed chunk files in the background to avoid stalling ticks.
```bash
python scripts/collect_phase1_vector.py \
    --mode heuristic \
    --level-mode random \
    --num-envs 16 \
    --total-steps 200000 \
    --sequence-length 16 \
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
