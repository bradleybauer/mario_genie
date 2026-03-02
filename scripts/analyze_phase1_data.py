#!/usr/bin/env python3
"""
scripts/analyze_phase1_data.py

Analyzes collected Mario gameplay data for:
1. Action Distribution (Diversity)
2. State Space Coverage (Level Diversity)
3. Causal Diversity (Deaths/Failures)

Usage:
    python scripts/analyze_phase1_data.py --data-dir data/phase1/human_play
"""

import argparse
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

# Add src to path to import project modules
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from mario_world_model_phase1.actions import get_action_meanings
except ImportError:
    print("Warning: Could not import mario_world_model_phase1.actions. Using default mappings.")
    get_action_meanings = None

def analyze_dataset(data_dir: Path):
    print(f"Analyzing data in: {data_dir}")
    
    chunk_files = sorted(list(data_dir.glob("chunk_*.npz")))
    if not chunk_files:
        print("No chunk_*.npz files found.")
        return

    total_frames = 0
    total_sequences = 0
    
    # 1. Action Distribution
    action_counts = Counter()
    
    # 2. Level Diversity
    level_counts = Counter() # (world, stage) tuples
    
    # 2b. X-Position Diversity (Coverage within level)
    # Store list of x_positions for each (world, stage)
    # Using a list might be memory intensive for huge datasets, 
    # so we'll store min, max, and a histogram/binned count.
    # Level ID -> [min_x, max_x, sum_x, count, set_of_100px_bins]
    level_x_stats = {} 

    # 3. Causal Diversity (Deaths)
    # We will track life changes.
    total_deaths = 0
    total_flag_get = 0
    
    # We can also track average episode length or similar if we had full episodes, 
    # but we have chunks of sequences. We can check 'dones'.
    total_dones = 0

    # Get Action Meanings
    if get_action_meanings:
        raw_meanings = get_action_meanings()
        ACTION_NAMES = {}
        for idx, meaning_list in enumerate(raw_meanings):
            ACTION_NAMES[idx] = "+".join(meaning_list) if meaning_list else "NOOP"
    else:
        raise RuntimeError("Action meanings not available. Please ensure mario_world_model_phase1 is installed and accessible.")

    print(f"Found {len(chunk_files)} chunks. Processing...")

    for cf in chunk_files:
        try:
            with np.load(cf) as data:
                # Load necessary arrays
                # shapes: 
                # actions: (B, T)
                # world: (B, T)
                # stage: (B, T)
                # life: (B, T)
                # flag_get: (B, T)
                # dones: (B, T)
                # x_pos: (B, T)

                actions = data['actions']
                world = data['world']
                stage = data['stage']
                life = data['life']
                flag_get = data['flag_get']
                dones = data['dones']
                # Check if x_pos exists (it should based on collection script)
                if 'x_pos' in data:
                    x_pos = data['x_pos']
                else:
                    x_pos = np.zeros_like(actions)

                # Update totals
                total_sequences += actions.shape[0]
                total_frames += actions.size
                
                # 1. Action Counts
                # Flatten and count
                unique, counts = np.unique(actions, return_counts=True)
                for u, c in zip(unique, counts):
                    action_counts[u] += c
                
                # 2. Level Diversity
                # We can sample one frame per sequence or count all frames. 
                # Counting all frames gives 'time spent' in each level.
                # structure: world and stage are arrays of same shape as actions
                # We can combine them into (world, stage) pairs
                # This might be slow for massive datasets, but fine for now.
                # Optimization: just take the distinct values from each sequence?
                # Let's count frames for accurate "coverage" metric.
                
                # Zip world and stage to count pairs
                # Flattening
                w_flat = world.flatten()
                s_flat = stage.flatten()
                
                # Fast way to count pairs:
                # world * 100 + stage (assuming stage < 100)
                level_codes = w_flat * 100 + s_flat
                unique_levels, u_counts = np.unique(level_codes, return_counts=True)
                
                for code, c in zip(unique_levels, u_counts):
                    w = code // 100
                    s = code % 100
                    level_counts[(w, s)] += c

                # 2b. X-Position Analysis
                x_flat = x_pos.flatten()
                
                # Iterate through unique levels found in this chunk
                for code in unique_levels:
                    w = code // 100
                    s = code % 100
                    mask = (level_codes == code)
                    xs_in_level = x_flat[mask]
                    
                    if xs_in_level.size > 0:
                        l_min = np.min(xs_in_level)
                        l_max = np.max(xs_in_level)
                        
                        # Binning
                        screens = set((xs_in_level // 256).tolist())
                        
                        if (w, s) not in level_x_stats:
                            level_x_stats[(w, s)] = {
                                'min': int(l_min),
                                'max': int(l_max),
                                'screens': screens,
                                'count': int(xs_in_level.size),
                                'deaths': 0,
                                'flags': 0, 
                                'coins': 0
                            }
                        else:
                            st = level_x_stats[(w, s)]
                            st['min'] = min(st['min'], int(l_min))
                            st['max'] = max(st['max'], int(l_max))
                            st['screens'].update(screens)
                            st['count'] += int(xs_in_level.size)

                # 3. Events per Level
                # We need to attribute deaths/flags to specific levels.
                # life slope analysis again.
                
                # Check where life decreases
                life_diff = life[:, 1:] - life[:, :-1]
                # Indices (b, t) where life dropped
                death_indices = np.where(life_diff < 0)
                
                if len(death_indices[0]) > 0:
                    for b, t in zip(death_indices[0], death_indices[1]):
                        # Get world/stage at this exact moment
                        # Note: life_diff is size T-1. index t maps to transition from t to t+1
                        final_w = world[b, t]
                        final_s = stage[b, t]
                        
                        # Initialize if not present (though X-stat loop usually catches it)
                        if (final_w, final_s) not in level_x_stats:
                             level_x_stats[(final_w, final_s)] = {'min':0, 'max':0, 'screens':set(), 'count':0, 'deaths':0, 'flags':0, 'coins':0}
                        
                        level_x_stats[(final_w, final_s)]['deaths'] += 1
                        total_deaths += 1

                # Flag Check (Transition 0->1 or just present?)
                # We only want to count *instances* of getting a flag, not every frame.
                # Detect rising edge of flag_get
                # flag_get shape: (B, T)
                # diff: (B, T-1)
                flag_diff = flag_get[:, 1:] - flag_get[:, :-1]
                flag_indices = np.where(flag_diff > 0)
                
                if len(flag_indices[0]) > 0:
                    for b, t in zip(flag_indices[0], flag_indices[1]):
                        final_w = world[b, t+1] # t+1 is where flag became 1
                        final_s = stage[b, t+1]
                        
                        if (final_w, final_s) not in level_x_stats:
                             level_x_stats[(final_w, final_s)] = {'min':0, 'max':0, 'screens':set(), 'count':0, 'deaths':0, 'flags':0, 'coins':0}
                        
                        level_x_stats[(final_w, final_s)]['flags'] += 1
                        total_flag_get += 1

                # 3. Deaths (Global Counter - kept for legacy print)
                # (We already updated total_deaths above)

                # We need to being careful across sequence boundaries because sequences might not be contiguous in time 
                # (collected from parallel envs then shuffled? No, `ChunkWriter` receives `add_sequence` from one env at a time).
                # But `ChunkWriter` just stacks them.
                # In `collect_phase1_vector.py`, `seq_frames` accumulates `sequence_length` frames then adds to writer.
                # So inside one row of (B, T), the frames are contiguous.
                # But row `i` and row `i+1` in the chunk? 
                # `collect_phase1_vector` calls `writer.add_sequence`. 
                # The writer accumulates them. They come from potentially different envs if running parallel.
                # So we can only trust continuity WITHIN a sequence (axis 1).
                
                # Calculate difference along time axis
                # life shape: (B, T)
                # diff along axis 1
                life_diff = life[:, 1:] - life[:, :-1]
                # Deaths: life went down (diff < 0)
                # Note: valid ranges of life are usually 1-3. 
                # We assume if life drops, it's a death.
                num_deaths = np.sum(life_diff < 0)
                total_deaths += num_deaths
                
                # Flag Get
                # Since flag_get is a boolean/int 0-1, we can sum it? 
                # Or checks transition 0 -> 1?
                # Usually flag_get is 1 when the flag represents acquired. 
                # It might stay 1 for the rest of the level end animation.
                # We 'collected' the flag if it was 0 then became 1, OR if we just see a 1 frame.
                # Let's count frames where flag_get is true? 
                # Or count sequences where flag_get > 0.
                
                # Let's just sum frames with flag_get > 0 to see how much "win state" data we have
                total_flag_get += np.sum(flag_get > 0)
                
                total_dones += np.sum(dones)

        except Exception as e:
            print(f"Error reading {cf}: {e}")

    # --- Reporting ---
    print("\n" + "="*40)
    print(f"ANALYSIS REPORT")
    print("="*40)
    print(f"Total Frames Analyzed: {total_frames}")
    print(f"Total Sequences: {total_sequences}")
    print("-" * 20)

    # 1. Action Distribution
    print("1. ACTION DISTRIBUTION")
    print(f"{'Action ID':<10} {'Name':<15} {'Count':<10} {'Percentage':<10}")
    print("-" * 50)
    sorted_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)
    for act_id, count in sorted_actions:
        pct = (count / total_frames) * 100
        name = ACTION_NAMES.get(act_id, f"Unknown-{act_id}")
        warning = " (!)" if pct < 5.0 else ""
        print(f"{act_id:<10} {name:<15} {count:<10} {pct:<6.2f}%{warning}")

    print("-" * 20)
    
    # 2. Level Coverage
    print("2. LEVEL COVERAGE (Top 10)")
    sorted_levels = sorted(level_counts.items(), key=lambda x: x[1], reverse=True)
    for (w, s), count in sorted_levels[:10]:
        pct = (count / total_frames) * 100
        print(f"World {w}-{s}: {count} frames ({pct:.2f}%)")
    
    unique_worlds = set(w for w, s in level_counts.keys())
    print(f"\nTotal Unique Levels Visited: {len(level_counts)}")
    print(f"Total Unique Worlds Visited: {len(unique_worlds)}")
    
    print("-" * 20)
    
    # 2b. X-Diversity Details
    print("3. DETAILED LEVEL PROGRESSION (Geography & Events)")
    print(f"{'Level':<8} {'Frames':<8} {'Max X':<6} {'Screens':<8} {'Cov':<5} {'Deaths':<8} {'Flags':<8}")
    print("-" * 80)
    
    # reuse sorted_levels from above
    for (w, s), count in sorted_levels[:20]: # Show top 20
        stats = level_x_stats.get((w,s))
        if stats:
            n_screens = len(stats['screens'])
            coverage_msg = ""
            if stats['max'] < 300:
                coverage_msg = "Bad"
            elif n_screens < 3:
                 coverage_msg = "Low"
            else:
                coverage_msg = "Ok"
                
            print(f"W {w}-{s:<3} {count:<8} {stats['max']:<6} {n_screens:<8} {coverage_msg:<5} {stats['deaths']:<8} {stats['flags']:<8}")
    
    print("-" * 20)

    # 3. Causal Events
    print(f"4. CAUSAL EVENTS SUMMARY")
    print(f"Total Deaths (Life lost): {total_deaths}")
    if total_sequences > 0:
        print(f"  Avg Deaths per Sequence: {total_deaths / total_sequences:.4f}")
    
    print(f"Frames with Flag Get: {total_flag_get}")
    print(f"Episode terminations (Dones): {total_dones}")
    
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/phase1/human_play"))
    args = parser.parse_args()
    
    if not args.data_dir.exists():
        print(f"Error: Directory {args.data_dir} does not exist.")
    else:
        analyze_dataset(args.data_dir)
