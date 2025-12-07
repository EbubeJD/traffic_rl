"""
Quick diagnostics for transitions.csv to assess data quality for BC/CQL.

Usage:
    python tools/diagnose_transitions.py --transitions outputs/.../transitions.csv

Reports:
- Number of rows and observation dimension
- Action distribution
- Reward stats
- Per-feature mean/max and percent of zeros
"""

import argparse
import csv
import json
import numpy as np


def load_transitions(path):
    obs_list = []
    actions = []
    rewards = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                obs = np.array(json.loads(row["obs_json"]), dtype=np.float32)
                action = int(row["action"])
                reward = float(row["reward"])
            except Exception:
                continue
            obs_list.append(obs)
            actions.append(action)
            rewards.append(reward)
    return np.stack(obs_list, axis=0) if obs_list else np.zeros((0,)), np.array(actions), np.array(rewards)


def main():
    parser = argparse.ArgumentParser(description="Diagnose transitions.csv content")
    parser.add_argument("--transitions", required=True, help="Path to transitions.csv")
    args = parser.parse_args()

    obs_arr, actions, rewards = load_transitions(args.transitions)
    if obs_arr.size == 0:
        print(f"No transitions loaded from {args.transitions}")
        return

    n, d = obs_arr.shape
    print(f"Loaded {n} transitions with obs_dim={d}")

    # Action distribution
    bincount = np.bincount(actions, minlength=actions.max() + 1 if actions.size else 0)
    print("Action counts:", dict(enumerate(bincount.tolist())))

    # Reward stats
    print(f"Reward mean={rewards.mean():.3f} min={rewards.min():.3f} max={rewards.max():.3f}")

    # Feature stats
    nonzero = (obs_arr != 0).sum(axis=0)
    percent_nz = nonzero / n * 100.0
    means = obs_arr.mean(axis=0)
    maxs = obs_arr.max(axis=0)
    print("Per-feature stats (idx: mean | max | %nonzero):")
    for i in range(d):
        print(f"  [{i:02d}] {means[i]:.4f} | {maxs[i]:.4f} | {percent_nz[i]:5.1f}%")


if __name__ == "__main__":
    main()
