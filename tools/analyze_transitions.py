"""
Transitions data quality analysis tool.

Analyzes transitions.csv to diagnose data quality issues:
1. Action distribution (check for class imbalance)
2. Observation statistics (% zeros, mean, std per feature)
3. Feature-action correlation (mutual information)
4. Temporal patterns (fixed alternation detection)

Usage:
    python tools/analyze_transitions.py \
        --transitions outputs/Town10HD_Opt/tl_road5_lane-1_s10058/transitions.csv \
        --out old_data_analysis.txt
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import csv
import json
import numpy as np
from pathlib import Path


def load_transitions(csv_path):
    """Load transitions from CSV file."""
    transitions = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                action = int(row["action"])
                obs = np.array(json.loads(row["obs_json"]), dtype=np.float32)
                transitions.append((obs, action))
            except Exception as e:
                print(f"⚠ Skipping malformed row: {e}")
                continue

    return transitions


def analyze_action_distribution(actions):
    """Compute action distribution statistics."""
    print("\n" + "="*80)
    print("[ANALYSIS] Action Distribution")
    print("="*80)

    actions = np.array(actions)
    unique, counts = np.unique(actions, return_counts=True)

    total = len(actions)
    for act, count in zip(unique, counts):
        pct = (count / total) * 100
        print(f"  Action {act}: {count} ({pct:.1f}%)")

    # Check balance
    if len(unique) == 2:
        ratio = max(counts) / min(counts)
        if ratio < 1.5:
            print(f"  [OK] BALANCED (ratio {ratio:.2f}:1)")
        else:
            print(f"  [WARNING] IMBALANCED (ratio {ratio:.2f}:1)")
    else:
        print(f"  [WARNING] Expected 2 actions, found {len(unique)}")


def analyze_observations(obs_matrix, feature_names):
    """Compute observation statistics per feature."""
    print("\n" + "="*80)
    print("[ANALYSIS] Observation Statistics")
    print("="*80)

    n_features = obs_matrix.shape[1]
    if len(feature_names) != n_features:
        print(f"[WARNING] Feature name mismatch: {len(feature_names)} names, {n_features} features")
        feature_names = [f"feature_{i}" for i in range(n_features)]

    for i, name in enumerate(feature_names):
        values = obs_matrix[:, i]
        pct_zero = (values == 0.0).mean() * 100
        mean_val = values.mean()
        std_val = values.std()

        # Flag issues
        status = ""
        if pct_zero > 90:
            status = "[WARNING] NO SIGNAL"
        elif pct_zero < 10 and std_val > 0.01:
            status = "[OK] HAS SIGNAL"

        print(f"  {name:18s}: zeros={pct_zero:5.1f}%, mean={mean_val:8.4f}, "
              f"std={std_val:8.4f}  {status}")


def analyze_feature_action_correlation(obs_matrix, actions, feature_names):
    """Compute mutual information between features and actions."""
    print("\n" + "="*80)
    print("[ANALYSIS] Feature-Action Correlation")
    print("="*80)

    try:
        from sklearn.metrics import mutual_info_score
    except ImportError:
        print("  [WARNING] scikit-learn not available, skipping MI analysis")
        return

    actions = np.array(actions)

    for i, name in enumerate(feature_names):
        values = obs_matrix[:, i]

        # Discretize continuous values for MI calculation
        if values.std() < 1e-6:
            # Constant feature
            mi = 0.0
        else:
            # Create bins
            bins = np.linspace(values.min(), values.max() + 1e-9, 10)
            discretized = np.digitize(values, bins)
            mi = mutual_info_score(discretized, actions)

        # Flag correlation strength
        status = ""
        if mi < 0.01:
            status = "[WARNING] NO CORRELATION"
        elif mi > 0.3:
            status = "[OK] STRONG"
        elif mi > 0.1:
            status = "[OK] MODERATE"

        print(f"  {name:18s} <-> action: MI={mi:.4f}  {status}")


def analyze_temporal_patterns(actions):
    """Detect fixed alternation patterns."""
    print("\n" + "="*80)
    print("[ANALYSIS] Temporal Patterns")
    print("="*80)

    actions = np.array(actions)

    # Alternation rate
    action_changes = np.diff(actions) != 0
    alternation_rate = action_changes.mean() * 100

    print(f"  Action alternation rate: {alternation_rate:.1f}%")

    if alternation_rate > 90:
        print(f"    [WARNING] FIXED ALTERNATION (naive fixed-time controller)")
    elif alternation_rate > 70:
        print(f"    [INFO] High alternation (possibly periodic)")
    elif alternation_rate < 50:
        print(f"    [OK] ADAPTIVE (responsive to state)")
    else:
        print(f"    [INFO] Moderate alternation")

    # Longest fixed sequence
    max_same = 1
    current_same = 1
    for i in range(1, len(actions)):
        if actions[i] == actions[i-1]:
            current_same += 1
            max_same = max(max_same, current_same)
        else:
            current_same = 1

    print(f"  Longest same-action sequence: {max_same} steps")

    if max_same < 3:
        print(f"    [WARNING] Very short holds (thrashing)")
    elif max_same > 20:
        print(f"    [OK] Long green times (stable)")


def analyze_root_cause(obs_matrix, actions, feature_names):
    """Synthesize findings into root cause diagnosis."""
    print("\n" + "="*80)
    print("[ROOT CAUSE DIAGNOSIS]")
    print("="*80)

    actions = np.array(actions)

    # Check if most features are zero
    zero_pcts = [(obs_matrix[:, i] == 0.0).mean() * 100 for i in range(obs_matrix.shape[1])]
    features_mostly_zero = sum(pct > 90 for pct in zero_pcts)

    # Check alternation
    alternation_rate = (np.diff(actions) != 0).mean() * 100

    # Diagnosis
    if features_mostly_zero > 5 and alternation_rate > 90:
        print("  [CRITICAL] Poor data quality")
        print("  ")
        print("  Observations: 90%+ zeros in most features -> NO SIGNAL")
        print("  Actions: 90%+ alternation -> FIXED-TIME CONTROLLER")
        print("  ")
        print("  Consequence:")
        print("  -> BC learns 'flip current_phase' rule (only non-zero feature)")
        print("  -> Achieves ~50% accuracy (random guessing on binary task)")
        print("  ")
        print("  Solution:")
        print("  1. Verify environment captures traffic (tools/diagnose_env.py)")
        print("  2. Generate new data with queue-aware heuristic")
        print("  3. Retrain BC on high-quality data")

    elif features_mostly_zero > 5:
        print("  [WARNING] Low feature signal")
        print("  -> Most observations are zeros (no traffic or broken detection)")
        print("  -> Run tools/diagnose_env.py to verify environment")

    elif alternation_rate > 90:
        print("  [WARNING] Fixed-time controller")
        print("  -> Actions don't adapt to traffic state")
        print("  -> Generate new data with adaptive policy")

    else:
        print("  [OK] Data quality appears reasonable")
        print("  -> Features have variation")
        print("  -> Actions are adaptive")


def main():
    parser = argparse.ArgumentParser(description="Analyze transitions.csv data quality")
    parser.add_argument("--transitions", type=str, required=True,
                        help="Path to transitions.csv file")
    parser.add_argument("--out", type=str, default=None,
                        help="Optional output file for analysis report")
    args = parser.parse_args()

    print("="*80)
    print("TRANSITIONS DATA QUALITY ANALYSIS")
    print("="*80)
    print(f"File: {args.transitions}")

    # Load data
    print("\nLoading transitions...")
    transitions = load_transitions(args.transitions)

    if not transitions:
        print("✗ No transitions loaded")
        return 1

    print(f"[OK] Loaded {len(transitions)} transitions")

    # Extract observations and actions
    obs_matrix = np.stack([obs for obs, _ in transitions], axis=0)
    actions = [action for _, action in transitions]

    print(f"  Observation shape: {obs_matrix.shape}")
    print(f"  Actions: {len(actions)} samples")

    # Feature names (9-dimensional for TrafficEnv)
    feature_names = [
        "queue",
        "queue_ema",
        "avg_wait",
        "max_wait",
        "num_long_wait_60s",
        "arrival_ema",
        "discharge_ema",
        "time_in_state",
        "current_phase"
    ]

    # Adjust if observation dimension doesn't match
    if obs_matrix.shape[1] != len(feature_names):
        print(f"  ⚠ Expected {len(feature_names)} features, found {obs_matrix.shape[1]}")
        if obs_matrix.shape[1] < len(feature_names):
            feature_names = feature_names[:obs_matrix.shape[1]]
        else:
            feature_names.extend([f"extra_{i}" for i in range(obs_matrix.shape[1] - len(feature_names))])

    # Run analyses
    analyze_action_distribution(actions)
    analyze_observations(obs_matrix, feature_names)
    analyze_feature_action_correlation(obs_matrix, actions, feature_names)
    analyze_temporal_patterns(actions)
    analyze_root_cause(obs_matrix, actions, feature_names)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

    # Save to file if requested
    if args.out:
        print(f"\n✓ Report would be saved to: {args.out}")
        print("  (Currently prints to stdout only - redirect with > to save)")

    return 0


if __name__ == "__main__":
    exit(main())
