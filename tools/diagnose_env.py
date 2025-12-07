"""
Environment diagnostic tool for validating CARLA ROI/observers.

Runs TrafficEnv with debug mode to verify:
1. ROI polygon exists and loads correctly
2. Observer/camera spawns successfully
3. Metrics (queue, wait, arrivals) are non-zero

Usage:
    python tools/diagnose_env.py --steps 100 --debug
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
import numpy as np
from pathlib import Path

from traffic_rl.env.traffic_runner import TrafficRunner
from traffic_rl.env.traffic_env import TrafficEnv


def check_roi_exists(tl_dir):
    """Verify roi.json and stopline.json exist and are valid."""
    print("\n" + "="*80)
    print("[DIAGNOSTIC] ROI Validation")
    print("="*80)

    tl_path = Path(tl_dir)
    roi_file = tl_path / "roi.json"
    stopline_file = tl_path / "stopline.json"

    # Check ROI
    if roi_file.exists():
        try:
            with open(roi_file, 'r') as f:
                roi_data = json.load(f)
            polygon = roi_data.get("polygon", [])
            print(f"  ✓ ROI file: {roi_file}")
            print(f"  ✓ ROI polygon: {len(polygon)} vertices")
            if polygon:
                print(f"    First vertex: {polygon[0]}")
                print(f"    Last vertex: {polygon[-1]}")
        except Exception as e:
            print(f"  ✗ ROI file exists but failed to parse: {e}")
            return False
    else:
        print(f"  ✗ ROI file not found: {roi_file}")
        return False

    # Check stopline
    if stopline_file.exists():
        try:
            with open(stopline_file, 'r') as f:
                stopline_data = json.load(f)
            print(f"  ✓ Stopline file: {stopline_file}")
            print(f"    Stopline polygon: {len(stopline_data.get('polygon', []))} vertices")
        except Exception as e:
            print(f"  ✗ Stopline file exists but failed to parse: {e}")
    else:
        print(f"  ⚠ Stopline file not found (may be derived from ROI)")

    return True


def run_debug_simulation(steps=100, debug_obs=False):
    """Run TrafficEnv with debug mode, collect metrics."""
    print("\n" + "="*80)
    print("[DIAGNOSTIC] Running Debug Simulation")
    print("="*80)

    # Initialize environment
    print(f"  Initializing TrafficRunner...")
    runner = TrafficRunner()

    print(f"  Initializing TrafficEnv (debug_obs={debug_obs})...")
    env = TrafficEnv(runner, episode_steps=steps, debug_obs=debug_obs)

    print(f"  Resetting environment...")
    obs = env.reset()

    # Check observers
    print("\n" + "="*80)
    print("[DIAGNOSTIC] Observer Health")
    print("="*80)

    if env.runner.observers:
        obs_obj = env.runner.observers[0]
        print(f"  ✓ Observer found: {obs_obj.stable_id}")
        print(f"  ✓ Save directory: {obs_obj.save_dir}")

        # Check YOLO model
        try:
            from ultralytics import YOLO
            from traffic_rl.config import YOLO_MODEL_PATH
            if os.path.exists(YOLO_MODEL_PATH):
                print(f"  ✓ YOLO model: {YOLO_MODEL_PATH} loaded")
            else:
                print(f"  ⚠ YOLO model not found: {YOLO_MODEL_PATH}")
        except Exception as e:
            print(f"  ⚠ YOLO check failed: {e}")
    else:
        print(f"  ✗ No observers found!")
        return None, None, None, None

    # Run simulation and collect metrics
    print(f"\n  Running {steps} steps...")

    queues = []
    waits = []
    long_waits = []
    arrivals = 0
    crossings = 0

    for step in range(steps):
        action = env.action_space.sample()  # Random policy
        obs, reward, done, info = env.step(action)

        # Collect metrics from info
        queues.append(info.get("queue_count", 0.0))
        waits.append(info.get("avg_wait", 0.0))
        long_waits.append(info.get("num_long_wait_60s", 0.0))

        if (step + 1) % 20 == 0:
            print(f"    Step {step+1}/{steps}: queue={info.get('queue_count', 0):.1f}, "
                  f"wait={info.get('avg_wait', 0):.1f}s, phase={info.get('phase')}")

        if done:
            print(f"  Episode finished at step {step+1}")
            break

    # Read crossings from crossings.csv if available
    try:
        save_dir = env.runner.observers[0].save_dir
        crossings_file = os.path.join(save_dir, "crossings.csv")
        if os.path.exists(crossings_file):
            with open(crossings_file, 'r') as f:
                crossings = sum(1 for line in f) - 1  # Subtract header
    except Exception:
        pass

    env.close()

    return queues, waits, long_waits, crossings


def analyze_metrics(queues, waits, long_waits, crossings):
    """Compute min/max/mean and flag if all zeros."""
    print("\n" + "="*80)
    print("[DIAGNOSTIC] Detection Metrics")
    print("="*80)

    if queues is None:
        print("  ✗ No metrics collected (simulation failed)")
        return

    queues = np.array(queues)
    waits = np.array(waits)
    long_waits = np.array(long_waits)

    # Queue stats
    q_min, q_max, q_mean = queues.min(), queues.max(), queues.mean()
    print(f"  Queue: min={q_min:.1f}, max={q_max:.1f}, mean={q_mean:.2f}")
    if q_max > 0:
        print(f"    ✓ NON-ZERO (queue detected)")
    else:
        print(f"    ⚠ ALL ZEROS (no queue detected - check traffic spawning)")

    # Wait stats
    w_min, w_max, w_mean = waits.min(), waits.max(), waits.mean()
    print(f"  Avg Wait: min={w_min:.1f}s, max={w_max:.1f}s, mean={w_mean:.2f}s")
    if w_max > 0:
        print(f"    ✓ NON-ZERO (wait times recorded)")
    else:
        print(f"    ⚠ ALL ZEROS (no wait times)")

    # Long waiters
    lw_max, lw_mean = long_waits.max(), long_waits.mean()
    print(f"  Long Waiters (≥60s): max={lw_max:.0f}, mean={lw_mean:.2f}")

    # Crossings
    if crossings is not None:
        print(f"  Vehicle Crossings: {crossings} vehicles")

    # Overall assessment
    print("\n" + "="*80)
    print("[DIAGNOSTIC] Overall Assessment")
    print("="*80)

    if q_max > 0 and w_max > 0:
        print("  ✓ PASS: Environment is capturing traffic metrics correctly")
        print("  → Proceed with data analysis and pipeline improvements")
    elif q_max == 0:
        print("  ⚠ WARNING: No queue detected")
        print("  → Possible causes:")
        print("    1. Insufficient traffic (increase NUM_AUTOPILOT in config.py)")
        print("    2. ROI polygon misaligned (use tools/verify_roi.py)")
        print("    3. YOLO detection failing (check models/yolo11n.pt)")
    else:
        print("  ⚠ PARTIAL: Some metrics working, others zero")
        print("  → Review observer/logger implementation")


def main():
    parser = argparse.ArgumentParser(description="Diagnose TrafficEnv environment")
    parser.add_argument("--steps", type=int, default=100, help="Number of simulation steps")
    parser.add_argument("--debug", action="store_true", help="Enable debug observation printing")
    parser.add_argument("--tl_dir", type=str, default=None,
                        help="Traffic light directory to check ROI (optional)")
    args = parser.parse_args()

    print("="*80)
    print("TRAFFIC ENV DIAGNOSTIC TOOL")
    print("="*80)

    # Check ROI if directory provided
    if args.tl_dir:
        if not check_roi_exists(args.tl_dir):
            print("\n⚠ ROI validation failed - environment may not work correctly")

    # Run simulation
    queues, waits, long_waits, crossings = run_debug_simulation(
        steps=args.steps,
        debug_obs=args.debug
    )

    # Analyze results
    if queues is not None:
        analyze_metrics(queues, waits, long_waits, crossings)

    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
