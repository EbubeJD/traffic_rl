"""
Capture tick logs for offline training.

This script runs the live CARLA pipeline (TrafficRunner + TLObserver) long enough
to accumulate ticks.csv for the configured CONTROL_TL_IDS. It does not save
images (keep SAVE_EVERY_N=0) and keeps settings lean for faster capture.

Usage:
    python tools/capture_ticks.py --sim-hours 3
    # or shorter smoke:
    python tools/capture_ticks.py --sim-minutes 30
"""

import argparse
import time
import sys
import os

# Ensure repo root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from traffic_rl.env.traffic_runner import TrafficRunner
from config import DT, CONTROL_TL_IDS


def main():
    parser = argparse.ArgumentParser(description="Capture ticks.csv for offline replay.")
    parser.add_argument("--sim-hours", type=float, default=3.0, help="Simulated hours to run.")
    parser.add_argument("--sim-minutes", type=float, default=None, help="Alternative: simulated minutes to run.")
    args = parser.parse_args()

    sim_seconds = (args.sim_minutes * 60.0) if args.sim_minutes is not None else (args.sim_hours * 3600.0)

    print(f"[CAPTURE] Target simulated duration: {sim_seconds/3600.0:.2f} h "
          f"({sim_seconds:.0f} s) for CONTROL_TL_IDS={CONTROL_TL_IDS}")

    runner = TrafficRunner()
    if not runner.is_initialized:
        runner.initialize()

    start_wall = time.time()
    sim_time = 0.0
    tick = 0

    try:
        while sim_time < sim_seconds:
            runner.step()  # synchronous tick; TLObserver writes ticks.csv
            tick += 1
            sim_time += DT

            if tick % 500 == 0:
                elapsed = time.time() - start_wall
                print(f"[CAPTURE] tick={tick} sim_time={sim_time/3600.0:.3f} h wall={elapsed/3600.0:.3f} h")
    except KeyboardInterrupt:
        print("[CAPTURE] Interrupted by user.")
    finally:
        runner.close()
        elapsed = time.time() - start_wall
        print(f"[CAPTURE] Done. Total ticks={tick}, sim_time={sim_time/3600.0:.2f} h, wall_time={elapsed/3600.0:.2f} h")


if __name__ == "__main__":
    main()
