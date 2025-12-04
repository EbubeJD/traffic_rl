"""
Quick random-policy sanity test for the single-lane TrafficEnv (road5_lane-1).

Run steps:
1) Start CARLA on localhost:2000 with Town10HD_Opt loaded and the ROI/ticks for road5 lane 1 present.
2) From the repo root, run:
   python test_traffic_env.py --steps 300

Assumes CONTROL_TL_IDS in config.py points at the road5_lane-1_s10058 ROI and that CONTROL_TL_IDS is in CARLA's TL groups.
"""

import argparse
import time

from traffic_rl.env.traffic_runner import TrafficRunner
from traffic_rl.env.traffic_env import TrafficEnv


def main():
    parser = argparse.ArgumentParser(description="Random rollout sanity test for TrafficEnv.")
    parser.add_argument("--steps", type=int, default=300, help="How many env steps to run before exiting.")
    args = parser.parse_args()

    runner = TrafficRunner()
    env = TrafficEnv(runner, debug_obs=True)

    obs = env.reset()
    print(f"[INIT] obs shape={obs.shape}, phase={env.current_phase}")

    try:
        for step_idx in range(args.steps):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)

            print(
                f"[STEP {step_idx+1:03d}] action={action} reward={reward:.2f} "
                f"state={info.get('state', '?')} queue={info.get('queue_count', 0)} "
                f"avg_wait={info.get('avg_wait', 0.0):.1f} long={info.get('num_long_wait_60s', 0)} "
                f"t_state={info.get('time_in_state', 0.0):.1f}"
            )

            if done:
                print("[RESET] Episode done, resetting env")
                obs = env.reset()

            time.sleep(max(0.01, runner.dt))
    finally:
        env.runner.close()
        print("Finished random rollout test.")


if __name__ == "__main__":
    main()
