"""
Generate offline RL training data using queue-aware heuristic policy.

Runs TrafficEnv with MaxQueuePolicy to collect high-quality transitions.csv
that has adaptive actions and rich observation features (non-zero queues/waits).

Usage:
    # Backup old data
    mv outputs/Town10HD_Opt/tl_road5_lane-1_s10058/transitions.csv transitions_old.csv

    # Generate new data
    python tools/generate_heuristic_data.py --policy max_queue --episodes 10 --steps 120

    # Verify improvement
    python tools/analyze_transitions.py --transitions outputs/.../transitions.csv
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import numpy as np

from traffic_rl.env.traffic_runner import TrafficRunner
from traffic_rl.env.traffic_env import TrafficEnv
from policies.queue_aware_heuristic import MaxQueuePolicy


def generate_data(args):
    """Run heuristic policy to generate transitions."""
    print("="*80)
    print("HEURISTIC DATA GENERATION")
    print("="*80)
    print(f"Policy: {args.policy}")
    print(f"Episodes: {args.episodes}")
    print(f"Steps per episode: {args.steps}")
    print(f"Min green time: {args.min_green}s")
    print(f"Queue threshold: {args.queue_threshold} vehicles")

    # Initialize policy
    if args.policy == "max_queue":
        policy = MaxQueuePolicy(
            min_green_time=args.min_green,
            queue_threshold=args.queue_threshold,
            dt=2.0  # TrafficEnv uses 2s decision intervals by default
        )
        print(f"\n✓ Initialized MaxQueuePolicy")
    else:
        raise ValueError(f"Unknown policy: {args.policy}")

    # Initialize environment
    print(f"\nInitializing TrafficRunner...")
    runner = TrafficRunner()

    print(f"Initializing TrafficEnv...")
    env = TrafficEnv(runner, episode_steps=args.steps)

    # Run episodes
    print(f"\nRunning {args.episodes} episodes...")
    print("="*80)

    episode_rewards = []
    episode_queues = []
    total_steps = 0

    for episode in range(args.episodes):
        print(f"\n[Episode {episode+1}/{args.episodes}]")

        obs = env.reset()
        policy.reset()

        episode_reward = 0.0
        episode_queue_sum = 0.0
        steps = 0

        for step in range(args.steps):
            # Get action from policy
            action = policy.act(obs)

            # Step environment
            obs, reward, done, info = env.step(action)

            # Track metrics
            episode_reward += reward
            episode_queue_sum += info.get("queue_count", 0.0)
            steps += 1
            total_steps += 1

            # Print progress
            if (step + 1) % 20 == 0:
                print(f"  Step {step+1}/{args.steps}: "
                      f"action={action}, phase={info.get('phase')}, "
                      f"queue={info.get('queue_count', 0):.1f}, "
                      f"reward={reward:.2f}")

            if done:
                print(f"  Episode finished at step {step+1}")
                break

        # Episode summary
        avg_queue = episode_queue_sum / max(steps, 1)
        episode_rewards.append(episode_reward)
        episode_queues.append(avg_queue)

        print(f"  Episode reward: {episode_reward:.2f}")
        print(f"  Average queue: {avg_queue:.2f}")

    # Close environment
    env.close()

    # Overall summary
    print("\n" + "="*80)
    print("DATA GENERATION COMPLETE")
    print("="*80)
    print(f"Total steps: {total_steps}")
    print(f"Episodes: {len(episode_rewards)}")
    print(f"Avg episode reward: {np.mean(episode_rewards):.2f}")
    print(f"Avg episode queue: {np.mean(episode_queues):.2f}")

    # Report save location
    if runner.observers:
        save_dir = runner.observers[0].save_dir
        transitions_file = os.path.join(save_dir, "transitions.csv")
        actions_file = os.path.join(save_dir, "actions.csv")

        print(f"\n✓ Data saved to:")
        print(f"  - {transitions_file}")
        print(f"  - {actions_file}")
        print(f"\nNext steps:")
        print(f"  1. Analyze new data:")
        print(f"     python tools/analyze_transitions.py --transitions {transitions_file}")
        print(f"  ")
        print(f"  2. Train BC on new data:")
        print(f"     python tools/train_bc.py --transitions {transitions_file} "
              f"--epochs 50 --normalize --out bc_policy_v2.pt")
    else:
        print(f"\n⚠ No observers found - transitions may not have been saved")

    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Generate offline RL data using heuristic policy"
    )
    parser.add_argument("--policy", type=str, default="max_queue",
                        choices=["max_queue"],
                        help="Heuristic policy to use")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes to run")
    parser.add_argument("--steps", type=int, default=120,
                        help="Steps per episode")
    parser.add_argument("--min_green", type=float, default=10.0,
                        help="Minimum green time (seconds)")
    parser.add_argument("--queue_threshold", type=float, default=0.5,
                        help="Queue threshold (vehicles) to keep green")
    args = parser.parse_args()

    generate_data(args)


if __name__ == "__main__":
    main()
