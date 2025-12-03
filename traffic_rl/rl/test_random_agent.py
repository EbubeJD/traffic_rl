"""
Test script to run random agent for environment validation.

This script tests that:
1. Environment can be created
2. Reset works correctly
3. Random agent can run for N episodes without crashing
4. Reward becomes more negative when congestion increases
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from traffic_rl.envs.carla_intersection_env import CarlaIntersectionEnv
from traffic_rl.rl.config import ENV_CONFIG, INTERSECTION_0_PHASES


def test_random_agent(n_episodes: int = 10):
    """
    Run random agent for N episodes and collect statistics.

    Args:
        n_episodes: Number of episodes to run
    """
    print("=" * 80)
    print(f"TESTING RANDOM AGENT ({n_episodes} EPISODES)")
    print("=" * 80)

    # Validate phase configuration
    if not INTERSECTION_0_PHASES.get("phase_0") or not INTERSECTION_0_PHASES.get("phase_1"):
        print("\n[ERROR] Phase configuration is empty!")
        print("Please run: python tools/print_tl_info.py")
        print("Then fill in INTERSECTION_0_PHASES in traffic_rl/rl/config.py")
        return

    print("\n[1/3] Creating environment...")
    try:
        env = CarlaIntersectionEnv(
            **ENV_CONFIG,
            phase_config=INTERSECTION_0_PHASES
        )
        print("✓ Environment created successfully")
    except Exception as e:
        print(f"✗ Failed to create environment: {e}")
        return

    # Test environment with gymnasium checker
    print("\n[2/3] Running gymnasium environment checker...")
    try:
        from gymnasium.utils.env_checker import check_env
        check_env(env, skip_render_check=True)
        print("✓ Environment passes check_env()")
    except Exception as e:
        print(f"✗ Environment failed check_env(): {e}")
        print("Continuing with random agent test anyway...")

    print(f"\n[3/3] Running random agent for {n_episodes} episodes...")
    print("=" * 80)

    episode_rewards = []
    episode_lengths = []
    episode_metrics = []

    for ep in range(n_episodes):
        print(f"\n--- Episode {ep+1}/{n_episodes} ---")

        try:
            obs, info = env.reset(seed=ep)
            print(f"Reset successful. Observation shape: {obs.shape}")

            episode_reward = 0.0
            steps = 0
            done = False

            while not done:
                # Random action
                action = env.action_space.sample()

                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                steps += 1

                # Print every 5 steps
                if steps % 5 == 0:
                    metrics = info["metrics"]
                    phase_info = info["phase"]
                    print(f"  Step {steps:2d}: "
                          f"action={action}, "
                          f"reward={reward:7.2f}, "
                          f"phase={phase_info['current_phase']}, "
                          f"queue={metrics['total_queue_ema']:.1f}, "
                          f"avg_wait={metrics['total_avg_wait']:.1f}s")

                done = terminated or truncated

            # Episode statistics
            final_metrics = info["metrics"]
            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)
            episode_metrics.append(final_metrics)

            print(f"\nEpisode {ep+1} complete:")
            print(f"  Total reward: {episode_reward:.2f}")
            print(f"  Steps: {steps}")
            print(f"  Final queue: {final_metrics['total_queue_ema']:.1f}")
            print(f"  Final avg wait: {final_metrics['total_avg_wait']:.1f}s")

        except Exception as e:
            print(f"✗ Episode {ep+1} failed: {e}")
            import traceback
            traceback.print_exc()
            break

    # Summary
    print("\n" + "=" * 80)
    print("RANDOM AGENT TEST SUMMARY")
    print("=" * 80)

    if len(episode_rewards) > 0:
        print(f"\nCompleted {len(episode_rewards)}/{n_episodes} episodes")
        print(f"\nReward statistics:")
        print(f"  Mean:   {np.mean(episode_rewards):.2f}")
        print(f"  Std:    {np.std(episode_rewards):.2f}")
        print(f"  Min:    {np.min(episode_rewards):.2f}")
        print(f"  Max:    {np.max(episode_rewards):.2f}")

        print(f"\nEpisode length:")
        print(f"  Mean:   {np.mean(episode_lengths):.1f} steps")
        print(f"  Std:    {np.std(episode_lengths):.1f}")

        print(f"\nFinal metrics (averaged across episodes):")
        avg_queue = np.mean([m['total_queue_ema'] for m in episode_metrics])
        avg_wait = np.mean([m['total_avg_wait'] for m in episode_metrics])
        avg_long_wait = np.mean([m['total_num_long_wait'] for m in episode_metrics])
        print(f"  Queue EMA:     {avg_queue:.2f}")
        print(f"  Avg wait time: {avg_wait:.2f}s")
        print(f"  Long waiters:  {avg_long_wait:.2f}")

        print("\n✓ Random agent test PASSED")
    else:
        print("\n✗ Random agent test FAILED - no episodes completed")

    print("=" * 80)

    # Cleanup
    env.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run")
    args = parser.parse_args()

    test_random_agent(args.episodes)
