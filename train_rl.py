"""
Main training script for PPO traffic signal control.

Usage:
    python train_rl.py
"""

import sys
import os
import time
import numpy as np

from traffic_rl.env.traffic_runner import TrafficRunner
from traffic_rl.env.traffic_env import TrafficEnv
from traffic_rl.agent.ppo import PPOAgent
from traffic_rl.agent.buffer import RolloutBuffer
from traffic_rl.config import PPO_CONFIG, ENV_CONFIG, TRAIN_CONFIG


def main():
    print("=" * 80)
    print("PPO TRAINING FOR TRAFFIC SIGNAL CONTROL")
    print("=" * 80)

    # Create environment
    print("\n[1/4] Creating environment...")
    runner = TrafficRunner()
    env = TrafficEnv(runner, episode_steps=ENV_CONFIG["episode_steps"])
    print("✓ Environment created")

    # Create agent
    print("\n[2/4] Creating PPO agent...")
    agent = PPOAgent(
        obs_dim=env.obs_dim,
        action_dim=env.action_space.n,
        lr=PPO_CONFIG["learning_rate"],
        gamma=PPO_CONFIG["gamma"],
        gae_lambda=PPO_CONFIG["gae_lambda"],
        clip_range=PPO_CONFIG["clip_range"],
        value_coef=PPO_CONFIG["value_coef"],
        entropy_coef=PPO_CONFIG["entropy_coef"],
        max_grad_norm=PPO_CONFIG["max_grad_norm"],
        hidden_sizes=PPO_CONFIG["hidden_sizes"],
    )
    print("✓ PPO agent created")

    # Create rollout buffer
    buffer = RolloutBuffer(
        buffer_size=PPO_CONFIG["n_steps"],
        obs_dim=env.obs_dim,
        gamma=PPO_CONFIG["gamma"],
        gae_lambda=PPO_CONFIG["gae_lambda"],
    )

    # Training loop
    print("\n[3/4] Starting training...")
    print(f"  - Total timesteps: {TRAIN_CONFIG['total_timesteps']}")
    print(f"  - Rollout size: {PPO_CONFIG['n_steps']}")
    print(f"  - Episode steps: {ENV_CONFIG['episode_steps']}")
    print("=" * 80)

    total_timesteps = 0
    episode_num = 0
    episode_rewards = []
    episode_lengths = []

    try:
        while total_timesteps < TRAIN_CONFIG["total_timesteps"]:
            # Reset environment
            obs = env.reset()
            episode_reward = 0.0
            episode_length = 0
            done = False

            episode_start = time.time()

            # Collect rollout
            while not done and not buffer.is_full():
                # Select action
                action, log_prob, value = agent.select_action(obs)

                # Step environment
                next_obs, reward, done, info = env.step(action)

                # Store in buffer
                buffer.add(obs, action, reward, value, log_prob, done)

                # Update state
                obs = next_obs
                episode_reward += reward
                episode_length += 1
                total_timesteps += 1

            # Finish trajectory in buffer
            if done:
                last_value = 0.0
            else:
                _, _, last_value = agent.select_action(obs)

            buffer.finish_path(last_value)

            # Log episode
            episode_num += 1
            episode_time = time.time() - episode_start
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            if episode_num % TRAIN_CONFIG["log_freq"] == 0:
                recent_rewards = episode_rewards[-10:]
                mean_reward = np.mean(recent_rewards)
                print(f"Episode {episode_num:4d} | "
                      f"Steps: {total_timesteps:6d}/{TRAIN_CONFIG['total_timesteps']} | "
                      f"Reward: {episode_reward:7.2f} | "
                      f"Mean(10): {mean_reward:7.2f} | "
                      f"Length: {episode_length:3d} | "
                      f"Time: {episode_time:.1f}s")

            # Update policy when buffer is full
            if buffer.is_full():
                print(f"\n[UPDATE] Training PPO at {total_timesteps} steps...")
                rollout_data = buffer.get()
                metrics = agent.update(
                    rollout_data,
                    n_epochs=PPO_CONFIG["n_epochs"],
                    batch_size=PPO_CONFIG["batch_size"]
                )
                print(f"  Policy Loss: {metrics['policy_loss']:.4f} | "
                      f"Value Loss: {metrics['value_loss']:.4f} | "
                      f"Entropy: {metrics['entropy']:.4f} | "
                      f"Approx KL: {metrics['approx_kl']:.4f}\n")

            # Save checkpoint
            if total_timesteps % TRAIN_CONFIG["save_freq"] == 0 and total_timesteps > 0:
                checkpoint_path = f"checkpoints/ppo_traffic_{total_timesteps}.pt"
                os.makedirs("checkpoints", exist_ok=True)
                agent.save(checkpoint_path)
                print(f"[CHECKPOINT] Saved to {checkpoint_path}")

    except KeyboardInterrupt:
        print("\n[INTERRUPT] Training interrupted by user")

    # Final summary
    print("\n" + "=" * 80)
    print("[4/4] TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nTotal episodes: {episode_num}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"\nReward statistics:")
    print(f"  Mean: {np.mean(episode_rewards):.2f}")
    print(f"  Std:  {np.std(episode_rewards):.2f}")
    print(f"  Min:  {np.min(episode_rewards):.2f}")
    print(f"  Max:  {np.max(episode_rewards):.2f}")

    # Save final model
    final_path = "ppo_traffic_final.pt"
    agent.save(final_path)
    print(f"\n✓ Final model saved to: {final_path}")
    print("=" * 80)

    # Cleanup
    env.runner.close()


if __name__ == "__main__":
    main()
