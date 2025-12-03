"""
PPO training script for traffic signal control.

Usage:
    python traffic_rl/rl/train_ppo.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.logger import configure

from traffic_rl.envs.carla_intersection_env import CarlaIntersectionEnv
from traffic_rl.rl.config import PPO_CONFIG, ENV_CONFIG, INTERSECTION_0_PHASES


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent for traffic signal control")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=PPO_CONFIG["total_timesteps"],
        help=f"Total training timesteps (default: {PPO_CONFIG['total_timesteps']})"
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=10000,
        help="Save checkpoint every N steps (default: 10000)"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Directory to save checkpoints (default: ./checkpoints)"
    )
    parser.add_argument(
        "--tensorboard-log",
        type=str,
        default=PPO_CONFIG["tensorboard_log"],
        help=f"TensorBoard log directory (default: {PPO_CONFIG['tensorboard_log']})"
    )
    args = parser.parse_args()

    print("=" * 80)
    print("PPO TRAINING FOR TRAFFIC SIGNAL CONTROL")
    print("=" * 80)

    # Validate phase configuration
    if not INTERSECTION_0_PHASES.get("phase_0") or not INTERSECTION_0_PHASES.get("phase_1"):
        print("\n[ERROR] Phase configuration is empty!")
        print("Please run: python tools/print_tl_info.py")
        print("Then fill in INTERSECTION_0_PHASES in traffic_rl/rl/config.py")
        print("=" * 80)
        return

    print(f"\n[1/4] Creating environment...")
    print(f"  - CARLA: {ENV_CONFIG['carla_host']}:{ENV_CONFIG['carla_port']}")
    print(f"  - Town: {ENV_CONFIG['town']}")
    print(f"  - Group: {ENV_CONFIG['group_index']}")
    print(f"  - Episode length: {ENV_CONFIG['episode_length_sec']}s")
    print(f"  - Decision interval: {ENV_CONFIG['decision_interval_sec']}s")

    # Create environment
    try:
        env = CarlaIntersectionEnv(
            **ENV_CONFIG,
            phase_config=INTERSECTION_0_PHASES
        )
        print(f"✓ Environment created")
    except Exception as e:
        print(f"✗ Failed to create environment: {e}")
        print("\nMake sure CARLA is running on the specified host/port")
        return

    print(f"\n[2/4] Creating PPO model...")
    print(f"  - Learning rate: {PPO_CONFIG['learning_rate']}")
    print(f"  - Network: {PPO_CONFIG['policy_kwargs']['net_arch']}")
    print(f"  - Batch size: {PPO_CONFIG['batch_size']}")
    print(f"  - n_steps: {PPO_CONFIG['n_steps']}")

    # Create PPO model
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=PPO_CONFIG["learning_rate"],
        n_steps=PPO_CONFIG["n_steps"],
        batch_size=PPO_CONFIG["batch_size"],
        n_epochs=PPO_CONFIG["n_epochs"],
        gamma=PPO_CONFIG["gamma"],
        gae_lambda=PPO_CONFIG["gae_lambda"],
        clip_range=PPO_CONFIG["clip_range"],
        clip_range_vf=PPO_CONFIG["clip_range_vf"],
        ent_coef=PPO_CONFIG["ent_coef"],
        vf_coef=PPO_CONFIG["vf_coef"],
        max_grad_norm=PPO_CONFIG["max_grad_norm"],
        policy_kwargs=PPO_CONFIG["policy_kwargs"],
        verbose=1,
        tensorboard_log=args.tensorboard_log,
    )
    print(f"✓ PPO model created")

    print(f"\n[3/4] Setting up callbacks...")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=args.checkpoint_dir,
        name_prefix="ppo_traffic",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    print(f"  - Checkpoints: {args.checkpoint_dir} (every {args.save_freq} steps)")
    print(f"  - TensorBoard: {args.tensorboard_log}")

    callbacks = CallbackList([checkpoint_callback])

    print(f"\n[4/4] Starting training...")
    print(f"  - Total timesteps: {args.timesteps}")
    print(f"  - Estimated episodes: ~{args.timesteps * ENV_CONFIG['decision_interval_sec'] / ENV_CONFIG['episode_length_sec']:.0f}")
    print("=" * 80)
    print()

    # Train
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            progress_bar=True,
        )
        print("\n" + "=" * 80)
        print("Training completed successfully!")
    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print("Training interrupted by user")
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"Training failed: {e}")
        raise
    finally:
        # Save final model
        final_model_path = "ppo_traffic_final"
        print(f"Saving final model to: {final_model_path}.zip")
        model.save(final_model_path)
        print("=" * 80)

        # Cleanup
        env.close()


if __name__ == "__main__":
    main()
