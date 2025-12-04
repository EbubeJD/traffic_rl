"""
Configuration for RL training.

Contains hyperparameters and reward weights.
"""

# Reward weights for traffic signal control
REWARD_WEIGHTS = {
    "wait": 1.0,    # Penalty for average wait time (seconds)
    "queue": 0.5,   # Penalty for queue length (vehicles)
    "long": 2.0,    # Penalty for long waiters (>60s) - prioritizes equity
    "change": 0.1   # Penalty for changing phase - discourages thrashing
}

# PPO hyperparameters
PPO_CONFIG = {
    # Training
    "n_steps": 512,              # Steps per rollout
    "batch_size": 64,             # Minibatch size
    "n_epochs": 10,               # Optimization epochs per update
    "learning_rate": 3e-4,        # Learning rate
    "gamma": 0.99,                # Discount factor
    "gae_lambda": 0.95,           # GAE lambda
    "clip_range": 0.2,            # PPO clipping parameter
    "value_coef": 0.5,            # Value loss coefficient
    "entropy_coef": 0.01,         # Entropy bonus coefficient
    "max_grad_norm": 0.5,         # Gradient clipping
    "hidden_sizes": (64, 64),     # Network architecture
}

# Environment settings
ENV_CONFIG = {
    "episode_steps": 50,         # Decision steps per episode
}

# Training settings
TRAIN_CONFIG = {
    "total_timesteps": 100_000,   # Total training steps
    "save_freq": 500,          # Save checkpoint every N steps
    "log_freq": 1,                # Log every N episodes
}
