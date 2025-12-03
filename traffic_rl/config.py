import torch

# RL Hyperparameters
LR = 3e-4
GAMMA = 0.99
LAMBDA = 0.95
CLIP_RANGE = 0.2
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5
UPDATE_EPOCHS = 10
BATCH_SIZE = 64

# Training Settings
TOTAL_TIMESTEPS = 100000
STEPS_PER_EPOCH = 2048  # Rollout buffer size
SAVE_INTERVAL = 10      # Save every N updates
EVAL_INTERVAL = 5       # Evaluate every N updates
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Environment Settings
ACTION_DURATION_SEC = 5.0  # Time to hold a phase
YELLOW_DURATION_SEC = 3.0  # Yellow time when switching
REWARD_WEIGHTS = {
    "wait": 1.0,    # Penalty for average wait time
    "queue": 0.5,   # Penalty for queue length
    "long": 2.0,    # Penalty for long waiters (>60s)
    "change": 0.1   # Penalty for changing phase
}

# Paths
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "runs"
