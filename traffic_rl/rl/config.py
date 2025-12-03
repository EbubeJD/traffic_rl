"""
Configuration for RL training and environment.

This file contains:
1. Environment configuration (CARLA connection, episode settings)
2. PPO hyperparameters
3. Phase configuration for intersections (TO BE FILLED after running print_tl_info.py)
"""

import torch
import carla


# ============================================================================
# ENVIRONMENT CONFIGURATION
# ============================================================================

ENV_CONFIG = {
    # CARLA Connection
    "carla_host": "localhost",
    "carla_port": 2000,
    "town": "Town10HD_Opt",
    "group_index": 0,                    # Which traffic light group to control

    # Traffic Settings
    "num_vehicles": 40,                  # Fixed number of NPC vehicles

    # Episode Settings
    "episode_length_sec": 180.0,         # 3 minutes per episode
    "decision_interval_sec": 5.0,        # Agent decides every 5 seconds
    "min_green_time_sec": 10.0,          # Minimum green time per phase
    "warmup_time_sec": 20.0,             # Warm-up period to populate queues
    "dt": 0.05,                          # CARLA timestep (20 FPS)

    # Reward Configuration
    "reward_config": {
        "alpha": 1.0,    # Queue length weight (vehicles)
        "beta": 0.5,     # Average wait weight (seconds)
        "gamma": 2.0,    # Long wait penalty (equity focus)
        "delta": 0.1,    # Phase switch penalty (discourage thrashing)
    },
}


# ============================================================================
# PPO HYPERPARAMETERS
# ============================================================================

PPO_CONFIG = {
    # Training
    "total_timesteps": 100_000,          # Start with 100k for MVP testing
    "n_steps": 2048,                     # Steps per policy update
    "batch_size": 64,                    # Minibatch size
    "n_epochs": 10,                      # Optimization epochs per update

    # Learning Rates
    "learning_rate": 3e-4,               # Standard PPO learning rate
    "gamma": 0.99,                       # Discount factor
    "gae_lambda": 0.95,                  # GAE parameter

    # PPO-Specific
    "clip_range": 0.2,                   # PPO clip parameter
    "clip_range_vf": None,               # No value function clipping
    "ent_coef": 0.01,                    # Entropy coefficient (mild exploration)
    "vf_coef": 0.5,                      # Value function coefficient
    "max_grad_norm": 0.5,                # Gradient clipping

    # Network Architecture
    "policy_kwargs": {
        "net_arch": [64, 64],            # 2-layer MLP, 64 units each
        "activation_fn": torch.nn.Tanh,
    },

    # Logging
    "tensorboard_log": "./runs/ppo_tensorboard/",
}


# ============================================================================
# PHASE CONFIGURATION FOR INTERSECTION 0
# ============================================================================
#
# IMPORTANT: Run `python tools/print_tl_info.py` to get the stable IDs
# for your traffic lights, then fill in this configuration.
#
# Each phase defines which traffic lights should be Green/Red.
# The agent will choose between "keep current phase" (action 0) or
# "switch to next phase" (action 1).
#
# Example configuration (replace with actual stable IDs):
#
# INTERSECTION_0_PHASES = {
#     "phase_0": {  # e.g., North-South Green
#         "road42_lane5_s1234": carla.TrafficLightState.Green,
#         "road42_lane6_s1235": carla.TrafficLightState.Green,
#         "road43_lane2_s2341": carla.TrafficLightState.Red,
#         "road43_lane3_s2342": carla.TrafficLightState.Red,
#     },
#     "phase_1": {  # e.g., East-West Green
#         "road42_lane5_s1234": carla.TrafficLightState.Red,
#         "road42_lane6_s1235": carla.TrafficLightState.Red,
#         "road43_lane2_s2341": carla.TrafficLightState.Green,
#         "road43_lane3_s2342": carla.TrafficLightState.Green,
#     },
# }

# Placeholder - MUST BE FILLED before running training
INTERSECTION_0_PHASES = {
    "phase_0": {
        # Fill in after running print_tl_info.py
        # "stable_id_1": carla.TrafficLightState.Green,
        # "stable_id_2": carla.TrafficLightState.Red,
    },
    "phase_1": {
        # Fill in after running print_tl_info.py
        # "stable_id_1": carla.TrafficLightState.Red,
        # "stable_id_2": carla.TrafficLightState.Green,
    },
}
