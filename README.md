# Traffic RL Controller

This project implements an adaptive traffic light controller using Proximal Policy Optimization (PPO) reinforcement learning, integrated with CARLA simulator.

## Project Structure

The project is organized into the following modules:

- `traffic_rl/`: Main package containing RL components.
  - `config.py`: Configuration for RL hyperparameters and environment settings.
  - `env/`: Environment wrappers.
    - `traffic_runner.py`: Handles the low-level interface with CARLA (actor management, simulation stepping).
    - `traffic_env.py`: OpenAI Gym-compatible wrapper that defines the State, Action, and Reward.
  - `agent/`: PPO implementation.
    - `ppo.py`: Contains the `PPOAgent` class and `ActorCritic` neural network.
    - `buffer.py`: `RolloutBuffer` for storing trajectories and computing GAE-Lambda.
- `train_rl.py`: Main entry point for training the agent.
- `run_sim.py`: Original simulation script (reference).
- `observers/`: Helper classes for camera and metric logging.
  - `tl_observer.py`: Computes queue lengths and wait times from camera data.

## Components Detail

### TrafficRunner (`traffic_rl/env/traffic_runner.py`)
This class encapsulates the CARLA simulation logic. It:
- Connects to the CARLA client and loads the specified town.
- Identifies traffic light groups.
- Spawns `TLObserver` instances for each traffic light in the group.
- Spawns autopilot vehicles for traffic generation.
- Provides a `step()` method that applies traffic light states and advances the simulation.
- Provides a `reset()` method that clears traffic and respawns vehicles.

### TrafficEnv (`traffic_rl/env/traffic_env.py`)
This class wraps `TrafficRunner` into a Gym environment.
- **Action Space**: Discrete(2). 
  - 0: Phase A (e.g., North-South Green)
  - 1: Phase B (e.g., East-West Green)
- **Observation Space**: Box(N). A vector containing queue length, queue EMA, and time-in-state for each traffic light, plus the current phase index.
- **Reward Function**: A weighted sum of negative penalties:
  - Average Wait Time (Weight: 1.0)
  - Queue Length (Weight: 0.5)
  - Long Waiters (>60s) (Weight: 2.0)
  - Phase Change Penalty (Weight: 0.1)

### PPO Agent (`traffic_rl/agent/ppo.py`)
Implements the PPO algorithm.
- **Actor**: 3-layer MLP outputting logits for the discrete action space.
- **Critic**: 3-layer MLP outputting state value estimate.
- **Optimization**: Uses Adam optimizer with clipped surrogate objective and GAE-Lambda for advantage estimation.

## Configuration

All hyperparameters are defined in `traffic_rl/config.py`. Key parameters include:
- `LR`: Learning rate (default: 3e-4)
- `GAMMA`: Discount factor (default: 0.99)
- `STEPS_PER_EPOCH`: Number of steps to collect before updating (default: 2048)
- `ACTION_DURATION_SEC`: Duration to hold a selected phase (default: 5.0s)

## How to Run

1. Ensure CARLA is running (e.g., `CarlaUE4.exe`).
2. Install dependencies:
   ```bash
   pip install torch numpy gym carla
   ```
3. Run the training script:
   ```bash
   python train_rl.py
   ```

## Outputs

Training artifacts are saved in the `runs/` directory. Each run creates a subdirectory with a timestamp.
- `log.csv`: Contains per-epoch metrics (Reward, Loss, FPS).
- `checkpoints/`: Contains saved model weights (`.pt` files).
