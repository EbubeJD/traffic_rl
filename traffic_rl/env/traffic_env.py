"""
TrafficEnv: Gym environment wrapper for traffic signal control.

Implements PROMPT.md specifications:
- 8 features per traffic light (normalized [0,1])
- Reward using all 4 weights (wait, queue, long, change)
- Phase change penalty
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import gym
from gym import spaces
import carla

from traffic_rl.env.traffic_runner import TrafficRunner
from traffic_rl.config import REWARD_WEIGHTS


# Normalization constants
MAX_QUEUE = 20.0
MAX_WAIT_SEC = 120.0
MAX_LONG_WAITERS = 20.0
MAX_FLOW = 2.0  # veh/s for arrival_ema / discharge_ema
MAX_TIME_IN_STATE = 60.0


class TrafficEnv(gym.Env):
    """
    Gym environment for traffic signal control.

    Observation:
        Per TL (8 features each):
        - queue (normalized by MAX_QUEUE)
        - queue_ema (normalized by MAX_QUEUE)
        - avg_wait (normalized by MAX_WAIT_SEC)
        - max_wait (normalized by MAX_WAIT_SEC)
        - num_long_wait_60s (normalized by MAX_LONG_WAITERS)
        - arrival_ema (normalized by MAX_FLOW)
        - discharge_ema (normalized by MAX_FLOW)
        - time_in_state (normalized by MAX_TIME_IN_STATE)
        Plus:
        - current_phase (0 or 1, normalized)

    Action:
        Discrete(2): 0 = phase 0, 1 = phase 1

    Reward:
        -(wait_weight * avg_wait + queue_weight * queue + long_weight * long_waiters + change_penalty)
    """

    def __init__(self, runner: TrafficRunner, episode_steps: int = 120):
        """
        Initialize environment.

        Args:
            runner: TrafficRunner instance
            episode_steps: Number of decision steps per episode
        """
        super().__init__()

        self.runner = runner
        self.episode_steps = episode_steps

        # Initialize runner
        if not self.runner.is_initialized:
            self.runner.initialize()

        # Get traffic light info
        self.num_tls = len(self.runner.group["actors"])
        self.stable_ids = self.runner.get_stable_ids()

        # Observation space: 8 features per TL + 1 for current_phase
        self.per_tl_features = 8
        self.obs_dim = self.num_tls * self.per_tl_features + 1
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        # Action space: phase 0 or phase 1
        self.action_space = spaces.Discrete(2)

        # Episode state
        self.current_phase = 0
        self.step_count = 0

        print(f"[TrafficEnv] Environment initialized")
        print(f"  - Observation dim: {self.obs_dim} (= {self.num_tls} TLs × {self.per_tl_features} features + 1 phase)")
        print(f"  - Action space: Discrete(2)")
        print(f"  - Episode steps: {self.episode_steps}")

    def reset(self):
        """
        Reset environment for new episode.

        Returns:
            Initial observation (old Gym API: just obs, not tuple)
        """
        # Reset runner
        self.runner.reset()

        # Reset episode state
        self.current_phase = 0
        self.step_count = 0

        # Apply initial phase
        self._apply_phase(self.current_phase)

        # Run a few steps to populate metrics
        for _ in range(5):
            self.runner.step()

        # Get initial metrics and observation
        metrics = self.runner.step()
        obs = self._get_obs(metrics)

        return obs

    def step(self, action):
        """
        Execute one environment step.

        Args:
            action: 0 or 1 (phase selection)

        Returns:
            obs, reward, done, info (old Gym API: 4-tuple)
        """
        # Track if phase changed
        phase_changed = (action != self.current_phase)

        # Apply phase
        if phase_changed:
            self.current_phase = action
            self._apply_phase(self.current_phase)

        # Step simulation multiple times (e.g., 5 seconds = 100 ticks at 0.05s)
        steps = int(5.0 / self.runner.dt)  # 5 second decision interval
        step_metrics = []
        for _ in range(steps):
            m = self.runner.step(self._get_light_states())
            step_metrics.append(m)

        # Get last metrics for observation
        last_metrics = step_metrics[-1] if step_metrics else {}

        # Compute reward
        reward = self._compute_reward(step_metrics, phase_changed)

        # Build observation
        obs = self._get_obs(last_metrics)

        # Update step count
        self.step_count += 1
        done = (self.step_count >= self.episode_steps)

        # Info
        info = {
            "phase": self.current_phase,
            "phase_changed": phase_changed,
            "step": self.step_count,
        }

        # Expose per-lane metrics for quick debugging (single-lane focus).
        primary_sid = self.runner.observers[0].stable_id if self.runner.observers else None
        if primary_sid and primary_sid in last_metrics:
            m = last_metrics[primary_sid]
            info.update({
                "state": m.get("state"),
                "queue_count": m.get("queue", 0.0),
                "avg_wait": m.get("avg_wait", 0.0),
                "num_long_wait_60s": m.get("num_long_wait_60s", 0.0),
                "time_in_state": m.get("time_in_state", 0.0),
            })

        return obs, reward, done, info

    def _get_obs(self, metrics: dict) -> np.ndarray:
        """
        Build observation vector from metrics.

        Args:
            metrics: Dict mapping stable_id -> metrics dict

        Returns:
            Observation array of shape (obs_dim,)
        """
        obs = []

        # For each traffic light (in stable order)
        for obs_obj in self.runner.observers:
            sid = obs_obj.stable_id

            if sid in metrics:
                m = metrics[sid]

                # Extract features
                queue = m.get("queue", 0.0)
                queue_ema = m.get("queue_ema", 0.0)
                avg_wait = m.get("avg_wait", 0.0)
                max_wait = m.get("max_wait", 0.0)
                num_long = m.get("num_long_wait_60s", 0.0)
                arrival = m.get("arrival_ema", 0.0)
                discharge = m.get("discharge_ema", 0.0)
                time_in_state = m.get("time_in_state", 0.0)

                # Normalize and clip to [0, 1]
                queue_norm = np.clip(queue / MAX_QUEUE, 0.0, 1.0)
                queue_ema_norm = np.clip(queue_ema / MAX_QUEUE, 0.0, 1.0)
                avg_wait_norm = np.clip(avg_wait / MAX_WAIT_SEC, 0.0, 1.0)
                max_wait_norm = np.clip(max_wait / MAX_WAIT_SEC, 0.0, 1.0)
                num_long_norm = np.clip(num_long / MAX_LONG_WAITERS, 0.0, 1.0)
                arrival_norm = np.clip(arrival / MAX_FLOW, 0.0, 1.0)
                discharge_norm = np.clip(discharge / MAX_FLOW, 0.0, 1.0)
                time_in_state_norm = np.clip(time_in_state / MAX_TIME_IN_STATE, 0.0, 1.0)

                per_tl_features = [
                    queue_norm,
                    queue_ema_norm,
                    avg_wait_norm,
                    max_wait_norm,
                    num_long_norm,
                    arrival_norm,
                    discharge_norm,
                    time_in_state_norm,
                ]
                obs.extend(per_tl_features)

                # Debug: raw + normalized values for this TL (helps align ROI/ticks to obs)
                debug_features = per_tl_features + [float(self.current_phase)]
                print(
                    f"[OBS][{sid}] q={queue:.1f}, q_ema={queue_ema:.1f}, "
                    f"avg_wait={avg_wait:.1f}, max_wait={max_wait:.1f}, "
                    f"long={num_long}, arr_ema={arrival:.1f}, "
                    f"dis_ema={discharge:.1f}, t_state={time_in_state:.2f}"
                )
                print(f"[OBS][{sid}] normalized={debug_features}")
            else:
                # No metrics available, use zeros
                obs.extend([0.0] * self.per_tl_features)

        # Append current phase (normalized to [0, 1])
        obs.append(float(self.current_phase))

        return np.array(obs, dtype=np.float32)

    def _compute_reward(self, step_metrics_list: list, phase_changed: bool) -> float:
        """
        Compute reward using all 4 REWARD_WEIGHTS.

        Args:
            step_metrics_list: List of metrics dicts from simulation steps
            phase_changed: Whether phase was changed this step

        Returns:
            Scalar reward (more negative = worse traffic)
        """
        total_penalty = 0.0
        count = 0

        # Average over time steps and traffic lights
        for metrics_dict in step_metrics_list:
            for sid, m in metrics_dict.items():
                avg_wait = m.get("avg_wait", 0.0)
                queue = m.get("queue", 0.0)
                num_long = m.get("num_long_wait_60s", 0.0)

                # Penalty = weighted sum
                step_penalty = (
                    REWARD_WEIGHTS["wait"] * avg_wait +
                    REWARD_WEIGHTS["queue"] * queue +
                    REWARD_WEIGHTS["long"] * num_long
                )

                total_penalty += step_penalty
                count += 1

        # Average penalty
        if count > 0:
            avg_penalty = total_penalty / count
        else:
            avg_penalty = 0.0

        # Reward is negative penalty
        reward = -avg_penalty

        # Add phase change penalty
        if phase_changed:
            reward -= REWARD_WEIGHTS["change"]

        return float(reward)

    def _apply_phase(self, phase: int):
        """
        Apply traffic light states for given phase.

        For simplicity, this is a basic 2-phase alternating pattern.
        Override this method for more complex phase configurations.

        Args:
            phase: 0 or 1
        """
        # Simple alternating pattern:
        # Phase 0: First half of lights green, second half red
        # Phase 1: First half red, second half green

        light_states = {}
        num_lights = len(self.runner.group["actors"])
        split_idx = num_lights // 2

        for i, tl in enumerate(self.runner.group["actors"]):
            if phase == 0:
                # Phase 0: first half green
                state = carla.TrafficLightState.Green if i < split_idx else carla.TrafficLightState.Red
            else:
                # Phase 1: second half green
                state = carla.TrafficLightState.Red if i < split_idx else carla.TrafficLightState.Green

            light_states[tl.id] = state

        # Apply states immediately
        for tl in self.runner.group["actors"]:
            if tl.id in light_states:
                tl.set_state(light_states[tl.id])

    def _get_light_states(self):
        """
        Get current light states (for passing to runner.step()).

        Returns None since we apply states directly in _apply_phase.
        """
        return None


# Sanity check test
if __name__ == "__main__":
    print("=" * 80)
    print("TRAFFIC ENV SANITY CHECK")
    print("=" * 80)

    runner = TrafficRunner()
    env = TrafficEnv(runner)

    print("\n[1] Testing reset()...")
    obs = env.reset()
    print(f"  ✓ Obs shape: {obs.shape}")
    print(f"  ✓ Obs min/max: {obs.min():.3f} / {obs.max():.3f}")
    print(f"  ✓ Obs dtype: {obs.dtype}")

    print("\n[2] Testing step()...")
    for i in range(3):
        action = env.action_space.sample()
        obs2, reward, done, info = env.step(action)
        print(f"  Step {i+1}: action={action}, reward={reward:.2f}, done={done}, phase={info['phase']}")

    print("\n[3] Testing observation space...")
    assert env.observation_space.contains(obs), "Observation not in space!"
    print("  ✓ Observation in observation_space")

    print("\n" + "=" * 80)
    print("✓ SANITY CHECK PASSED")
    print("=" * 80)

    env.runner.close()
