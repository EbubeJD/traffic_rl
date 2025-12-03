"""
Rollout buffer for PPO with Generalized Advantage Estimation (GAE).
"""

import numpy as np
import torch


class RolloutBuffer:
    """
    Buffer for storing trajectories during PPO training.

    Computes advantages using Generalized Advantage Estimation (GAE).
    """

    def __init__(self, buffer_size: int, obs_dim: int, gamma: float = 0.99, gae_lambda: float = 0.95):
        """
        Initialize rollout buffer.

        Args:
            buffer_size: Maximum number of timesteps to store
            obs_dim: Observation dimension
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # Storage
        self.observations = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.int64)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)

        # Computed after trajectory ends
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)

        self.ptr = 0
        self.path_start_idx = 0

    def add(self, obs, action, reward, value, log_prob, done):
        """
        Add a single timestep to the buffer.

        Args:
            obs: Observation
            action: Action taken
            reward: Reward received
            value: Value estimate from critic
            log_prob: Log probability of action
            done: Whether episode ended
        """
        assert self.ptr < self.buffer_size, "Buffer overflow"

        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done

        self.ptr += 1

    def finish_path(self, last_value=0.0):
        """
        Compute advantages and returns using GAE when trajectory ends.

        Args:
            last_value: Bootstrap value if episode was truncated (not done)
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = self.rewards[path_slice]
        values = self.values[path_slice]
        dones = self.dones[path_slice]

        # Append bootstrap value
        values_with_bootstrap = np.append(values, last_value)

        # Compute GAE advantages
        advantages = np.zeros_like(rewards)
        last_gae_lam = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam

        # Returns = advantages + values
        returns = advantages + values

        self.advantages[path_slice] = advantages
        self.returns[path_slice] = returns

        self.path_start_idx = self.ptr

    def get(self):
        """
        Get all data from buffer and reset.

        Returns:
            Dictionary with observations, actions, log_probs, advantages, returns, values
        """
        assert self.ptr == self.buffer_size, "Buffer not full"
        self.ptr, self.path_start_idx = 0, 0

        # Normalize advantages
        adv_mean = self.advantages.mean()
        adv_std = self.advantages.std()
        self.advantages = (self.advantages - adv_mean) / (adv_std + 1e-8)

        data = {
            "observations": torch.from_numpy(self.observations),
            "actions": torch.from_numpy(self.actions),
            "log_probs": torch.from_numpy(self.log_probs),
            "advantages": torch.from_numpy(self.advantages),
            "returns": torch.from_numpy(self.returns),
            "values": torch.from_numpy(self.values),
        }

        return data

    def is_full(self):
        """Check if buffer is full."""
        return self.ptr >= self.buffer_size
