"""
Custom PPO agent implementation with Actor-Critic network.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np


class ActorCritic(nn.Module):
    """
    Actor-Critic network for discrete action spaces.

    Actor outputs action probabilities.
    Critic outputs state value estimates.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes=(64, 64)):
        """
        Initialize Actor-Critic network.

        Args:
            obs_dim: Observation dimension
            action_dim: Number of discrete actions
            hidden_sizes: Tuple of hidden layer sizes
        """
        super().__init__()

        # Shared feature extraction
        layers = []
        prev_size = obs_dim
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.Tanh()
            ])
            prev_size = hidden_size

        self.features = nn.Sequential(*layers)

        # Actor head (policy)
        self.actor = nn.Linear(prev_size, action_dim)

        # Critic head (value function)
        self.critic = nn.Linear(prev_size, 1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights with orthogonal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

        # Actor head gets smaller init
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.constant_(self.actor.bias, 0.0)

    def forward(self, obs):
        """
        Forward pass through network.

        Args:
            obs: Observations tensor [batch_size, obs_dim]

        Returns:
            action_logits: Logits for action distribution [batch_size, action_dim]
            values: State value estimates [batch_size, 1]
        """
        features = self.features(obs)
        action_logits = self.actor(features)
        values = self.critic(features)
        return action_logits, values

    def get_action(self, obs, deterministic=False):
        """
        Sample action from policy.

        Args:
            obs: Observation [obs_dim] or [batch_size, obs_dim]
            deterministic: If True, take argmax action

        Returns:
            action: Sampled action
            log_prob: Log probability of action
            value: Value estimate
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        action_logits, value = self.forward(obs)
        dist = Categorical(logits=action_logits)

        if deterministic:
            action = action_logits.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.squeeze().item()

    def evaluate_actions(self, obs, actions):
        """
        Evaluate actions for PPO update.

        Args:
            obs: Observations [batch_size, obs_dim]
            actions: Actions taken [batch_size]

        Returns:
            log_probs: Log probabilities of actions [batch_size]
            values: Value estimates [batch_size]
            entropy: Entropy of action distribution [batch_size]
        """
        action_logits, values = self.forward(obs)
        dist = Categorical(logits=action_logits)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, values.squeeze(), entropy


class PPOAgent:
    """
    Proximal Policy Optimization agent.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        hidden_sizes=(64, 64),
        device="cpu"
    ):
        """
        Initialize PPO agent.

        Args:
            obs_dim: Observation dimension
            action_dim: Number of discrete actions
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda
            clip_range: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Gradient clipping threshold
            hidden_sizes: Network hidden layer sizes
            device: torch device
        """
        self.device = torch.device(device)

        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        # Network
        self.policy = ActorCritic(obs_dim, action_dim, hidden_sizes).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def select_action(self, obs, deterministic=False):
        """
        Select action given observation.

        Args:
            obs: Observation array
            deterministic: If True, select argmax action

        Returns:
            action, log_prob, value
        """
        obs_tensor = torch.FloatTensor(obs).to(self.device)
        with torch.no_grad():
            action, log_prob, value = self.policy.get_action(obs_tensor, deterministic)
        return action, log_prob, value

    def update(self, rollout_data, n_epochs=10, batch_size=64):
        """
        Update policy using PPO.

        Args:
            rollout_data: Dictionary from RolloutBuffer.get()
            n_epochs: Number of optimization epochs
            batch_size: Minibatch size

        Returns:
            Dictionary of training metrics
        """
        # Move data to device
        observations = rollout_data["observations"].to(self.device)
        actions = rollout_data["actions"].to(self.device)
        old_log_probs = rollout_data["log_probs"].to(self.device)
        advantages = rollout_data["advantages"].to(self.device)
        returns = rollout_data["returns"].to(self.device)

        n_samples = observations.shape[0]

        # Training metrics
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        n_updates = 0

        for epoch in range(n_epochs):
            # Shuffle indices
            indices = torch.randperm(n_samples)

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]

                # Get batch
                obs_batch = observations[batch_indices]
                actions_batch = actions[batch_indices]
                old_log_probs_batch = old_log_probs[batch_indices]
                advantages_batch = advantages[batch_indices]
                returns_batch = returns[batch_indices]

                # Evaluate actions
                log_probs, values, entropy = self.policy.evaluate_actions(obs_batch, actions_batch)

                # Policy loss (PPO clip objective)
                ratio = torch.exp(log_probs - old_log_probs_batch)
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages_batch
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (MSE)
                value_loss = nn.functional.mse_loss(values, returns_batch)

                # Entropy bonus (for exploration)
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Metrics
                with torch.no_grad():
                    approx_kl = (old_log_probs_batch - log_probs).mean().item()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_approx_kl += approx_kl
                n_updates += 1

        # Average metrics
        metrics = {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
            "approx_kl": total_approx_kl / n_updates,
        }

        return metrics

    def save(self, path):
        """Save model checkpoint."""
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
