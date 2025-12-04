"""
Evaluate a BC policy (bc_policy.pt) in the live TrafficEnv and report congestion metrics.

Usage:
  python tools/eval_bc.py --policy bc_policy.pt --steps 500
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from traffic_rl.env.traffic_env import TrafficEnv
from traffic_rl.env.traffic_runner import TrafficRunner


class BCPolicy(torch.nn.Module):
    def __init__(self, obs_dim, hidden_sizes=(64, 64), n_actions=2):
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden_sizes:
            layers += [torch.nn.Linear(last, h), torch.nn.ReLU()]
            last = h
        layers.append(torch.nn.Linear(last, n_actions))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def act(self, obs: np.ndarray) -> int:
        with torch.no_grad():
            x = torch.from_numpy(obs).float().unsqueeze(0)
            logits = self.forward(x)
            return int(torch.argmax(logits, dim=1).item())


def main():
    parser = argparse.ArgumentParser(description="Evaluate BC policy in live TrafficEnv.")
    parser.add_argument("--policy", required=True, help="Path to bc_policy.pt")
    parser.add_argument("--steps", type=int, default=500, help="Number of decision steps to run")
    args = parser.parse_args()

    runner = TrafficRunner()
    env = TrafficEnv(runner, debug_obs=False)

    obs = env.reset()
    policy = BCPolicy(obs_dim=obs.shape[0])
    policy.load_state_dict(torch.load(args.policy, map_location="cpu"))
    policy.eval()

    rewards = []
    queues = []
    waits = []
    long_waits = []

    try:
        for i in range(args.steps):
            action = policy.act(obs)
            obs, reward, done, info = env.step(action)

            rewards.append(reward)
            queues.append(info.get("queue_count", 0.0))
            waits.append(info.get("avg_wait", 0.0))
            long_waits.append(info.get("num_long_wait_60s", 0.0))

            if done:
                obs = env.reset()
    finally:
        env.close()

    print(f"Ran {len(rewards)} steps.")
    print(f"Reward: mean={np.mean(rewards):.2f}, min={np.min(rewards):.2f}, max={np.max(rewards):.2f}")
    print(f"Queue: mean={np.mean(queues):.2f}, max={np.max(queues):.2f}")
    print(f"Avg wait: mean={np.mean(waits):.2f}, max={np.max(waits):.2f}")
    print(f"Long waiters: mean={np.mean(long_waits):.2f}, max={np.max(long_waits):.2f}")


if __name__ == "__main__":
    main()
