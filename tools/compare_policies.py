"""
Multi-policy comparison and evaluation tool.

Runs multiple policies (random, fixed-time, heuristic, BC, CQL) in TrafficEnv
and collects comprehensive congestion metrics for comparison.

Usage:
    python tools/compare_policies.py \
        --policies random,fixed,heuristic,bc,cql \
        --episodes 10 \
        --bc_model bc_policy_v2.pt \
        --cql_model cql_actor_v2.pt \
        --out comparison_results.json
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
import numpy as np
import torch

from traffic_rl.env.traffic_runner import TrafficRunner
from traffic_rl.env.traffic_env import TrafficEnv
from policies.queue_aware_heuristic import MaxQueuePolicy


class RandomPolicy:
    """Random action selection baseline."""
    def __init__(self, n_actions=2):
        self.n_actions = n_actions

    def act(self, obs, info=None):
        return np.random.randint(0, self.n_actions)

    def reset(self):
        pass


class FixedTimePolicy:
    """Classic fixed-cycle controller (alternates every N steps)."""
    def __init__(self, cycle_length=5):
        self.cycle_length = cycle_length
        self.step_count = 0

    def act(self, obs, info=None):
        phase = (self.step_count // self.cycle_length) % 2
        self.step_count += 1
        return phase

    def reset(self):
        self.step_count = 0


class BCPolicy:
    """Behavior cloning policy (MLP classifier)."""
    def __init__(self, model_path, obs_dim=9, n_actions=2):
        from tools.train_bc import MLPPolicy

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MLPPolicy(obs_dim=obs_dim, hidden_sizes=(64, 64), n_actions=n_actions)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def act(self, obs, info=None):
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)
            logits = self.model(obs_tensor)
            action = logits.argmax(dim=1).item()
        return action

    def reset(self):
        pass


class CQLPolicy:
    """CQL actor policy."""
    def __init__(self, model_path, obs_dim=9, n_actions=2):
        from tools.train_cql import PolicyNet

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PolicyNet(obs_dim=obs_dim, hidden_sizes=(64, 64), n_actions=n_actions)

        # Load from checkpoint (may contain multiple keys)
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "actor" in checkpoint:
            self.model.load_state_dict(checkpoint["actor"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

    def act(self, obs, info=None):
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)
            logits = self.model(obs_tensor)
            action = logits.argmax(dim=1).item()
        return action

    def reset(self):
        pass


class PolicyEvaluator:
    """Evaluate a policy and collect metrics."""

    def run_policy(self, policy, policy_name, env, episodes=10, steps=120):
        """
        Run policy for multiple episodes and collect metrics.

        Returns:
            results: {
                "episode_rewards": [...],
                "episode_queues": [...],
                "episode_waits": [...],
                "episode_long_waits": [...],
                "summary": {
                    "reward_mean": float,
                    "queue_mean": float,
                    ...
                }
            }
        """
        print(f"\n{'='*80}")
        print(f"Evaluating: {policy_name}")
        print(f"{'='*80}")

        episode_rewards = []
        episode_queues = []
        episode_waits = []
        episode_long_waits = []
        episode_actions = []

        for episode in range(episodes):
            obs = env.reset()
            policy.reset()

            ep_reward = 0.0
            ep_queues = []
            ep_waits = []
            ep_long_waits = []
            ep_actions = []

            for step in range(steps):
                action = policy.act(obs)
                obs, reward, done, info = env.step(action)

                ep_reward += reward
                ep_queues.append(info.get("queue_count", 0.0))
                ep_waits.append(info.get("avg_wait", 0.0))
                ep_long_waits.append(info.get("num_long_wait_60s", 0.0))
                ep_actions.append(action)

                if done:
                    break

            episode_rewards.append(ep_reward)
            episode_queues.append(ep_queues)
            episode_waits.append(ep_waits)
            episode_long_waits.append(ep_long_waits)
            episode_actions.append(ep_actions)

            print(f"  Episode {episode+1}/{episodes}: reward={ep_reward:.2f}, "
                  f"avg_queue={np.mean(ep_queues):.2f}, avg_wait={np.mean(ep_waits):.2f}")

        # Compute summary statistics
        all_queues = [q for ep in episode_queues for q in ep]
        all_waits = [w for ep in episode_waits for w in ep]
        all_long_waits = [lw for ep in episode_long_waits for lw in ep]

        # Action statistics
        all_actions = [a for ep in episode_actions for a in ep]
        action_changes = sum(1 for i in range(1, len(all_actions)) if all_actions[i] != all_actions[i-1])
        alternation_rate = (action_changes / max(len(all_actions) - 1, 1)) * 100
        phase_0_fraction = np.mean([a == 0 for a in all_actions]) * 100

        summary = {
            "reward_mean": float(np.mean(episode_rewards)),
            "reward_median": float(np.median(episode_rewards)),
            "reward_min": float(np.min(episode_rewards)),
            "queue_mean": float(np.mean(all_queues)),
            "queue_median": float(np.median(all_queues)),
            "queue_p95": float(np.percentile(all_queues, 95)),
            "queue_max": float(np.max(all_queues)),
            "wait_mean": float(np.mean(all_waits)),
            "wait_median": float(np.median(all_waits)),
            "wait_p95": float(np.percentile(all_waits, 95)),
            "wait_max": float(np.max(all_waits)),
            "long_wait_mean": float(np.mean(all_long_waits)),
            "long_wait_max": float(np.max(all_long_waits)),
            "alternation_rate": float(alternation_rate),
            "phase_0_fraction": float(phase_0_fraction),
        }

        print(f"\nSummary:")
        print(f"  Reward: mean={summary['reward_mean']:.2f}, min={summary['reward_min']:.2f}")
        print(f"  Queue: mean={summary['queue_mean']:.2f}, p95={summary['queue_p95']:.2f}, max={summary['queue_max']:.2f}")
        print(f"  Wait: mean={summary['wait_mean']:.2f}, p95={summary['wait_p95']:.2f}")
        print(f"  Long waiters: mean={summary['long_wait_mean']:.2f}, max={summary['long_wait_max']:.2f}")

        return {
            "episode_rewards": episode_rewards,
            "episode_queues": episode_queues,
            "episode_waits": episode_waits,
            "episode_long_waits": episode_long_waits,
            "summary": summary
        }


def main():
    parser = argparse.ArgumentParser(description="Compare multiple traffic policies")
    parser.add_argument("--policies", type=str, default="random,fixed,heuristic",
                        help="Comma-separated list of policies to evaluate")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes per policy")
    parser.add_argument("--steps", type=int, default=120,
                        help="Steps per episode")
    parser.add_argument("--bc_model", type=str, default=None,
                        help="Path to BC policy model (.pt file)")
    parser.add_argument("--cql_model", type=str, default=None,
                        help="Path to CQL actor model (.pt file)")
    parser.add_argument("--out", type=str, default="comparison_results.json",
                        help="Output JSON file for results")
    args = parser.parse_args()

    policy_names = [p.strip() for p in args.policies.split(",")]

    print("="*80)
    print("MULTI-POLICY COMPARISON")
    print("="*80)
    print(f"Policies: {policy_names}")
    print(f"Episodes per policy: {args.episodes}")
    print(f"Steps per episode: {args.steps}")

    # Initialize environment (reused for all policies)
    print(f"\nInitializing TrafficRunner...")
    runner = TrafficRunner()
    env = TrafficEnv(runner, episode_steps=args.steps)

    # Evaluator
    evaluator = PolicyEvaluator()

    # Run each policy
    all_results = {}

    for policy_name in policy_names:
        # Instantiate policy
        if policy_name == "random":
            policy = RandomPolicy()
        elif policy_name == "fixed":
            policy = FixedTimePolicy(cycle_length=5)
        elif policy_name == "heuristic":
            policy = MaxQueuePolicy(min_green_time=10.0, queue_threshold=0.5, dt=2.0)
        elif policy_name == "bc":
            if not args.bc_model:
                print(f"\n⚠ Skipping BC: --bc_model not provided")
                continue
            try:
                policy = BCPolicy(args.bc_model)
            except Exception as e:
                print(f"\n⚠ Failed to load BC model: {e}")
                continue
        elif policy_name == "cql":
            if not args.cql_model:
                print(f"\n⚠ Skipping CQL: --cql_model not provided")
                continue
            try:
                policy = CQLPolicy(args.cql_model)
            except Exception as e:
                print(f"\n⚠ Failed to load CQL model: {e}")
                continue
        else:
            print(f"\n⚠ Unknown policy: {policy_name}, skipping")
            continue

        # Evaluate
        results = evaluator.run_policy(policy, policy_name, env, args.episodes, args.steps)
        all_results[policy_name] = results

    # Close environment
    env.close()

    # Save results
    with open(args.out, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"COMPARISON COMPLETE")
    print(f"{'='*80}")
    print(f"✓ Results saved to: {args.out}")
    print(f"\nNext steps:")
    print(f"  Generate report:")
    print(f"    python tools/generate_report.py --results {args.out} --baseline random --out comparison_report.md")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
