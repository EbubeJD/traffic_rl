"""
Discrete CQL trainer for offline traffic control data.

Expects transitions.csv with columns:
decision_step,episode_id,action,reward,done,obs_json,next_obs_json

Usage:
  python tools/train_cql.py --transitions outputs/Town10HD_Opt/tl_road5_lane-1_s10058/transitions.csv
"""

import argparse
import csv
import json
import os
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split


class TransitionDataset(Dataset):
    def __init__(self, data: List[Tuple[np.ndarray, int, float, np.ndarray, float]]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        obs, action, reward, next_obs, done = self.data[idx]
        return (
            torch.from_numpy(obs),
            torch.tensor(action, dtype=torch.long),
            torch.tensor(reward, dtype=torch.float32),
            torch.from_numpy(next_obs),
            torch.tensor(done, dtype=torch.float32),
        )


def load_transitions(path: str) -> List[Tuple[np.ndarray, int, float, np.ndarray, float]]:
    out = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                action = int(row["action"])
                reward = float(row["reward"])
                done = float(row["done"])
                obs = np.array(json.loads(row["obs_json"]), dtype=np.float32)
                next_obs = np.array(json.loads(row["next_obs_json"]), dtype=np.float32)
                out.append((obs, action, reward, next_obs, done))
            except Exception:
                continue
    return out


class QNet(nn.Module):
    def __init__(self, obs_dim: int, hidden_sizes=(128, 128), n_actions: int = 2):
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, hidden_sizes=(64, 64), n_actions: int = 2):
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_cql(args):
    transitions = load_transitions(args.transitions)
    if not transitions:
        raise SystemExit(f"No transitions found in {args.transitions}")

    obs_dim = transitions[0][0].shape[0]
    dataset = TransitionDataset(transitions)
    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q1 = QNet(obs_dim, n_actions=args.n_actions).to(device)
    q2 = QNet(obs_dim, n_actions=args.n_actions).to(device)
    q1_tgt = QNet(obs_dim, n_actions=args.n_actions).to(device)
    q2_tgt = QNet(obs_dim, n_actions=args.n_actions).to(device)
    q1_tgt.load_state_dict(q1.state_dict())
    q2_tgt.load_state_dict(q2.state_dict())

    opt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=args.lr)
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    actor = PolicyNet(obs_dim, n_actions=args.n_actions).to(device)
    if args.bc_init:
        try:
            state_dict = torch.load(args.bc_init, map_location=device)
            missing, unexpected = actor.load_state_dict(state_dict, strict=False)
            print(f"Loaded BC weights from {args.bc_init} (missing={missing}, unexpected={unexpected})")
        except Exception as e:
            print(f"Warning: failed to load BC weights from {args.bc_init}: {e}")
    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)

    def step(loader, train: bool):
        total_loss, total = 0.0, 0
        q1.eval(); q2.eval(); actor.eval()
        if train:
            q1.train(); q2.train(); actor.train()
        with torch.set_grad_enabled(train):
            for obs, act, rew, next_obs, done in loader:
                obs = obs.to(device)
                next_obs = next_obs.to(device)
                act = act.to(device)
                rew = rew.to(device)
                done = done.to(device)

                q1_vals = q1(obs)
                q2_vals = q2(obs)

                with torch.no_grad():
                    next_q1 = q1_tgt(next_obs)
                    next_q2 = q2_tgt(next_obs)
                    next_q = torch.min(next_q1, next_q2)
                    next_v, _ = next_q.max(dim=1)
                    target = rew + args.gamma * (1 - done) * next_v

                q1_sa = q1_vals.gather(1, act.view(-1, 1)).squeeze(1)
                q2_sa = q2_vals.gather(1, act.view(-1, 1)).squeeze(1)
                bellman_loss = mse(q1_sa, target) + mse(q2_sa, target)

                # CQL conservative loss: logsumexp over actions minus Q(s,a_data)
                logsumexp_q1 = torch.logsumexp(q1_vals, dim=1)
                logsumexp_q2 = torch.logsumexp(q2_vals, dim=1)
                conservative_loss = (logsumexp_q1 - q1_sa).mean() + (logsumexp_q2 - q2_sa).mean()

                loss = bellman_loss + args.cql_alpha * conservative_loss

                if train:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                    # Actor update: maximize expected Q while staying close to behavior (BC)
                    logits = actor(obs)
                    probs = torch.softmax(logits, dim=1)
                    with torch.no_grad():
                        q_policy = torch.min(q1(obs), q2(obs))
                    expected_q = (probs * q_policy).sum(dim=1).mean()
                    bc_loss = ce(logits, act)
                    policy_loss = -expected_q + args.policy_bc_weight * bc_loss
                    actor_opt.zero_grad()
                    policy_loss.backward()
                    actor_opt.step()

                total_loss += loss.item() * obs.size(0)
                total += obs.size(0)
        return total_loss / max(total, 1)

    for epoch in range(1, args.epochs + 1):
        train_loss = step(train_loader, train=True)
        val_loss = step(val_loader, train=False)
        if epoch % args.log_every == 0:
            print(f"[epoch {epoch:03d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        # Soft update targets
        with torch.no_grad():
            for p, tp in zip(q1.parameters(), q1_tgt.parameters()):
                tp.data.mul_(1 - args.tau).add_(args.tau * p.data)
            for p, tp in zip(q2.parameters(), q2_tgt.parameters()):
                tp.data.mul_(1 - args.tau).add_(args.tau * p.data)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        torch.save({"q1": q1.state_dict(), "q2": q2.state_dict(), "actor": actor.state_dict()}, args.out)
        print(f"Saved CQL critics to {args.out}")
    if args.actor_out:
        os.makedirs(os.path.dirname(args.actor_out) or ".", exist_ok=True)
        torch.save(actor.state_dict(), args.actor_out)
        print(f"Saved CQL actor to {args.actor_out}")


def main():
    parser = argparse.ArgumentParser(description="Discrete CQL trainer for transitions.csv")
    parser.add_argument("--transitions", required=True, help="Path to transitions.csv")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--cql_alpha", type=float, default=1.0, help="Weight on conservative term")
    parser.add_argument("--tau", type=float, default=0.005, help="Target soft-update rate")
    parser.add_argument("--n_actions", type=int, default=2)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--out", type=str, default="cql_critics.pt")
    parser.add_argument("--actor_lr", type=float, default=3e-4, help="Learning rate for policy")
    parser.add_argument("--policy_bc_weight", type=float, default=1.0, help="Weight on BC regularizer for actor")
    parser.add_argument("--bc_init", type=str, default=None, help="Path to BC policy weights to initialize actor")
    parser.add_argument("--actor_out", type=str, default="cql_actor.pt", help="Where to save the learned actor")
    args = parser.parse_args()
    train_cql(args)


if __name__ == "__main__":
    main()
