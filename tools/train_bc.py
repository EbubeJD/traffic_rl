"""
Simple behavior cloning trainer for discrete actions using transitions.csv.

It loads a transitions.csv (produced by TrafficEnv) with columns:
    decision_step,episode_id,action,reward,done,obs_json,next_obs_json
and trains a small MLP classifier obs -> action via cross-entropy.

Usage:
    python tools/train_bc.py --transitions outputs/Town10HD_Opt/tl_road5_lane-1_s10058/transitions.csv
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
    def __init__(self, transitions: List[Tuple[np.ndarray, int]]):
        self.data = transitions

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        obs, action = self.data[idx]
        return torch.from_numpy(obs), torch.tensor(action, dtype=torch.long)


class MLPPolicy(nn.Module):
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


def load_transitions_csv(path: str) -> List[Tuple[np.ndarray, int]]:
    transitions = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                action = int(row["action"])
                obs = np.array(json.loads(row["obs_json"]), dtype=np.float32)
                transitions.append((obs, action))
            except Exception:
                continue
    return transitions


def train_bc(args):
    transitions = load_transitions_csv(args.transitions)
    if not transitions:
        raise SystemExit(f"No transitions loaded from {args.transitions}")

    # Dataset statistics for optional normalization
    obs_mat = np.stack([o for o, _ in transitions], axis=0)
    obs_mean = obs_mat.mean(axis=0)
    obs_std = obs_mat.std(axis=0) + 1e-6  # avoid divide-by-zero
    if args.normalize:
        transitions = [((o - obs_mean) / obs_std, a) for o, a in transitions]

    obs_dim = transitions[0][0].shape[0]
    hidden_sizes = tuple(int(x) for x in args.hidden_sizes.split(",") if x)
    dataset = TransitionDataset(transitions)

    # Optional class weights for imbalanced labels
    weight_tensor = None
    if args.class_weighted:
        _, actions = zip(*transitions)
        counts = np.bincount(actions, minlength=args.n_actions)
        counts = np.maximum(counts, 1)
        weights = counts.sum() / (len(counts) * counts)
        weight_tensor = torch.tensor(weights, dtype=torch.float32)

    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPPolicy(obs_dim=obs_dim, hidden_sizes=hidden_sizes, n_actions=args.n_actions).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    if weight_tensor is not None:
        loss_fn = nn.CrossEntropyLoss(weight=weight_tensor.to(device))
    else:
        loss_fn = nn.CrossEntropyLoss()

    def run_epoch(loader, train: bool):
        total_loss, total, correct = 0.0, 0, 0
        model.train() if train else model.eval()
        with torch.set_grad_enabled(train):
            for obs, act in loader:
                obs = obs.to(device)
                act = act.to(device)
                logits = model(obs)
                loss = loss_fn(logits, act)
                if train:
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                total_loss += loss.item() * obs.size(0)
                pred = logits.argmax(dim=1)
                correct += (pred == act).sum().item()
                total += obs.size(0)
        return total_loss / total, correct / total

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(train_loader, train=True)
        val_loss, val_acc = run_epoch(val_loader, train=False)
        if epoch % args.log_every == 0:
            print(f"[epoch {epoch:03d}] train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        torch.save(model.state_dict(), args.out)
        print(f"Saved BC policy to {args.out}")


def main():
    parser = argparse.ArgumentParser(description="Behavior cloning on transitions.csv")
    parser.add_argument("--transitions", required=True, help="Path to transitions.csv")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n_actions", type=int, default=2)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--out", type=str, default="bc_policy.pt", help="Where to save the trained policy")
    parser.add_argument("--hidden_sizes", type=str, default="64,64", help="Comma-separated hidden layer sizes")
    parser.add_argument("--normalize", action="store_true", help="Normalize observations using dataset mean/std")
    parser.add_argument("--class_weighted", action="store_true", help="Use inverse-frequency class weights")
    args = parser.parse_args()
    train_bc(args)


if __name__ == "__main__":
    main()
