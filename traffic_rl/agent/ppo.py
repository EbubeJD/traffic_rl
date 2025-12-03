import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, act_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, obs):
        logits = self.actor(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        val = self.critic(obs)
        return action, logp, val

    def get_logp_val(self, obs, act):
        logits = self.actor(obs)
        dist = Categorical(logits=logits)
        logp = dist.log_prob(act)
        entropy = dist.entropy()
        val = self.critic(obs)
        return logp, val, entropy

class PPOAgent:
    def __init__(self, obs_dim, act_dim, lr=3e-4, gamma=0.99, clip_ratio=0.2, 
                 target_kl=0.01, entropy_coef=0.01, value_coef=0.5, 
                 train_iters=10, device="cpu"):
        self.device = device
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.train_iters = train_iters
        
        self.ac = ActorCritic(obs_dim, act_dim).to(device)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr)

    def step(self, obs):
        with torch.no_grad():
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            action, logp, val = self.ac.act(obs)
        return action.item(), logp.item(), val.item()

    def update(self, buffer):
        data = buffer.get()
        obs = data['obs']
        act = data['act']
        ret = data['ret']
        adv = data['adv']
        logp_old = data['logp']
        
        # Normalize advantage
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        
        for i in range(self.train_iters):
            self.optimizer.zero_grad()
            
            logp, val, entropy = self.ac.get_logp_val(obs, act)
            
            # Policy Loss
            ratio = torch.exp(logp - logp_old)
            clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv
            loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()
            
            # Value Loss
            loss_v = ((val.squeeze() - ret)**2).mean()
            
            # Entropy Loss
            loss_ent = -entropy.mean()
            
            # Total Loss
            loss = loss_pi + self.value_coef * loss_v + self.entropy_coef * loss_ent
            
            # Approximate KL for early stopping
            with torch.no_grad():
                kl = (logp_old - logp).mean().item()
            if kl > 1.5 * self.target_kl:
                # print(f"Early stopping at iter {i} due to KL")
                break
                
            loss.backward()
            self.optimizer.step()
            
        return loss_pi.item(), loss_v.item(), kl

    def save(self, path):
        torch.save(self.ac.state_dict(), path)

    def load(self, path):
        self.ac.load_state_dict(torch.load(path, map_location=self.device))
