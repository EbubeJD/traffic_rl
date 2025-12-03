import torch
import numpy as np

class RolloutBuffer:
    def __init__(self, size, obs_dim, act_dim, device="cpu"):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(size, dtype=np.float32) # Discrete action
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.device = device

    def add(self, obs, act, rew, val, logp, done):
        if self.ptr >= self.max_size:
            raise IndexError("Buffer is full")
            
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.done_buf[self.ptr] = done
        self.ptr += 1

    def finish_path(self, last_val=0, gamma=0.99, lam=0.95):
        """
        Compute GAE-Lambda and Returns.
        Call this at the end of an episode or epoch.
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        dones = np.append(self.done_buf[path_slice], 0) # Assume last step is not done if not specified
        
        # GAE
        deltas = rews[:-1] + gamma * vals[1:] * (1 - dones[:-1]) - vals[:-1]
        
        advs = np.zeros_like(deltas)
        last_gae_lam = 0
        for t in reversed(range(len(deltas))):
            last_gae_lam = deltas[t] + gamma * lam * (1 - dones[t]) * last_gae_lam
            advs[t] = last_gae_lam
            
        self.adv_buf[path_slice] = advs
        self.ret_buf[path_slice] = advs + vals[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Get all data. Reset buffer.
        """
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        
        data = dict(
            obs=self.obs_buf,
            act=self.act_buf,
            ret=self.ret_buf,
            adv=self.adv_buf,
            logp=self.logp_buf,
            val=self.val_buf
        )
        
        return {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k,v in data.items()}
