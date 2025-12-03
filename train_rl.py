import os
import time
import numpy as np
import torch
import csv
from datetime import datetime

from traffic_rl.config import (
    LR, GAMMA, LAMBDA, CLIP_RANGE, ENTROPY_COEF, VALUE_COEF,
    UPDATE_EPOCHS, BATCH_SIZE, TOTAL_TIMESTEPS, STEPS_PER_EPOCH,
    SAVE_INTERVAL, SEED, DEVICE, LOG_DIR, CHECKPOINT_DIR
)
from traffic_rl.env.traffic_runner import TrafficRunner
from traffic_rl.env.traffic_env import TrafficEnv
from traffic_rl.agent.ppo import PPOAgent
from traffic_rl.agent.buffer import RolloutBuffer

def train():
    # Setup Paths
    run_name = f"ppo_traffic_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(LOG_DIR, run_name)
    ckpt_dir = os.path.join(run_dir, CHECKPOINT_DIR)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Logging
    log_f = open(os.path.join(run_dir, "log.csv"), "w", newline="")
    writer = csv.writer(log_f)
    writer.writerow(["epoch", "total_steps", "ep_ret_mean", "ep_len_mean", "loss_pi", "loss_v", "kl", "fps"])
    
    # Env & Agent
    runner = TrafficRunner()
    env = TrafficEnv(runner)
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    agent = PPOAgent(obs_dim, act_dim, lr=LR, gamma=GAMMA, clip_ratio=CLIP_RANGE,
                     entropy_coef=ENTROPY_COEF, value_coef=VALUE_COEF, 
                     train_iters=UPDATE_EPOCHS, device=DEVICE)
    
    buffer = RolloutBuffer(STEPS_PER_EPOCH, obs_dim, act_dim, device=DEVICE)
    
    # Training Loop
    obs = env.reset()
    ep_ret, ep_len = 0, 0
    ep_rets = []
    total_steps = 0
    
    start_time = time.time()
    
    num_epochs = TOTAL_TIMESTEPS // STEPS_PER_EPOCH
    
    try:
        for epoch in range(num_epochs):
            for t in range(STEPS_PER_EPOCH):
                # Select Action
                action, logp, val = agent.step(obs)
                
                # Step Env
                next_obs, reward, done, info = env.step(action)
                
                # Store
                buffer.add(obs, action, reward, val, logp, done)
                
                obs = next_obs
                ep_ret += reward
                ep_len += 1
                total_steps += 1
                
                # Handle Done (Simulated or Timeout)
                # Since our env is continuous, we might not get 'done' often
                # But if we did:
                if done: # or timeout
                    ep_rets.append(ep_ret)
                    obs = env.reset()
                    ep_ret, ep_len = 0, 0
            
            # Finish Path
            _, _, last_val = agent.step(obs)
            buffer.finish_path(last_val, GAMMA, LAMBDA)
            
            # Update
            loss_pi, loss_v, kl = agent.update(buffer)
            
            # Log
            fps = int(STEPS_PER_EPOCH / (time.time() - start_time))
            start_time = time.time()
            
            avg_ret = np.mean(ep_rets) if ep_rets else ep_ret # Use current if no full ep
            print(f"Epoch {epoch}: Ret={avg_ret:.2f}, LossPi={loss_pi:.4f}, FPS={fps}")
            
            writer.writerow([epoch, total_steps, avg_ret, ep_len, loss_pi, loss_v, kl, fps])
            log_f.flush()
            
            # Save
            if (epoch + 1) % SAVE_INTERVAL == 0:
                agent.save(os.path.join(ckpt_dir, f"model_{epoch}.pt"))
                
    except KeyboardInterrupt:
        print("Training interrupted.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Closing environment...")
        env.runner.close()
        log_f.close()

if __name__ == "__main__":
    train()
