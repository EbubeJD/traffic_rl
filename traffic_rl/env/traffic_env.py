import gym
import numpy as np
import carla
from gym import spaces
import time

from traffic_rl.env.traffic_runner import TrafficRunner
from traffic_rl.config import ACTION_DURATION_SEC, YELLOW_DURATION_SEC, REWARD_WEIGHTS, DT

class TrafficEnv(gym.Env):
    def __init__(self, runner: TrafficRunner):
        self.runner = runner
        self.action_space = spaces.Discrete(2) # 0: Phase A, 1: Phase B
        
        # Observation: [Queue_i, Wait_i, Phase_i] for each TL
        # Assuming max 4 TLs for now, can be dynamic
        self.num_tls = len(self.runner.group["actors"])
        self.obs_dim = self.num_tls * 3 + 1 # +1 for current phase index
        self.observation_space = spaces.Box(low=0, high=999, shape=(self.obs_dim,), dtype=np.float32)
        
        self.current_phase = 0
        self.tl_actors = self.runner.group["actors"]
        self.road_groups = self._group_tls_by_road()
        
    def _group_tls_by_road(self):
        # Heuristic: Group by Road ID to determine phases
        # We assume 2 main phases (Road Group A vs Road Group B)
        roads = {}
        for tl in self.tl_actors:
            # Extract road ID from stable_id (e.g., road12_lane1_s500)
            # We need to get stable_id from the observer matching this TL
            # But runner has observers. Let's map TL ID to Road ID.
            # For now, let's just use the order in the list and split in half.
            # This is a simplification.
            pass
        
        # Simple split: First half = Phase 0, Second half = Phase 1
        mid = len(self.tl_actors) // 2
        return [self.tl_actors[:mid], self.tl_actors[mid:]]

    def reset(self):
        metrics = self.runner.reset()
        self.current_phase = 0
        self._apply_phase(0) # Start with Phase 0
        return self._get_obs(metrics)

    def step(self, action):
        # Action is desired phase (0 or 1)
        # If action != current_phase, we need to switch
        
        info = {}
        total_reward = 0
        
        if action != self.current_phase:
            # Transition: Yellow for current Green lights
            self._set_lights(self.current_phase, carla.TrafficLightState.Yellow)
            self._set_lights(action, carla.TrafficLightState.Red)
            
            # Tick for Yellow duration
            steps = int(YELLOW_DURATION_SEC / DT)
            for _ in range(steps):
                metrics = self.runner.step(self._get_light_states())
            
            self.current_phase = action
            
        # Set new phase
        self._apply_phase(self.current_phase)
        
        # Hold phase for Action Duration
        steps = int(ACTION_DURATION_SEC / DT)
        step_metrics = []
        for _ in range(steps):
            m = self.runner.step(self._get_light_states())
            step_metrics.append(m)
            
        # Aggregate metrics for reward/obs
        last_metrics = step_metrics[-1] if step_metrics else {}
        reward = self._compute_reward(step_metrics)
        obs = self._get_obs(last_metrics)
        done = False # Continuous task
        
        info["metrics"] = last_metrics
        return obs, reward, done, info

    def _apply_phase(self, phase_idx):
        # Phase idx Green, others Red
        for i, group in enumerate(self.road_groups):
            state = carla.TrafficLightState.Green if i == phase_idx else carla.TrafficLightState.Red
            for tl in group:
                self._light_states[tl.id] = state
                
    def _set_lights(self, group_idx, state):
        for tl in self.road_groups[group_idx]:
            self._light_states[tl.id] = state

    def _get_light_states(self):
        if not hasattr(self, "_light_states"):
            self._light_states = {tl.id: carla.TrafficLightState.Red for tl in self.tl_actors}
        return self._light_states

    def _get_obs(self, metrics):
        # Vectorize metrics
        obs = []
        for ob in self.runner.observers:
            sid = ob.stable_id
            if sid in metrics:
                m = metrics[sid]
                obs.extend([m['queue'], m['queue_ema'], m['time_in_state']])
            else:
                obs.extend([0, 0, 0])
        obs.append(self.current_phase)
        return np.array(obs, dtype=np.float32)

    def _compute_reward(self, step_metrics_list):
        # Average reward over the steps
        # Reward = - (w_wait * avg_wait + w_queue * queue + ...)
        
        total_r = 0
        count = 0
        
        for metrics in step_metrics_list:
            step_r = 0
            # Sum over all TLs
            for sid, m in metrics.items():
                r_wait = REWARD_WEIGHTS["wait"] * m.get('avg_wait', 0)
                r_queue = REWARD_WEIGHTS["queue"] * m['queue']
                step_r -= (r_wait + r_queue)
                
            total_r += step_r
            count += 1
            
        return total_r / max(1, count)
