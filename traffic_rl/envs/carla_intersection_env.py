"""
Gymnasium environment for CARLA-based traffic signal control.

This environment wraps CARLA simulation to provide a standard Gym interface
for reinforcement learning agents to control traffic lights at an intersection.
"""

import sys
import os
import random
from typing import Optional, Dict, Tuple, Any, List

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import carla

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.carla_helpers import get_tl_groups, tl_stable_id, spawn_autopilot_vehicles
from observers.tl_observer import TLObserver
from traffic_rl.reward.traffic_reward import compute_traffic_reward
from traffic_rl.envs.phase_controller import PhaseController


class CarlaIntersectionEnv(gym.Env):
    """
    Gymnasium environment for single-intersection traffic signal control in CARLA.

    Observation:
        Box(0, 1, shape=(11*n_lights,)) - Normalized traffic metrics per light

    Actions:
        Discrete(2):
            0 = Keep current phase
            1 = Switch to next phase

    Reward:
        Negative congestion: -(alpha*queue + beta*wait + gamma*long_wait + delta*switch)

    Episode Termination:
        truncated=True when episode_time >= episode_length_sec
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        carla_host: str = "localhost",
        carla_port: int = 2000,
        town: str = "Town10HD_Opt",
        group_index: int = 0,
        num_vehicles: int = 40,
        episode_length_sec: float = 180.0,
        decision_interval_sec: float = 5.0,
        min_green_time_sec: float = 10.0,
        warmup_time_sec: float = 20.0,
        dt: float = 0.05,
        reward_config: Optional[Dict[str, float]] = None,
        phase_config: Optional[Dict[str, Dict]] = None,
    ):
        """
        Initialize CARLA intersection environment.

        Args:
            carla_host: CARLA server host
            carla_port: CARLA server port
            town: CARLA town/map name
            group_index: Which traffic light group to control
            num_vehicles: Number of NPC vehicles to spawn
            episode_length_sec: Episode duration in seconds
            decision_interval_sec: Time between agent decisions
            min_green_time_sec: Minimum green time before phase can switch
            warmup_time_sec: Warm-up period to populate queues
            dt: CARLA simulation timestep
            reward_config: Reward weight configuration
            phase_config: Traffic light phase configuration
        """
        super().__init__()

        # Store configuration
        self.carla_host = carla_host
        self.carla_port = carla_port
        self.town = town
        self.group_index = group_index
        self.num_vehicles = num_vehicles
        self.episode_length_sec = episode_length_sec
        self.decision_interval_sec = decision_interval_sec
        self.decision_interval_ticks = int(decision_interval_sec / dt)
        self.min_green_time_sec = min_green_time_sec
        self.warmup_time_sec = warmup_time_sec
        self.warmup_ticks = int(warmup_time_sec / dt)
        self.dt = dt

        # Reward configuration
        from traffic_rl.reward.traffic_reward import DEFAULT_REWARD_CONFIG
        self.reward_config = reward_config or DEFAULT_REWARD_CONFIG

        # Phase configuration
        self.phase_config = phase_config
        if self.phase_config is None:
            raise ValueError(
                "phase_config is required. "
                "Run 'python tools/print_tl_info.py' and fill in traffic_rl/rl/config.py"
            )

        # CARLA connection (lazy initialization in reset)
        self.client: Optional[carla.Client] = None
        self.world: Optional[carla.World] = None
        self.tl_group: Optional[Dict] = None
        self.tl_actors: List[carla.TrafficLight] = []
        self.observers: List[TLObserver] = []
        self.vehicles: List[carla.Vehicle] = []

        # Phase controller
        self.phase_controller: Optional[PhaseController] = None
        self.stable_id_map: Dict[int, str] = {}  # actor.id -> stable_id

        # Episode state
        self.current_phase = 0
        self.time_in_current_phase = 0.0
        self.episode_step = 0
        self.episode_time = 0.0
        self.total_ticks = 0

        # Gym spaces (defined after first reset when we know n_lights)
        self.observation_space: Optional[spaces.Box] = None
        self.action_space = spaces.Discrete(2)  # 0=Stay, 1=Switch

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment for a new episode.

        Args:
            seed: Random seed for reproducibility
            options: Additional reset options

        Returns:
            observation: Initial observation
            info: Additional information
        """
        # Set random seeds
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Connect to CARLA (first time only)
        if self.client is None:
            print(f"[ENV] Connecting to CARLA at {self.carla_host}:{self.carla_port}...")
            try:
                self.client = carla.Client(self.carla_host, self.carla_port)
                self.client.set_timeout(10.0)
                print(f"[ENV] Connected to CARLA")
            except Exception as e:
                raise RuntimeError(f"Failed to connect to CARLA: {e}")

            # Load world
            print(f"[ENV] Loading world '{self.town}'...")
            self.world = self.client.load_world(self.town, map_layers=carla.MapLayer.NONE)

            # Set synchronous mode
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = self.dt
            self.world.apply_settings(settings)
            print(f"[ENV] World loaded in synchronous mode (dt={self.dt}s)")

        # Cleanup previous episode
        self._cleanup()

        # Discover traffic light groups
        groups = get_tl_groups(self.world)
        if not groups:
            raise RuntimeError("No traffic light groups found in world")
        if self.group_index >= len(groups):
            raise IndexError(
                f"GROUP_INDEX {self.group_index} out of range. "
                f"Found {len(groups)} group(s)"
            )

        self.tl_group = groups[self.group_index]

        # Sort traffic lights by stable_id for consistent ordering
        self.tl_actors = sorted(
            self.tl_group["actors"],
            key=lambda tl: tl_stable_id(self.world, tl)
        )

        # Create stable_id map
        self.stable_id_map = {
            tl.id: tl_stable_id(self.world, tl)
            for tl in self.tl_actors
        }

        print(f"[ENV] Selected group {self.group_index} with {len(self.tl_actors)} traffic light(s)")
        for tl in self.tl_actors:
            print(f"      - {self.stable_id_map[tl.id]}")

        # Define observation space (now that we know n_lights)
        if self.observation_space is None:
            n_lights = len(self.tl_actors)
            obs_dim = 11 * n_lights
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(obs_dim,),
                dtype=np.float32
            )
            print(f"[ENV] Observation space: Box(0, 1, shape=({obs_dim},))")

        # Initialize phase controller
        self.phase_controller = PhaseController(self.phase_config)
        print(f"[ENV] Phase controller initialized with {self.phase_controller.num_phases} phases")

        # Create TLObservers
        bp_lib = self.world.get_blueprint_library()
        for tl in self.tl_actors:
            try:
                observer = TLObserver(self.world, tl, bp_lib)
                self.observers.append(observer)
            except Exception as e:
                print(f"[WARN] Failed to create observer for {self.stable_id_map[tl.id]}: {e}")

        print(f"[ENV] Created {len(self.observers)} observer(s)")

        # Spawn vehicles
        print(f"[ENV] Spawning {self.num_vehicles} NPC vehicles...")
        self.vehicles = spawn_autopilot_vehicles(self.world, self.client, self.num_vehicles)
        print(f"[ENV] Spawned {len(self.vehicles)} vehicle(s)")

        # Initialize phase state
        self.current_phase = 0
        self.time_in_current_phase = 0.0
        self.episode_step = 0
        self.episode_time = 0.0
        self.total_ticks = 0

        # Apply initial phase
        self.phase_controller.apply_phase(
            self.current_phase,
            self.tl_actors,
            self.stable_id_map
        )
        print(f"[ENV] Applied initial phase {self.current_phase}")

        # Warm-up period to populate queues
        print(f"[ENV] Running {self.warmup_time_sec}s warm-up...")
        for _ in range(self.warmup_ticks):
            self.world.tick()
            frame = self.world.get_snapshot().frame
            for obs in self.observers:
                try:
                    obs.get(frame)
                except Exception as e:
                    pass  # Ignore errors during warm-up

        print(f"[ENV] Warm-up complete, starting episode")

        # Build initial observation
        obs = self._get_observation()

        info = {
            "group_id": list(self.tl_group["ids"]),
            "stable_ids": [self.stable_id_map[tl.id] for tl in self.tl_actors],
            "episode": {
                "step": 0,
                "time_sec": 0.0,
            },
        }

        return obs, info

    def step(
        self,
        action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one environment step (decision_interval_sec of simulation).

        Args:
            action: 0=keep phase, 1=switch phase

        Returns:
            observation: Next observation
            reward: Step reward
            terminated: Whether episode naturally ended (always False)
            truncated: Whether episode time limit reached
            info: Additional information
        """
        phase_changed = False

        # Handle action
        if action == 1:  # Switch requested
            if self.time_in_current_phase >= self.min_green_time_sec:
                # Apply yellow/all-red transition
                self.phase_controller.apply_yellow(self.tl_actors)

                # Advance simulation during yellow period (3 seconds)
                yellow_ticks = int(3.0 / self.dt)
                for _ in range(yellow_ticks):
                    self.world.tick()
                    frame = self.world.get_snapshot().frame
                    self.total_ticks += 1
                    for obs in self.observers:
                        try:
                            obs.get(frame)
                        except Exception as e:
                            pass

                # Switch to next phase
                self.current_phase = self.phase_controller.get_next_phase_id(self.current_phase)
                self.phase_controller.apply_phase(
                    self.current_phase,
                    self.tl_actors,
                    self.stable_id_map
                )
                self.time_in_current_phase = 0.0
                phase_changed = True
            # else: ignore action due to min green constraint

        # Advance simulation for decision interval
        for _ in range(self.decision_interval_ticks):
            self.world.tick()
            frame = self.world.get_snapshot().frame
            self.total_ticks += 1

            for obs in self.observers:
                try:
                    obs.get(frame)
                except Exception as e:
                    pass  # Continue even if one observer fails

        # Update episode counters
        self.episode_step += 1
        self.episode_time += self.decision_interval_sec
        self.time_in_current_phase += self.decision_interval_sec

        # Aggregate metrics from observers
        group_metrics = self._aggregate_metrics()

        # Compute reward
        reward = compute_traffic_reward(
            group_metrics,
            phase_changed,
            self.reward_config
        )

        # Build observation
        obs = self._get_observation()

        # Check termination
        terminated = False
        truncated = (self.episode_time >= self.episode_length_sec)

        # Info dict
        info = {
            "episode": {
                "step": self.episode_step,
                "time_sec": self.episode_time,
            },
            "metrics": group_metrics,
            "phase": {
                "current_phase": self.current_phase,
                "phase_changed": phase_changed,
                "time_in_phase": self.time_in_current_phase,
            },
            "group_id": list(self.tl_group["ids"]),
            "stable_ids": [self.stable_id_map[tl.id] for tl in self.tl_actors],
        }

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """
        Build observation vector from current observer states.

        Returns:
            Normalized observation vector of shape (11*n_lights,)
        """
        features = []

        for obs in self.observers:
            # Get current state from logger
            state_name = obs.logger.last_state if obs.logger.last_state else "Red"
            time_in_state = obs.logger.time_in_state
            queue_ema = obs.logger.queue_ema
            avg_wait = obs.logger.avg_wait
            max_wait = obs.logger.max_wait
            num_long_wait = obs.logger.num_long_wait_60s
            arrival_ema = obs.logger.arrival_ema
            discharge_ema = obs.logger.discharge_ema

            # Phase encoding (one-hot: Red, Yellow, Green)
            phase_red = 1.0 if "Red" in state_name else 0.0
            phase_yellow = 1.0 if "Yellow" in state_name else 0.0
            phase_green = 1.0 if "Green" in state_name else 0.0

            # Normalized features
            time_in_state_norm = min(time_in_state / 60.0, 1.0)
            queue_ema_norm = min(queue_ema / 20.0, 1.0)
            avg_wait_norm = min(avg_wait / 120.0, 1.0)
            max_wait_norm = min(max_wait / 120.0, 1.0)
            num_long_wait_norm = min(num_long_wait / 20.0, 1.0)
            arrival_ema_norm = min(arrival_ema / 2.0, 1.0)
            discharge_ema_norm = min(discharge_ema / 2.0, 1.0)

            # Append 11 features per light
            features.extend([
                phase_red,
                phase_yellow,
                phase_green,
                time_in_state_norm,
                queue_ema_norm,
                avg_wait_norm,
                max_wait_norm,
                num_long_wait_norm,
                arrival_ema_norm,
                discharge_ema_norm,
            ])

        return np.array(features, dtype=np.float32)

    def _aggregate_metrics(self) -> Dict[str, float]:
        """
        Aggregate group-level metrics from all observers.

        Returns:
            Dictionary with total metrics across all lights
        """
        total_queue_ema = 0.0
        total_avg_wait = 0.0
        total_num_long_wait = 0

        for obs in self.observers:
            total_queue_ema += obs.logger.queue_ema
            total_avg_wait += obs.logger.avg_wait
            total_num_long_wait += obs.logger.num_long_wait_60s

        return {
            "total_queue_ema": total_queue_ema,
            "total_avg_wait": total_avg_wait,
            "total_num_long_wait": total_num_long_wait,
        }

    def _cleanup(self):
        """Destroy vehicles and observers from previous episode."""
        # Destroy observers
        for obs in self.observers:
            try:
                obs.destroy()
            except Exception as e:
                pass

        # Destroy vehicles
        for v in self.vehicles:
            try:
                v.destroy()
            except Exception as e:
                pass

        self.observers = []
        self.vehicles = []

    def close(self):
        """Clean up resources."""
        self._cleanup()

        # Disable synchronous mode
        if self.world is not None:
            try:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                self.world.apply_settings(settings)
            except:
                pass
