"""
TrafficRunner: Manages CARLA connection and simulation stepping.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import carla
import random
from typing import List, Dict, Optional, Sequence

from config import TOWN, GROUP_INDEX, NUM_AUTOPILOT, DT, CONTROL_TL_IDS
from utils.carla_helpers import get_tl_groups, tl_stable_id, spawn_autopilot_vehicles
from observers.tl_observer import TLObserver


class TrafficRunner:
    """
    Manages CARLA simulation, traffic light groups, and TLObservers.

    This class handles:
    - CARLA connection and world setup
    - Traffic light group discovery
    - Observer creation and management
    - Vehicle spawning
    - Simulation stepping
    """

    def __init__(
        self,
        carla_host: str = "localhost",
        carla_port: int = 2000,
        town: str = TOWN,
        group_index: int = GROUP_INDEX,
        num_vehicles: int = NUM_AUTOPILOT,
        dt: float = DT,
        control_tl_ids: Optional[Sequence[str]] = None,
    ):
        """
        Initialize TrafficRunner.

        Args:
            carla_host: CARLA server host
            carla_port: CARLA server port
            town: CARLA map/town name
            group_index: Which traffic light group to use
            num_vehicles: Number of autopilot vehicles to spawn
            dt: Simulation timestep (fixed_delta_seconds)
        """
        self.carla_host = carla_host
        self.carla_port = carla_port
        self.town = town
        self.group_index = group_index
        self.num_vehicles = num_vehicles
        self.dt = dt
        # Stable IDs for the specific TL(s)/lane(s) to control.
        # Defaults to CONTROL_TL_IDS configured in config.py so we can point at a known ROI/ticks set.
        self.control_tl_ids = list(control_tl_ids) if control_tl_ids is not None else list(CONTROL_TL_IDS)

        # CARLA objects
        self.client: Optional[carla.Client] = None
        self.world: Optional[carla.World] = None
        self.group: Optional[Dict] = None
        self.observers: List[TLObserver] = []
        self.vehicles: List = []

        # Simulation state
        self.frame_count = 0
        self.is_initialized = False

    def initialize(self):
        """Connect to CARLA and set up simulation."""
        if self.is_initialized:
            return

        print(f"[TrafficRunner] Connecting to CARLA at {self.carla_host}:{self.carla_port}...")
        self.client = carla.Client(self.carla_host, self.carla_port)
        self.client.set_timeout(30.0)

        print(f"[TrafficRunner] Loading world '{self.town}'...")
        self.world = self.client.load_world(self.town, map_layers=carla.MapLayer.NONE)

        # Set synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.dt
        self.world.apply_settings(settings)

        # Discover traffic light groups
        groups = get_tl_groups(self.world)
        if not groups:
            raise RuntimeError("No traffic light groups found")
        if self.group_index >= len(groups):
            raise IndexError(f"GROUP_INDEX {self.group_index} out of range (found {len(groups)} groups)")

        selected_group = groups[self.group_index]

        # Optionally narrow the group to just the configured stable_id(s) (e.g., road5_lane-1).
        if self.control_tl_ids:
            filtered_actors = []
            for tl in selected_group["actors"]:
                sid = tl_stable_id(self.world, tl)
                if sid in self.control_tl_ids:
                    filtered_actors.append(tl)

            if filtered_actors:
                selected_group = {
                    **selected_group,
                    "actors": filtered_actors,
                    "ids": frozenset(a.id for a in filtered_actors),
                }
                print(f"[TrafficRunner] Narrowed to {len(filtered_actors)} TL(s) by CONTROL_TL_IDS={self.control_tl_ids}")
            else:
                print(f"[TrafficRunner] CONTROL_TL_IDS {self.control_tl_ids} not found in group {self.group_index}; using full group.")

        self.group = selected_group
        print(f"[TrafficRunner] Selected group {self.group_index} with {len(self.group['actors'])} traffic lights")

        # Create observers
        bp_lib = self.world.get_blueprint_library()
        for tl in self.group["actors"]:
            try:
                observer = TLObserver(self.world, tl, bp_lib)
                self.observers.append(observer)
                print(f"  - Observer for {tl_stable_id(self.world, tl)}")
            except Exception as e:
                print(f"  - Failed to create observer: {e}")

        # Spawn vehicles
        print(f"[TrafficRunner] Spawning {self.num_vehicles} vehicles...")
        self.vehicles = spawn_autopilot_vehicles(self.world, self.client, self.num_vehicles)
        print(f"  - Spawned {len(self.vehicles)} vehicles")

        # Warm-up: tick a few times to stabilize
        print("[TrafficRunner] Running warm-up...")
        for _ in range(20):
            self.world.tick()
            self.frame_count += 1

        self.is_initialized = True
        print("[TrafficRunner] Initialization complete")

    def step(self, light_states: Optional[Dict[int, carla.TrafficLightState]] = None) -> Dict[str, Dict]:
        """
        Step the simulation forward by one tick.

        Args:
            light_states: Optional dict mapping actor.id -> TrafficLightState
                         If provided, sets traffic lights before ticking

        Returns:
            Dictionary mapping stable_id -> metrics dict from TLObserver
        """
        if not self.is_initialized:
            self.initialize()

        # Set traffic light states if provided
        if light_states:
            for tl in self.group["actors"]:
                if tl.id in light_states:
                    tl.set_state(light_states[tl.id])

        # Tick simulation
        self.world.tick()
        snapshot = self.world.get_snapshot()
        frame = snapshot.frame
        self.frame_count += 1

        # Collect metrics from observers
        metrics = {}
        for obs in self.observers:
            try:
                result = obs.get(frame)
                if result:
                    stable_id = result["stable_id"]
                    metrics[stable_id] = result
            except Exception as e:
                # Continue if one observer fails
                pass

        return metrics

    def get_stable_ids(self) -> List[str]:
        """Get list of stable IDs for traffic lights in order."""
        if not self.is_initialized:
            self.initialize()
        return [tl_stable_id(self.world, tl) for tl in self.group["actors"]]

    def reset(self):
        """Reset simulation (destroy vehicles, respawn)."""
        print("[TrafficRunner] Resetting simulation...")

        # Destroy existing vehicles
        for v in self.vehicles:
            try:
                v.destroy()
            except:
                pass
        self.vehicles = []

        # Respawn vehicles
        self.vehicles = spawn_autopilot_vehicles(self.world, self.client, self.num_vehicles)
        print(f"  - Respawned {len(self.vehicles)} vehicles")

        # Reset frame count
        self.frame_count = 0

        # Warm-up
        for _ in range(20):
            self.world.tick()
            self.frame_count += 1

        print("[TrafficRunner] Reset complete")

    def close(self):
        """Clean up resources."""
        print("[TrafficRunner] Cleaning up...")

        # Destroy observers
        for obs in self.observers:
            try:
                obs.destroy()
            except:
                pass

        # Destroy vehicles
        for v in self.vehicles:
            try:
                v.destroy()
            except:
                pass

        # Disable synchronous mode
        if self.world:
            try:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                self.world.apply_settings(settings)
            except:
                pass

        print("[TrafficRunner] Cleanup complete")
