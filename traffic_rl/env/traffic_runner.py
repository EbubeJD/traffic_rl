import time
import random
import carla
import numpy as np
from collections import deque

# Import from root
import sys
import os
sys.path.append(os.getcwd())

from config import TOWN, GROUP_INDEX, NUM_AUTOPILOT, DT
from utils.carla_helpers import get_tl_groups, fly_to_camera
from observers.tl_observer import TLObserver

class TrafficRunner:
    def __init__(self, town=TOWN, group_index=GROUP_INDEX, num_vehicles=NUM_AUTOPILOT, dt=DT):
        self.town = town
        self.group_index = group_index
        self.num_vehicles = num_vehicles
        self.dt = dt
        
        self.client = None
        self.world = None
        self.group = None
        self.observers = []
        self.vehicles = []
        self.tm = None
        
        self._init_world()

    def _init_world(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        
        # Load world (if not already loaded to save time, but for safety load it)
        # Check if current map is correct
        if self.client.get_world().get_map().name.split('/')[-1] != self.town:
            self.world = self.client.load_world(self.town, map_layers=carla.MapLayer.NONE)
        else:
            self.world = self.client.get_world()

        # Settings
        s = self.world.get_settings()
        s.synchronous_mode = True
        s.fixed_delta_seconds = self.dt
        self.world.apply_settings(s)
        
        # Traffic Manager
        self.tm = self.client.get_trafficmanager()
        self.tm.set_synchronous_mode(True)
        self.tm.global_percentage_speed_difference(10.0)

        # Get Group
        groups = get_tl_groups(self.world)
        if not groups: raise RuntimeError("No traffic-light groups found.")
        if self.group_index >= len(groups): raise IndexError(f"Bad GROUP_INDEX {self.group_index}")
        self.group = groups[self.group_index]
        
        # Setup Observers
        bp_lib = self.world.get_blueprint_library()
        self.observers = [TLObserver(self.world, tl, bp_lib) for tl in self.group["actors"]]
        if self.observers:
            fly_to_camera(self.world, self.observers[0].cam, dz=12.0)

    def reset(self):
        """Soft reset: respawn vehicles, clear queues."""
        self._cleanup_vehicles()
        self.vehicles = self._spawn_vehicles()
        
        # Warmup
        for _ in range(20):
            self.world.tick()
            
        return self.get_metrics()

    def step(self, action_state_dict):
        """
        Apply state to traffic lights and tick.
        action_state_dict: {tl_id: carla.TrafficLightState}
        """
        # Apply states
        for tl in self.group["actors"]:
            if tl.id in action_state_dict:
                tl.set_state(action_state_dict[tl.id])
                tl.freeze(True) # Keep it in this state

        # Tick world
        self.world.tick()
        frame = self.world.get_snapshot().frame
        
        # Update observers
        metrics = {}
        for ob in self.observers:
            try:
                data = ob.get(frame)
                if data:
                    metrics[ob.stable_id] = data
            except Exception as e:
                print(f"[Runner Error] {e}")
                
        return metrics

    def _spawn_vehicles(self):
        if self.num_vehicles <= 0: return []
        bp_lib = self.world.get_blueprint_library()
        spawns = self.world.get_map().get_spawn_points()
        random.shuffle(spawns)
        
        vehicles = []
        tm_port = self.tm.get_port()
        
        for sp in spawns[:self.num_vehicles*2]:
            if len(vehicles) >= self.num_vehicles: break
            bp = random.choice(bp_lib.filter("vehicle.*"))
            if bp.has_attribute("role_name"):
                bp.set_attribute("role_name", "autopilot")
            v = self.world.try_spawn_actor(bp, sp)
            if v:
                v.set_autopilot(True, tm_port)
                vehicles.append(v)
        return vehicles

    def _cleanup_vehicles(self):
        for v in self.vehicles:
            if v.is_alive: v.destroy()
        self.vehicles = []

    def close(self):
        self._cleanup_vehicles()
        for ob in self.observers:
            ob.destroy()
        # Disable sync mode
        s = self.world.get_settings()
        s.synchronous_mode = False
        self.world.apply_settings(s)
