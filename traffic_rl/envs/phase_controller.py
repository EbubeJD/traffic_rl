"""
Phase controller for managing traffic light states in CARLA intersections.

This module handles the mapping between discrete RL actions and CARLA traffic light states.
"""

import carla
from typing import Dict, List


class PhaseController:
    """
    Manages traffic light phases for a single intersection group.

    A "phase" is a coordinated configuration of traffic light states across
    multiple traffic lights in an intersection.
    """

    def __init__(self, phase_config: Dict[str, Dict[str, carla.TrafficLightState]]):
        """
        Initialize phase controller with phase configuration.

        Args:
            phase_config: Dictionary mapping phase names to traffic light configurations.
                Example:
                {
                    "phase_0": {
                        "road42_lane5_s1234": carla.TrafficLightState.Green,
                        "road42_lane6_s1235": carla.TrafficLightState.Red,
                    },
                    "phase_1": {
                        "road42_lane5_s1234": carla.TrafficLightState.Red,
                        "road42_lane6_s1235": carla.TrafficLightState.Green,
                    },
                }
        """
        self.phase_config = phase_config
        self.phase_names = sorted(phase_config.keys())
        self.num_phases = len(self.phase_names)

        if self.num_phases == 0:
            raise ValueError("Phase configuration is empty")

        # Validate that all phases have the same traffic lights
        all_stable_ids = [set(phase_config[name].keys()) for name in self.phase_names]
        if len(set(tuple(sorted(ids)) for ids in all_stable_ids)) != 1:
            raise ValueError("All phases must configure the same set of traffic lights")

        self.stable_ids = sorted(all_stable_ids[0])

    def apply_phase(
        self,
        phase_id: int,
        tl_actors: List[carla.TrafficLight],
        stable_id_map: Dict[int, str]
    ):
        """
        Apply a specific phase configuration to traffic light actors.

        Args:
            phase_id: Index of phase to apply (0, 1, 2, ...)
            tl_actors: List of CARLA traffic light actor objects
            stable_id_map: Mapping from actor.id to stable_id string
        """
        if phase_id < 0 or phase_id >= self.num_phases:
            raise ValueError(f"Invalid phase_id {phase_id}. Must be in range [0, {self.num_phases})")

        phase_name = self.phase_names[phase_id]
        phase_states = self.phase_config[phase_name]

        for tl in tl_actors:
            stable_id = stable_id_map.get(tl.id)
            if stable_id is None:
                print(f"[WARN] No stable_id found for traffic light actor {tl.id}")
                continue

            if stable_id not in phase_states:
                print(f"[WARN] Stable ID '{stable_id}' not in phase '{phase_name}' configuration")
                continue

            desired_state = phase_states[stable_id]
            try:
                tl.set_state(desired_state)
            except Exception as e:
                print(f"[ERROR] Failed to set state for {stable_id}: {e}")

    def apply_yellow(self, tl_actors: List[carla.TrafficLight]):
        """
        Set all traffic lights to red (yellow/all-red transition).

        This is used during phase transitions to ensure safe clearance.

        Args:
            tl_actors: List of CARLA traffic light actor objects
        """
        for tl in tl_actors:
            try:
                tl.set_state(carla.TrafficLightState.Red)
            except Exception as e:
                print(f"[ERROR] Failed to set red state: {e}")

    def get_next_phase_id(self, current_phase_id: int) -> int:
        """
        Get the next phase in the cycle.

        Args:
            current_phase_id: Current phase index

        Returns:
            Next phase index (wraps around)
        """
        return (current_phase_id + 1) % self.num_phases
