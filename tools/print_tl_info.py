"""
Helper script to print traffic light information for manual phase mapping.

Usage:
    python tools/print_tl_info.py

This script connects to CARLA, discovers traffic light groups,
and prints detailed information about GROUP_INDEX=0 for manual phase configuration.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import carla
from config import TOWN, GROUP_INDEX
from utils.carla_helpers import get_tl_groups, tl_stable_id


def print_traffic_light_info():
    """Connect to CARLA and print traffic light information for manual phase mapping."""

    print("=" * 80)
    print("TRAFFIC LIGHT GROUP INFO EXTRACTOR")
    print("=" * 80)

    # Connect to CARLA
    print("\n[1/4] Connecting to CARLA server...")
    try:
        client = carla.Client("localhost", 2000)
        client.set_timeout(10.0)
        print("✓ Connected to CARLA server")
    except Exception as e:
        print(f"✗ Failed to connect to CARLA: {e}")
        print("\nMake sure CARLA is running on localhost:2000")
        return

    # Load world
    print(f"\n[2/4] Loading world '{TOWN}'...")
    try:
        world = client.load_world(TOWN, map_layers=carla.MapLayer.NONE)
        print(f"✓ Loaded world: {TOWN}")
    except Exception as e:
        print(f"✗ Failed to load world: {e}")
        return

    # Discover traffic light groups
    print("\n[3/4] Discovering traffic light groups...")
    try:
        groups = get_tl_groups(world)
        print(f"✓ Found {len(groups)} traffic light group(s)")
    except Exception as e:
        print(f"✗ Failed to discover groups: {e}")
        return

    if not groups:
        print("\n✗ No traffic light groups found in this world!")
        return

    if GROUP_INDEX >= len(groups):
        print(f"\n✗ GROUP_INDEX={GROUP_INDEX} is out of range!")
        print(f"   Available indices: 0 to {len(groups-1)}")
        return

    # Print info for selected group
    print(f"\n[4/4] Printing details for GROUP_INDEX={GROUP_INDEX}...")
    print("=" * 80)

    group = groups[GROUP_INDEX]
    tl_actors = group["actors"]

    print(f"\nGroup {GROUP_INDEX} contains {len(tl_actors)} traffic light(s):\n")

    for i, tl in enumerate(tl_actors):
        stable_id = tl_stable_id(world, tl)
        transform = tl.get_transform()
        location = transform.location
        rotation = transform.rotation

        print(f"Traffic Light #{i+1}")
        print(f"  Stable ID:  {stable_id}")
        print(f"  Actor ID:   {tl.id}")
        print(f"  Location:   x={location.x:.2f}, y={location.y:.2f}, z={location.z:.2f}")
        print(f"  Rotation:   pitch={rotation.pitch:.2f}, yaw={rotation.yaw:.2f}, roll={rotation.roll:.2f}")
        print(f"  State:      {tl.get_state()}")

        # Try to get trigger volume info
        try:
            trigger_volume = tl.trigger_volume
            if trigger_volume:
                tv_loc = trigger_volume.location
                print(f"  Trigger Vol: x={tv_loc.x:.2f}, y={tv_loc.y:.2f}, z={tv_loc.z:.2f}")
        except:
            pass

        print()

    # Print template for phase configuration
    print("=" * 80)
    print("\nPHASE CONFIGURATION TEMPLATE")
    print("Copy this to traffic_rl/rl/config.py and fill in Green/Red states:\n")
    print("INTERSECTION_0_PHASES = {")
    print("    \"phase_0\": {  # e.g., North-South Green")
    for i, tl in enumerate(tl_actors):
        stable_id = tl_stable_id(world, tl)
        print(f"        \"{stable_id}\": carla.TrafficLightState.Green,  # TL #{i+1}")
    print("    },")
    print("    \"phase_1\": {  # e.g., East-West Green")
    for i, tl in enumerate(tl_actors):
        stable_id = tl_stable_id(world, tl)
        print(f"        \"{stable_id}\": carla.TrafficLightState.Red,  # TL #{i+1}")
    print("    },")
    print("}")
    print("\n" + "=" * 80)
    print("\nNEXT STEPS:")
    print("1. Look at the traffic lights in CARLA to identify which approaches they control")
    print("2. Use the yaw angle to infer direction (0°=North, 90°=East, 180°=South, 270°=West)")
    print("3. Update the phase configuration in traffic_rl/rl/config.py")
    print("4. Define phase_0 and phase_1 as complementary green/red patterns")
    print("=" * 80)


if __name__ == "__main__":
    print_traffic_light_info()
