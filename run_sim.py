import time, random
import os, csv 
import carla

from config import (TOWN, GROUP_INDEX, NUM_AUTOPILOT, DT, SAVE_ROOT)
from utils.carla_helpers import get_tl_groups, fly_to_camera
from observers.tl_observer import TLObserver


def spawn_autopilot_vehicles(world, client, num=NUM_AUTOPILOT):
    if num <= 0: return []
    random.seed(7)
    bp_lib = world.get_blueprint_library()
    spawns = world.get_map().get_spawn_points(); random.shuffle(spawns)
    tm = client.get_trafficmanager()
    tm_port = tm.get_port()
    tm.set_synchronous_mode(True)
    tm.global_percentage_speed_difference(10.0)
    vehicles=[]
    for sp in spawns[:num*2]:
        if len(vehicles) >= num: break
        bp = random.choice(bp_lib.filter("vehicle.*"))
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", "autopilot")
        v = world.try_spawn_actor(bp, sp)
        if v:
            v.set_autopilot(True, tm_port)
            vehicles.append(v)
    print(f"[spawn] {len(vehicles)} vehicles")
    return vehicles

def log_groups(groups):
    """
    Write a CSV mapping group_index → roads / lanes / TL ids
    and print a summary to the console.
    """
    town_dir = os.path.join(SAVE_ROOT, TOWN)
    os.makedirs(town_dir, exist_ok=True)
    out_path = os.path.join(town_dir, "tl_groups.csv")

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group_index", "rep_tl_id", "road_ids", "lane_ids", "actor_ids"])
        for idx, g in enumerate(groups):
            roads = g.get("roads", [])
            lanes = g.get("lanes", [])
            actor_ids = [a.id for a in g["actors"]]
            w.writerow([
                idx,
                g["rep"].id,
                ";".join(str(r) for r in roads),
                ";".join(str(l) for l in lanes),
                ";".join(str(i) for i in actor_ids),
            ])

    print(f"[groups] wrote metadata to {out_path}")
    print(f"[groups] found {len(groups)} traffic-light groups:")
    for idx, g in enumerate(groups):
        roads = g.get("roads", [])
        lanes = g.get("lanes", [])
        print(f"  [G{idx}] rep={g['rep'].id} roads={roads} lanes={lanes} actors={len(g['actors'])}")


def main():
    client = carla.Client("localhost", 2000); client.set_timeout(10.0)
    world = client.load_world(TOWN, map_layers=carla.MapLayer.NONE)
    s = world.get_settings(); s.synchronous_mode=True; s.fixed_delta_seconds=DT; world.apply_settings(s)

    groups = get_tl_groups(world)
    if not groups: raise RuntimeError("No traffic-light groups found.")

    log_groups(groups)

    if GROUP_INDEX >= len(groups): raise IndexError(f"Bad GROUP_INDEX {GROUP_INDEX}")
    group = groups[GROUP_INDEX]

    bp_lib = world.get_blueprint_library()

    observers = [TLObserver(world, tl, bp_lib) for tl in group["actors"]]
    if observers: fly_to_camera(world, observers[0].cam, dz=12.0)

    vehicles = spawn_autopilot_vehicles(world, client, NUM_AUTOPILOT)

    try:
        t0 = time.time()
        while True:
            world.tick()
            frame = world.get_snapshot().frame

            for ob in observers:
                try:
                    out = ob.get(frame)
                    if out and frame % 40 == 0:
                        print(f"[F{frame:06d}] TL[{out['stable_id']}] "
                              f"{out['state']:<6} t={out['time_in_state']:.1f}s "
                              f"q={out['queue']} (ema={out['queue_ema']:.2f})")
                except Exception as e:
                    print(f"[OBS ERROR] TL {getattr(ob,'stable_id','?')}: {e}")

            if time.time() - t0 > 120: break
    finally:
        print("[cleanup] destroying actors…")
        for ob in observers:
            try: ob.destroy()
            except: pass
        for v in vehicles:
            try: v.destroy()
            except: pass
        s.synchronous_mode=False; world.apply_settings(s)
        print("[cleanup] done.")

if __name__ == "__main__":
    main()
