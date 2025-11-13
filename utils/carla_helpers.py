import math
import carla

from config import RES, CAM_FOV

def _fwd_from_yaw(yaw_deg):
    r = math.radians(yaw_deg)
    return carla.Vector3D(math.cos(r), math.sin(r), 0.0)

def _right_from_yaw(yaw_deg):
    r = math.radians(yaw_deg + 90.0)
    return carla.Vector3D(math.cos(r), math.sin(r), 0.0)

def get_tl_groups(world):
    groups, seen = [], set()
    for tl in world.get_actors().filter("traffic.traffic_light*"):
        gl = tl.get_group_traffic_lights() or [tl]
        key = frozenset(a.id for a in gl)
        if key in seen: 
            continue
        seen.add(key)
        groups.append({"ids": key, "actors": list(gl), "rep": gl[0]})
    groups.sort(key=lambda g: min(g["ids"]))
    return groups

def _tl_center(tl):
    tf = tl.get_transform()
    tv = getattr(tl, "trigger_volume", None) or (tl.get_trigger_volume() if hasattr(tl, "get_trigger_volume") else None)
    return tf.transform(tv.location) if tv and getattr(tv, "location", None) else tf.location

def tl_stable_id(world, tl):
    m = world.get_map()
    c = _tl_center(tl)
    wp = m.get_waypoint(c, project_to_road=True, lane_type=carla.LaneType.Driving)
    s_val = getattr(wp, "s", None)
    if s_val is not None:
        s_idx = int(round(float(s_val) * 100))
    else:
        s_idx = int(round(wp.transform.location.distance(c) * 100))
    return f"road{wp.road_id}_lane{wp.lane_id}_s{s_idx}"

def spawn_tl_camera(world, tl, bp_lib, back_m=15.0, side_m=1.2, height_m=5.0):
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(RES[0]))
    cam_bp.set_attribute("image_size_y", str(RES[1]))
    cam_bp.set_attribute("fov", str(CAM_FOV))

    m = world.get_map()
    center = _tl_center(tl)
    wp = m.get_waypoint(center, project_to_road=True, lane_type=carla.LaneType.Driving)

    lane_yaw = wp.transform.rotation.yaw
    facing_yaw = (lane_yaw + 180.0) % 360.0
    fwd = _fwd_from_yaw(facing_yaw)
    right = _right_from_yaw(facing_yaw)

    cam_loc = carla.Location(center.x - fwd.x*back_m + right.x*side_m,
                             center.y - fwd.y*back_m + right.y*side_m,
                             center.z + height_m)
    dx,dy,dz = center.x-cam_loc.x, center.y-cam_loc.y, center.z-cam_loc.z
    pitch = math.degrees(math.atan2(dz, math.hypot(dx,dy))) - CAM_FOV*0.10

    cam = world.spawn_actor(
        cam_bp,
        carla.Transform(cam_loc, carla.Rotation(pitch=pitch, yaw=facing_yaw, roll=0.0))
    )
    return cam

def debug_draw_camera_ray(world, cam, life=6.0):
    tf = cam.get_transform()
    f = _fwd_from_yaw(tf.rotation.yaw)
    s = tf.location
    e = carla.Location(s.x+f.x*8, s.y+f.y*8, s.z+f.z*8)
    world.debug.draw_arrow(s, e, thickness=0.08, arrow_size=0.3, color=carla.Color(0,255,0), life_time=life)

def fly_to_camera(world, cam, dz=12.0):
    tf = cam.get_transform()
    world.get_spectator().set_transform(
        carla.Transform(carla.Location(tf.location.x, tf.location.y, tf.location.z+dz),
                        carla.Rotation(pitch=-90))
    )
