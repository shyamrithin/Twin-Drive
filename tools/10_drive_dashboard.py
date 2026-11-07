#!/usr/bin/env python3
# =============================================================================
# File        : tools/10_drive_dashboard.py
# Project     : TwinDrive — Lane-Free Autonomous Driving on a Digital Twin
#               of an Indian Campus (SRM Town / cm2, CARLA 0.9.16)
# Phase       : 1 — Ego vehicle, sensor suite and dashboard (step 1)
# Runs on     : WSL2 Ubuntu 22.04 (client)  ->  Windows CarlaUE4.exe (server)
# Env         : conda activate twindrive   (+ pip install pygame)
# =============================================================================
"""
Description
-----------
Spawns the TwinDrive ego car at a campus landmark, attaches the sensor suite
(front camera, 32-channel 3D LiDAR, IMU, collision), fills campus with
background traffic, and shows everything live on a pygame dashboard:

    front camera with ground-truth vehicle boxes   3D LiDAR chase view
    heading-up BEV map (drivable area + obstacles)  speed gauge, steering
    wheel, pedals, gear                             lead vehicle gap / TTC
    compass + artificial horizon                    g-g acceleration plot
    speed / steering / pedal history                status bar

You drive with the keyboard, or press P to hand over to CARLA's Traffic
Manager autopilot. Our own planner and controller replace the autopilot in
the next step.

Simulation runs in SYNCHRONOUS mode at a fixed dt: the server only advances
when this script calls world.tick(), so every sensor reading in a frame
describes the same instant. Settings are always restored on exit, even after
a crash; if this script is killed hard (closing the terminal), restart
CarlaUE4.exe because it will be waiting for a tick that never comes.

Lead vehicle: the nearest vehicle ahead whose box overlaps the ego's width,
using ground-truth positions (a straight corridor, so it can miss cars on
tight curves). Real perception replaces this in Phase 3.

Usage
-----
    python tools/10_drive_dashboard.py                       # Main Gate, 30 cars
    python tools/10_drive_dashboard.py --spawn tech_park --traffic 50
    python tools/10_drive_dashboard.py --autopilot --scale 0.8
    python tools/10_drive_dashboard.py --no-camera           # lighter on the GPU
"""

import os

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")  # WSL has no sound device

import argparse
import math
import queue
import random
import time
from pathlib import Path

import carla
import numpy as np
import pygame

from twindrive import DEFAULT_MAP
from twindrive.sensors import CAMERA_CFG, SensorSuite
from twindrive.viz import BEV_CAR_ROW_FRAC, BEV_M_PER_PX, BEV_SIZE, WINDOW_SIZE, Dashboard
from twindrive.world import (DrivableMap, is_routable, load_landmarks,
                             routable_lane_keys, world_to_local)

NEAR_M = 60.0          # vehicles closer than this are drawn and considered
LANDMARK_VIEW_M = 120.0
# 8 box corners, index = sx*4 + sy*2 + sz  (matches twindrive.viz.dashboard)
CORNER_SIGNS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
                        dtype=np.float64)


# ============================================================================ keyboard
class KeyboardDriver:
    """W/S/A/D driving with smooth steering, like CARLA's manual_control."""

    def __init__(self):
        self.steer = 0.0
        self.throttle = 0.0
        self.brake = 0.0
        self.reverse = False
        self.autopilot = False

    def control(self, keys, ms):
        c = carla.VehicleControl()
        up = keys[pygame.K_w] or keys[pygame.K_UP]
        down = keys[pygame.K_s] or keys[pygame.K_DOWN]
        left = keys[pygame.K_a] or keys[pygame.K_LEFT]
        right = keys[pygame.K_d] or keys[pygame.K_RIGHT]

        self.throttle = min(1.0, self.throttle + 0.002 * ms) if up else 0.0
        self.brake = min(1.0, self.brake + 0.006 * ms) if down else 0.0
        inc = 5e-4 * ms
        if left:
            self.steer = min(0.0, self.steer) - inc
        elif right:
            self.steer = max(0.0, self.steer) + inc
        else:
            self.steer = 0.0
        self.steer = float(np.clip(self.steer, -0.7, 0.7))

        c.throttle, c.brake, c.steer = self.throttle, self.brake, self.steer
        c.hand_brake = bool(keys[pygame.K_SPACE])
        c.reverse = self.reverse
        return c


# ============================================================================ setup
def spawn_ego(world, cmap, landmarks, name, bp_id):
    if name not in landmarks:
        raise SystemExit(f"Unknown landmark '{name}'. Known: {', '.join(sorted(landmarks))}")
    lm = landmarks[name]
    bp_lib = world.get_blueprint_library()
    bps = bp_lib.filter(bp_id)
    bp = bps[0] if bps else bp_lib.filter("vehicle.*")[0]
    bp.set_attribute("role_name", "hero")
    base = carla.Transform(carla.Location(lm["x"], lm["y"], lm["z"] + 0.5),
                           carla.Rotation(yaw=lm["yaw"]))
    ego = world.try_spawn_actor(bp, base)
    if ego is None:  # spot occupied: try a few points further along the lane
        wp = cmap.get_waypoint(base.location)
        for k in range(1, 10):
            nxt = wp.next(6.0 * k)
            if not nxt:
                break
            tf = nxt[0].transform
            tf.location.z += 0.5
            ego = world.try_spawn_actor(bp, tf)
            if ego:
                break
    if ego is None:
        raise SystemExit(f"Could not spawn the ego car near '{name}'.")
    print(f"Ego: {ego.type_id} at {lm['label']}")
    return ego


def spawn_traffic(client, world, cmap, tm, n, bike_frac, ego_loc, rng):
    if n <= 0:
        return []
    routable = routable_lane_keys(cmap)
    points = [sp for sp in cmap.get_spawn_points()
              if sp.location.distance(ego_loc) > 15.0
              and is_routable(cmap.get_waypoint(sp.location), routable)]
    rng.shuffle(points)

    def base_type(bp):
        if bp.has_attribute("base_type"):
            return bp.get_attribute("base_type").as_str().lower()
        return "motorcycle" if int(bp.get_attribute("number_of_wheels")) == 2 else "car"

    all_bps = world.get_blueprint_library().filter("vehicle.*")
    cars = [b for b in all_bps if base_type(b) in ("car", "van")]
    bikes = [b for b in all_bps if base_type(b) == "motorcycle"]
    if not cars:  # custom builds may leave base_type empty
        cars = [b for b in all_bps if int(b.get_attribute("number_of_wheels")) == 4]
    batch = []
    for sp in points[:n]:
        pool = bikes if (bikes and rng.random() < bike_frac) else cars
        bp = rng.choice(pool)
        if bp.has_attribute("color"):
            bp.set_attribute("color", rng.choice(bp.get_attribute("color").recommended_values))
        bp.set_attribute("role_name", "traffic")
        batch.append(carla.command.SpawnActor(bp, sp).then(
            carla.command.SetAutopilot(carla.command.FutureActor, True, tm.get_port())))
    ids = [r.actor_id for r in client.apply_batch_sync(batch, True) if not r.error]
    print(f"Traffic: {len(ids)} vehicles spawned ({len(points)} free spawn points)")
    return ids


# ============================================================================ per-frame state
def box_corners_in_ego(actor, ego_inv):
    bb = actor.bounding_box
    ext = np.array([bb.extent.x, bb.extent.y, bb.extent.z])
    off = np.array([bb.location.x, bb.location.y, bb.location.z])
    local = np.c_[CORNER_SIGNS * ext + off, np.ones(8)]
    world_pts = local @ np.array(actor.get_transform().get_matrix()).T
    return (world_pts @ ego_inv.T)[:, :3]


def build_state(ego, others, data, imu_f, landmarks, dm, sim_time, status):
    tf = ego.get_transform()
    ex, ey, yaw = tf.location.x, tf.location.y, tf.rotation.yaw
    v = ego.get_velocity()
    speed = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
    fwd = tf.get_forward_vector()
    v_ego_fwd = v.x * fwd.x + v.y * fwd.y + v.z * fwd.z
    ebb = ego.bounding_box.extent
    ego_inv = np.array(tf.get_inverse_matrix())

    vehicles, lead, best = [], None, NEAR_M
    for a in others:
        try:
            loc = a.get_location()
        except RuntimeError:
            continue
        if loc.distance(tf.location) > NEAR_M:
            continue
        corners = box_corners_in_ego(a, ego_inv)
        cx, cy = corners[:, 0].mean(), corners[:, 1].mean()
        info = {"corners": corners, "dist": float(math.hypot(cx, cy)), "lead": False}
        vehicles.append(info)
        aext = a.bounding_box.extent
        in_path = cx > 0 and abs(cy) < ebb.y + aext.y + 0.3
        if in_path and cx < best:
            best = cx
            av = a.get_velocity()
            rel = (av.x * fwd.x + av.y * fwd.y + av.z * fwd.z) - v_ego_fwd  # m/s
            gap = max(cx - ebb.x - aext.x, 0.0)
            closing = -rel
            lead = {"info": info, "gap_m": gap, "rel_kmh": rel * 3.6,
                    "ttc_s": gap / closing if closing > 0.3 else None,
                    "speed_kmh": math.sqrt(av.x ** 2 + av.y ** 2) * 3.6, "type": a.type_id}
    if lead:
        lead["info"]["lead"] = True
        lead["info"]["dist"] = lead["gap_m"]
        lead.pop("info")

    ctl = ego.get_control()
    try:
        road_deg = ego.get_wheel_steer_angle(carla.VehicleWheelLocation.FL_Wheel)
    except (AttributeError, RuntimeError):
        road_deg = None

    lms = []
    for v_ in landmarks.values():
        f, r = world_to_local(v_["x"], v_["y"], ex, ey, yaw)
        if math.hypot(f, r) < LANDMARK_VIEW_M:
            lms.append((v_["label"], float(f), float(r)))

    status["n_near"] = len(vehicles)
    return {
        "camera": data.get("camera"),
        "lidar": data.get("lidar"),
        "bev": dm.render_local(ex, ey, yaw, BEV_SIZE, BEV_M_PER_PX, BEV_CAR_ROW_FRAC) if dm else None,
        "ego": {"speed_kmh": speed * 3.6, "x": ex, "y": ey, "yaw": yaw,
                "pitch": tf.rotation.pitch, "roll": tf.rotation.roll,
                "length": 2 * ebb.x, "width": 2 * ebb.y, "height": 2 * ebb.z},
        "control": {"throttle": ctl.throttle, "brake": ctl.brake, "steer": ctl.steer,
                    "hand_brake": ctl.hand_brake, "reverse": ctl.reverse, "gear": ctl.gear,
                    "road_wheel_deg": road_deg},
        "imu": imu_f,
        "vehicles": vehicles,
        "lead": lead,
        "landmarks": lms,
        "status": status,
        "camera_cfg": CAMERA_CFG,
    }


# ============================================================================ main
def main():
    p = argparse.ArgumentParser(description="TwinDrive Phase 1: drive with the dashboard")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--tm-port", type=int, default=8000)
    p.add_argument("--spawn", default="main_gate", help="landmark name to start at")
    p.add_argument("--vehicle", default="vehicle.tesla.model3")
    p.add_argument("--traffic", type=int, default=30)
    p.add_argument("--bikes", type=float, default=0.35, help="fraction of traffic that is two-wheelers")
    p.add_argument("--dt", type=float, default=0.05, help="simulation step (s)")
    p.add_argument("--lidar-pps", type=int, default=300000)
    p.add_argument("--no-camera", action="store_true")
    p.add_argument("--autopilot", action="store_true", help="start in autopilot")
    p.add_argument("--no-follow", action="store_true", help="don't move the CARLA spectator")
    p.add_argument("--scale", type=float, default=1.0, help="window scale, e.g. 0.8 for small screens")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    rng = random.Random(args.seed)
    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    world = client.get_world()
    cmap = world.get_map()
    if cmap.name.split("/")[-1] != DEFAULT_MAP:
        raise SystemExit(f"Loaded map is {cmap.name}; load '{DEFAULT_MAP}' first.")

    landmarks = load_landmarks()
    map_dir = Path(f"maps/{DEFAULT_MAP}")
    dm = DrivableMap(map_dir) if (map_dir / "drivable.npy").exists() else None
    if dm is None:
        print("NOTE: maps/cm2/drivable.npy missing, BEV map will be blank. "
              "Run tools/02_build_drivable_mask.py to create it.")

    original = world.get_settings()
    tm = client.get_trafficmanager(args.tm_port)
    ego, suite, traffic = None, None, []
    try:
        # ---------------------------------------------------- synchronous mode
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = args.dt
        world.apply_settings(settings)
        tm.set_synchronous_mode(True)
        tm.set_random_device_seed(args.seed)
        tm.set_global_distance_to_leading_vehicle(2.5)
        tm.global_percentage_speed_difference(30.0)  # campus: 30% under the limit

        # ---------------------------------------------------- actors
        ego = spawn_ego(world, cmap, landmarks, args.spawn, args.vehicle)
        world.tick()
        traffic = spawn_traffic(client, world, cmap, tm, args.traffic, args.bikes,
                                ego.get_location(), rng)
        suite = SensorSuite(world, ego, args.dt, args.lidar_pps, camera=not args.no_camera)
        world.tick()
        others = [a for a in world.get_actors().filter("vehicle.*") if a.id != ego.id]

        # ---------------------------------------------------- pygame
        pygame.display.init()
        pygame.font.init()
        win_size = (int(WINDOW_SIZE[0] * args.scale), int(WINDOW_SIZE[1] * args.scale))
        window = pygame.display.set_mode(win_size)
        pygame.display.set_caption("TwinDrive · dashboard")
        canvas = window if args.scale == 1.0 else pygame.Surface(WINDOW_SIZE)
        dash = Dashboard(canvas)
        clock = pygame.time.Clock()

        driver = KeyboardDriver()
        driver.autopilot = args.autopilot
        ego.set_autopilot(driver.autopilot, args.tm_port)
        spectator = world.get_spectator()
        imu_f = None
        fps_hist, last_wall = [], time.time()
        frame_count = 0

        running = True
        while running:
            ms = clock.tick()  # no cap: the simulator paces us
            # ------------------------------------------------ input
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        running = False
                    elif ev.key == pygame.K_p:
                        driver.autopilot = not driver.autopilot
                        ego.set_autopilot(driver.autopilot, args.tm_port)
                    elif ev.key == pygame.K_q:
                        driver.reverse = not driver.reverse
                    elif ev.key == pygame.K_h:
                        dash.show_help = not dash.show_help
            if not driver.autopilot:
                ego.apply_control(driver.control(pygame.key.get_pressed(), ms))

            # ------------------------------------------------ step the world
            frame = world.tick()
            try:
                data = suite.get(frame)
            except queue.Empty:
                print("WARNING: sensor data timed out for frame", frame)
                continue
            snap = world.get_snapshot()
            sim_time = snap.timestamp.elapsed_seconds

            # low-pass the IMU so collisions/bumps don't make the plot jump
            if "imu" in data:
                m = data["imu"]
                if imu_f is None:
                    imu_f = dict(m)
                else:
                    for k in ("ax", "ay", "az", "gz"):
                        imu_f[k] = 0.8 * imu_f[k] + 0.2 * m[k]
                    imu_f["compass"] = m["compass"]

            frame_count += 1
            if frame_count % 40 == 0:  # pick up vehicles that appeared/despawned
                others = [a for a in world.get_actors().filter("vehicle.*") if a.id != ego.id]

            if not args.no_follow:
                tf = ego.get_transform()
                f = tf.get_forward_vector()
                cam_loc = carla.Location(tf.location.x - 9.0 * f.x,
                                         tf.location.y - 9.0 * f.y,
                                         tf.location.z + 4.5)
                spectator.set_transform(carla.Transform(
                    cam_loc, carla.Rotation(pitch=-18.0, yaw=tf.rotation.yaw)))

            now = time.time()
            wall = max(now - last_wall, 1e-3)
            last_wall = now
            fps_hist = (fps_hist + [wall])[-30:]
            mean_wall = sum(fps_hist) / len(fps_hist)
            coll = None
            if suite.last_collision and sim_time - suite.last_collision[0] < 3.0:
                coll = suite.last_collision[1]
            status = {"mode": "AUTOPILOT" if driver.autopilot else "MANUAL",
                      "sim_time": sim_time, "frame": frame, "rtf": args.dt / mean_wall,
                      "client_fps": 1.0 / mean_wall, "map": DEFAULT_MAP, "collision": coll}

            # ------------------------------------------------ draw
            state = build_state(ego, others, data, imu_f, landmarks, dm, sim_time, status)
            dash.render(state)
            if canvas is not window:
                pygame.transform.smoothscale(canvas, win_size, window)
            pygame.display.flip()

    except KeyboardInterrupt:
        pass
    finally:
        # ---------------------------------------------------- always clean up
        print("Cleaning up ...")
        if suite:
            suite.destroy()
        ids = list(traffic) + ([ego.id] if ego else [])
        if ids:
            client.apply_batch_sync([carla.command.DestroyActor(i) for i in ids], True)
        tm.set_synchronous_mode(False)
        world.apply_settings(original)
        pygame.quit()
        print("Done: actors destroyed, simulator back in asynchronous mode.")


if __name__ == "__main__":
    main()
