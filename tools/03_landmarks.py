#!/usr/bin/env python3
# =============================================================================
# File        : tools/03_landmarks.py
# Project     : TwinDrive — Lane-Free Autonomous Driving on a Digital Twin
#               of an Indian Campus (SRM Town / cm2, CARLA 0.9.16)
# Phase       : 0 — Map readiness (step 4)
# Runs on     : WSL2 Ubuntu 22.04 (client)  ->  Windows CarlaUE4.exe (server)
# Env         : conda activate twindrive
# =============================================================================
"""
Description
-----------
Builds the list of named campus landmarks that TwinDrive uses as route goals
("drive to Tech Park"). You fly the CARLA spectator camera over a place, then
run `add`; the tool snaps that spot to the NEAREST ROUTABLE driving lane (so a
goal can never sit on the highway or a dead end) and saves it.

Stored in config/landmarks_cm2.json:
    name -> label, snapped x/y/z/yaw, road_id, lane_id, snap distance

Commands
    add <name> "<Label>"   save the spot under the spectator camera
        --at X Y             ...or use a known map point instead
        --force              allow overwriting an existing name (refused otherwise)
    list                   print all landmarks
    remove <name>          delete one
    show [--seconds N]     draw all landmark labels inside the CARLA window
    plot                   overlay landmarks on maps/cm2/drivable_preview.png
                           -> maps/cm2/landmarks.png

Flying the spectator (click inside the CARLA window first)
    W/A/S/D  move      Q/E  down/up      hold right mouse + move  look around
    Shift    faster

Usage
-----
    python tools/03_landmarks.py add tech_park "Tech Park"
    python tools/03_landmarks.py show
    python tools/03_landmarks.py plot
"""

import argparse
import json
import math
import re
import time
from pathlib import Path

import carla
import numpy as np

from twindrive import DEFAULT_MAP
from twindrive.world import is_routable, routable_lane_keys

LANDMARK_FILE = Path(f"config/landmarks_{DEFAULT_MAP}.json")
MASK_DIR = Path(f"maps/{DEFAULT_MAP}")
SNAP_WARN_M = 30.0


# ----------------------------------------------------------------- storage
def load_db():
    if LANDMARK_FILE.exists():
        return json.loads(LANDMARK_FILE.read_text())
    return {"map": DEFAULT_MAP, "landmarks": {}}


def save_db(db):
    LANDMARK_FILE.parent.mkdir(parents=True, exist_ok=True)
    db["landmarks"] = dict(sorted(db["landmarks"].items()))
    LANDMARK_FILE.write_text(json.dumps(db, indent=2))


# ----------------------------------------------------------------- carla
def connect(args):
    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    world = client.get_world()
    name = world.get_map().name.split("/")[-1]
    if name != DEFAULT_MAP:
        raise SystemExit(f"Loaded map is '{name}', expected '{DEFAULT_MAP}'.")
    return world


def routable_candidates(cmap):
    """All routable driving waypoints every 1 m, plus their XY as an array."""
    routable = routable_lane_keys(cmap)
    wps = [w for w in cmap.generate_waypoints(1.0) if is_routable(w, routable)]
    xy = np.array([[w.transform.location.x, w.transform.location.y] for w in wps])
    return wps, xy


# ----------------------------------------------------------------- commands
def cmd_add(args):
    if not re.fullmatch(r"[a-z0-9_]+", args.name):
        raise SystemExit("name must be lowercase letters, digits, underscores (e.g. tech_park)")
    db = load_db()
    if args.name in db["landmarks"] and not args.force:
        old = db["landmarks"][args.name]["label"]
        raise SystemExit(f"'{args.name}' already exists (label '{old}'). "
                         "Pick a new name, or add --force to overwrite it.")
    world = connect(args)
    cmap = world.get_map()
    if args.at:
        cam = carla.Location(x=args.at[0], y=args.at[1], z=0.0)
        print(f"Using given point x={cam.x:.1f} y={cam.y:.1f}")
    else:
        cam = world.get_spectator().get_transform().location
        print(f"Spectator at x={cam.x:.1f} y={cam.y:.1f} z={cam.z:.1f}")

    # nearest routable waypoint in 2D (the camera is high above the road,
    # so we ignore height on purpose)
    wps, xy = routable_candidates(cmap)
    d = np.hypot(xy[:, 0] - cam.x, xy[:, 1] - cam.y)
    i = int(np.argmin(d))
    wp, dist = wps[i], float(d[i])
    t = wp.transform
    print(f"Snapped to road {wp.road_id} lane {wp.lane_id} at "
          f"x={t.location.x:.1f} y={t.location.y:.1f}  ({dist:.1f} m away)")
    if dist > SNAP_WARN_M:
        print(f"WARNING: nearest routable lane is {dist:.0f} m away. "
              "Fly closer to the road in front of the landmark and re-run.")

    if args.name in db["landmarks"]:
        print(f"(overwriting existing '{args.name}' because of --force)")
    db["landmarks"][args.name] = {
        "label": args.label,
        "x": round(t.location.x, 2), "y": round(t.location.y, 2),
        "z": round(t.location.z, 2), "yaw": round(t.rotation.yaw, 1),
        "road_id": wp.road_id, "lane_id": wp.lane_id,
        "snap_dist_m": round(dist, 1),
        "added": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_db(db)
    draw_landmark(world, args.label, t, seconds=30.0)
    print(f"Saved '{args.name}' -> {LANDMARK_FILE}  (marker visible in CARLA for 30 s)")


def cmd_list(_args):
    lms = load_db()["landmarks"]
    if not lms:
        print("No landmarks yet.")
        return
    print(f"{'name':22s} {'label':28s} {'x':>8s} {'y':>8s} {'snap m':>7s}")
    for n, v in lms.items():
        print(f"{n:22s} {v['label']:28s} {v['x']:8.1f} {v['y']:8.1f} {v['snap_dist_m']:7.1f}")
    print(f"\n{len(lms)} landmarks")


def cmd_remove(args):
    db = load_db()
    if db["landmarks"].pop(args.name, None) is None:
        raise SystemExit(f"No landmark '{args.name}'.")
    save_db(db)
    print(f"Removed '{args.name}'.")


def draw_landmark(world, label, transform, seconds):
    loc = transform.location
    world.debug.draw_point(loc + carla.Location(z=0.5), size=0.4,
                           color=carla.Color(255, 40, 40), life_time=seconds)
    # an arrow shows the lane's driving direction at the goal
    yaw = math.radians(transform.rotation.yaw)
    tip = loc + carla.Location(x=6 * math.cos(yaw), y=6 * math.sin(yaw), z=0.5)
    world.debug.draw_arrow(loc + carla.Location(z=0.5), tip, thickness=0.2,
                           arrow_size=0.6, color=carla.Color(255, 40, 40),
                           life_time=seconds)
    world.debug.draw_string(loc + carla.Location(z=4.0), label,
                            color=carla.Color(255, 255, 0), life_time=seconds)


def cmd_show(args):
    world = connect(args)
    lms = load_db()["landmarks"]
    for v in lms.values():
        t = carla.Transform(carla.Location(v["x"], v["y"], v["z"]),
                            carla.Rotation(yaw=v["yaw"]))
        draw_landmark(world, v["label"], t, seconds=args.seconds)
    print(f"Drew {len(lms)} landmarks in CARLA for {args.seconds:.0f} s.")


def cmd_plot(_args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    meta = json.loads((MASK_DIR / "drivable_meta.json").read_text())
    img = Image.open(MASK_DIR / "drivable_preview.png")
    res, ox, oy = meta["resolution_m"], meta["origin_x"], meta["origin_y"]
    lms = load_db()["landmarks"]

    fig, ax = plt.subplots(figsize=(14, 11))
    ax.imshow(img)
    for v in lms.values():
        col, row = (v["x"] - ox) / res, (v["y"] - oy) / res
        ax.plot(col, row, "o", ms=9, mfc="red", mec="white", mew=1.5)
        ax.annotate(v["label"], (col, row), xytext=(8, -8), textcoords="offset points",
                    color="yellow", fontsize=10, weight="bold")
    ax.set_axis_off()
    ax.set_title(f"{DEFAULT_MAP} — campus landmarks ({len(lms)})")
    out = MASK_DIR / "landmarks.png"
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"Saved {out}")


# ----------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description="TwinDrive: campus landmark manager")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add")
    a.add_argument("name")
    a.add_argument("label")
    a.add_argument("--at", nargs=2, type=float, metavar=("X", "Y"),
                   help="use this map point instead of the spectator position")
    a.add_argument("--force", action="store_true",
                   help="overwrite an existing landmark with the same name")
    a.set_defaults(func=cmd_add)

    sub.add_parser("list").set_defaults(func=cmd_list)

    r = sub.add_parser("remove")
    r.add_argument("name")
    r.set_defaults(func=cmd_remove)

    s = sub.add_parser("show")
    s.add_argument("--seconds", type=float, default=120.0)
    s.set_defaults(func=cmd_show)

    sub.add_parser("plot").set_defaults(func=cmd_plot)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
