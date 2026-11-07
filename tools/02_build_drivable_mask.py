#!/usr/bin/env python3
# =============================================================================
# File        : tools/02_build_drivable_mask.py
# Project     : TwinDrive — Lane-Free Autonomous Driving on a Digital Twin
#               of an Indian Campus (SRM Town / cm2, CARLA 0.9.16)
# Phase       : 0 — Map readiness (step 3)
# Runs on     : WSL2 Ubuntu 22.04 (client)  ->  Windows CarlaUE4.exe (server)
# Env         : conda activate twindrive   (after `pip install -e .`)
# =============================================================================
"""
Description
-----------
Builds the GLOBAL drivable-area mask of campus: a top-down raster where every
pixel says what kind of surface it is. This is the foundation of TwinDrive's
lane-free driving: the agent reasons about drivable SPACE, not lane centres.

Classes (uint8)
    0  off-road            buildings, grass, lake
    1  soft drivable       shoulder / parking / bidirectional lanes
    2  lane, not routable  highway and dead-end roads (outside biggest SCC)
    3  lane, routable      inner campus, where route goals may be placed

Method
    CARLA gives a waypoint (centre point + lane width) every 1 m on every lane.
    For each waypoint and its successor 1 m ahead we draw a quadrilateral as
    wide as the lane. Neighbouring quads overlap and form continuous surface.
    Drawing order soft -> lane -> routable lane, so higher classes win.

Pixel convention
    col = (x - origin_x) / res,  row = (y - origin_y) / res
    CARLA y grows "down" in this image, so it matches our top-down plots.

Outputs (in --out, default maps/cm2/)
    drivable.npy          raw uint8 class array (not committed to git)
    drivable_meta.json    origin, resolution, shape, classes, map name
    drivable_preview.png  colour preview for humans

Usage
-----
    python tools/02_build_drivable_mask.py
    python tools/02_build_drivable_mask.py --res 0.25 --margin 30
"""

import argparse
import json
import time
from pathlib import Path

import carla
import numpy as np
from PIL import Image, ImageDraw

from twindrive import DEFAULT_MAP
from twindrive.world import is_routable, routable_lane_keys

OFFROAD, SOFT, LANE, LANE_ROUTABLE = 0, 1, 2, 3
SOFT_TYPES = {carla.LaneType.Shoulder, carla.LaneType.Parking,
              carla.LaneType.Bidirectional}
PREVIEW_COLOURS = np.array([
    [35, 35, 40],     # 0 off-road      dark grey
    [196, 164, 110],  # 1 soft          tan
    [235, 140, 40],   # 2 lane          orange
    [200, 200, 205],  # 3 routable lane light grey
], dtype=np.uint8)


def quad(a, b, origin_x, origin_y, res):
    """Four corners (in pixels) of the strip between waypoint a and b."""
    pts = []
    for w, sides in ((a, (-1, 1)), (b, (1, -1))):  # a-left, a-right, b-right, b-left
        loc = w.transform.location
        r = w.transform.get_right_vector()
        half = w.lane_width / 2.0
        for s in sides:
            px = (loc.x + s * r.x * half - origin_x) / res
            py = (loc.y + s * r.y * half - origin_y) / res
            pts.append((px, py))
    return pts


def soft_neighbours(wp, max_hops=4):
    """Shoulder/parking lanes beside wp, found by walking left/right.

    We track lane ids already seen because get_left_lane() on an
    opposite-direction lane points back toward the centre (a CARLA gotcha),
    which would otherwise make us walk in circles.
    """
    seen, frontier, found = {wp.lane_id}, [wp], []
    for _ in range(max_hops):
        nxt = []
        for w in frontier:
            for n in (w.get_left_lane(), w.get_right_lane()):
                if n is None or n.road_id != wp.road_id or n.lane_id in seen:
                    continue
                seen.add(n.lane_id)
                if n.lane_type in SOFT_TYPES:
                    found.append(n)
                nxt.append(n)
        frontier = nxt
    return found


def main():
    parser = argparse.ArgumentParser(description="TwinDrive: build global drivable mask")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--res", type=float, default=0.5, help="metres per pixel")
    parser.add_argument("--margin", type=float, default=20.0, help="border in metres")
    parser.add_argument("--out", default=f"maps/{DEFAULT_MAP}")
    args = parser.parse_args()

    t0 = time.time()
    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    world = client.get_world()
    cmap = world.get_map()
    map_short = cmap.name.split("/")[-1]
    print(f"Map: {cmap.name}")
    if map_short != DEFAULT_MAP:
        print(f"WARNING: expected '{DEFAULT_MAP}', got '{map_short}'.")

    # ---------------------------------------------------- 1. routable lanes
    routable = routable_lane_keys(cmap)
    print(f"[1] routable lane segments: {len(routable)}")

    # ---------------------------------------------------- 2. waypoints + bounds
    wps = cmap.generate_waypoints(1.0)  # driving lanes only
    xs = np.array([w.transform.location.x for w in wps])
    ys = np.array([w.transform.location.y for w in wps])
    origin_x, origin_y = xs.min() - args.margin, ys.min() - args.margin
    width = int(np.ceil((xs.max() + args.margin - origin_x) / args.res))
    height = int(np.ceil((ys.max() + args.margin - origin_y) / args.res))
    print(f"[2] driving waypoints @1 m: {len(wps)}   raster: {width} x {height} px "
          f"({width * args.res:.0f} x {height * args.res:.0f} m)")

    # ---------------------------------------------------- 3. collect polygons
    polys = {SOFT: [], LANE: [], LANE_ROUTABLE: []}
    soft_seen = set()
    widths = []
    for w in wps:
        widths.append(w.lane_width)
        cls = LANE_ROUTABLE if is_routable(w, routable) else LANE
        for n in w.next(1.0):
            polys[cls].append(quad(w, n, origin_x, origin_y, args.res))
        for s in soft_neighbours(w):
            key = (s.road_id, s.section_id, s.lane_id, round(s.s, 1))
            if key in soft_seen:
                continue
            soft_seen.add(key)
            for sn in s.next(1.0):
                polys[SOFT].append(quad(s, sn, origin_x, origin_y, args.res))
    print(f"[3] polygons: soft={len(polys[SOFT])}  lane={len(polys[LANE])}  "
          f"routable={len(polys[LANE_ROUTABLE])}")

    # ---------------------------------------------------- 4. rasterise
    img = Image.new("L", (width, height), OFFROAD)
    draw = ImageDraw.Draw(img)
    for cls in (SOFT, LANE, LANE_ROUTABLE):  # later classes overwrite earlier
        for p in polys[cls]:
            draw.polygon(p, fill=cls)
    mask = np.array(img, dtype=np.uint8)

    # ---------------------------------------------------- 5. save
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "drivable.npy", mask)
    Image.fromarray(PREVIEW_COLOURS[mask]).save(out / "drivable_preview.png")
    meta = {
        "map": cmap.name,
        "server_version": client.get_server_version(),
        "resolution_m": args.res,
        "origin_x": float(origin_x),
        "origin_y": float(origin_y),
        "width_px": width,
        "height_px": height,
        "pixel_convention": "col=(x-origin_x)/res, row=(y-origin_y)/res",
        "classes": {"0": "offroad", "1": "soft_drivable",
                    "2": "lane_not_routable", "3": "lane_routable"},
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out / "drivable_meta.json").write_text(json.dumps(meta, indent=2))

    # ---------------------------------------------------- 6. stats
    px_area = args.res ** 2
    widths = np.array(widths)
    print("\n[6] Surface area")
    for cls, name in ((SOFT, "soft drivable"), (LANE, "lane, not routable"),
                      (LANE_ROUTABLE, "lane, routable")):
        print(f"    {name:20s}: {np.count_nonzero(mask == cls) * px_area / 1e4:7.2f} ha")
    print("    Lane widths (m): "
          f"min {widths.min():.2f}  median {np.median(widths):.2f}  max {widths.max():.2f}")
    print(f"    lanes narrower than 3.0 m: {100 * np.mean(widths < 3.0):.1f}% of waypoints")
    print(f"\nSaved to {out}/  (took {time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
