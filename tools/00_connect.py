#!/usr/bin/env python3
# =============================================================================
# File        : 00_connect.py
# Project     : TwinDrive — Lane-Free Autonomous Driving on a Digital Twin
#               of an Indian Campus (SRM Town, CARLA 0.9.16)
# Phase       : 0 — Map readiness audit
# Runs on     : WSL2 Ubuntu 22.04 (client)  ->  Windows CarlaUE4.exe (server)
# =============================================================================
"""
Description
-----------
First contact with the SRM Town server. This script is READ-ONLY: it spawns
nothing and changes nothing in the world.

It checks:
  1. Client/server connection and that both run the same CARLA version.
  2. Which map is loaded and which maps the packaged build contains.
  3. Road network health: spawn points, topology (road segments), waypoints.
  4. Whether a pedestrian navigation mesh exists (needed for CARLA's built-in
     walker AI; our own campus traffic module will not depend on it).
  5. Which vehicle / walker / sensor blueprints are available.
  6. Saves a top-down plot of the road network + spawn points to
     srm_town_topology.png so you can compare it with Google Maps.

Usage
-----
    source .venv/bin/activate
    python 00_connect.py                 # defaults: localhost:2000
    python 00_connect.py --host 127.0.0.1 --port 2000
"""

import argparse
import sys

import carla


def main():
    parser = argparse.ArgumentParser(description="TwinDrive Phase 0 connection audit")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--no-plot", action="store_true", help="skip the PNG plot")
    args = parser.parse_args()

    # ---------------------------------------------------------------- 1. connect
    print(f"[1] Connecting to CARLA at {args.host}:{args.port} ...")
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    try:
        server_ver = client.get_server_version()
    except RuntimeError as e:
        print(f"    FAILED: {e}")
        print("    Is CarlaUE4.exe running on Windows? Does `nc -zv localhost 2000` succeed?")
        sys.exit(1)
    client_ver = client.get_client_version()
    print(f"    client version : {client_ver}")
    print(f"    server version : {server_ver}")
    if client_ver != server_ver:
        print("    WARNING: versions differ — install the matching `carla` pip package.")

    # ---------------------------------------------------------------- 2. maps
    world = client.get_world()
    cmap = world.get_map()
    print(f"\n[2] Loaded map   : {cmap.name}")
    print("    Maps in this build:")
    for m in sorted(client.get_available_maps()):
        print(f"      - {m}")

    # ---------------------------------------------------------------- 3. roads
    spawn_points = cmap.get_spawn_points()
    topology = cmap.get_topology()          # list of (entry_wp, exit_wp) pairs
    waypoints = cmap.generate_waypoints(2.0)  # one waypoint every 2 m on all lanes
    print("\n[3] Road network")
    print(f"    spawn points        : {len(spawn_points)}")
    print(f"    topology segments   : {len(topology)}")
    print(f"    waypoints @ 2 m     : {len(waypoints)}")
    road_ids = {wp.road_id for wp in waypoints}
    junction_wps = sum(1 for wp in waypoints if wp.is_junction)
    print(f"    unique road ids     : {len(road_ids)}")
    print(f"    waypoints in junctions: {junction_wps}")

    # ---------------------------------------------------------------- 4. walkers
    print("\n[4] Pedestrian navigation mesh")
    nav_loc = world.get_random_location_from_navigation()
    if nav_loc is None:
        print("    NOT FOUND — built-in walker AI unavailable (fine: we'll drive walkers ourselves).")
    else:
        print(f"    FOUND — sample sidewalk point: ({nav_loc.x:.1f}, {nav_loc.y:.1f}, {nav_loc.z:.1f})")

    # ---------------------------------------------------------------- 5. blueprints
    bp_lib = world.get_blueprint_library()
    vehicles = bp_lib.filter("vehicle.*")
    two_wheelers = [b for b in vehicles if b.has_attribute("number_of_wheels")
                    and int(b.get_attribute("number_of_wheels")) == 2]
    walkers = bp_lib.filter("walker.pedestrian.*")
    sensors = bp_lib.filter("sensor.*")
    print("\n[5] Blueprints")
    print(f"    vehicles            : {len(vehicles)}  (two-wheelers: {len(two_wheelers)})")
    for b in two_wheelers:
        print(f"      - {b.id}")
    print(f"    pedestrians         : {len(walkers)}")
    print(f"    sensors             : {len(sensors)}")
    for s in ("sensor.lidar.ray_cast", "sensor.lidar.ray_cast_semantic",
              "sensor.camera.rgb", "sensor.other.gnss", "sensor.other.imu"):
        print(f"      {s:32s}: {'yes' if bp_lib.filter(s) else 'NO'}")

    # ---------------------------------------------------------------- 6. plot
    if not args.no_plot:
        import matplotlib
        matplotlib.use("Agg")  # no display needed in WSL
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 10))
        xs = [wp.transform.location.x for wp in waypoints]
        ys = [wp.transform.location.y for wp in waypoints]
        ax.scatter(xs, ys, s=0.5, c="grey", label="lane waypoints (2 m)")
        sx = [sp.location.x for sp in spawn_points]
        sy = [sp.location.y for sp in spawn_points]
        ax.scatter(sx, sy, s=12, c="red", label="spawn points")
        ax.set_aspect("equal")
        ax.invert_yaxis()  # CARLA is left-handed: flip Y so it matches a top-down map view
        ax.set_title(f"{cmap.name} — road network")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig("srm_town_topology.png", dpi=150)
        print("\n[6] Saved road-network plot -> srm_town_topology.png")

    print("\nDone. Nothing was spawned or modified.")


if __name__ == "__main__":
    main()
