#!/usr/bin/env python3
# =============================================================================
# File        : 01_compare_maps.py
# Project     : TwinDrive — Lane-Free Autonomous Driving on a Digital Twin
#               of an Indian Campus (SRM Town, CARLA 0.9.16)
# Phase       : 0 — Map readiness audit (step 2)
# Runs on     : WSL2 Ubuntu 22.04 (client)  ->  Windows CarlaUE4.exe (server)
# Env         : conda activate twindrive
# =============================================================================
"""
Description
-----------
The packaged build contains two campus maps: `cm2` and `college map`.
This script loads each one in turn, audits it, and saves a plot, so we can
decide which map TwinDrive is built on.

For each map it reports:
  1. Size of the map (bounding box of all driving lanes, in metres).
  2. Spawn points, road segments, approximate total lane length.
  3. Dead ends: lane segments that lead nowhere. A car routed into one
     gets stuck, so the route planner must avoid them.
  4. Routability: the road graph is directed (one-way lanes), so we compute
     its strongly connected components (SCCs). Two places in the SAME SCC can
     reach each other by car; the biggest SCC is the part of campus where
     "drive from any landmark to any other" actually works.

Plot legend (maps_<name>.png):
  grey   = lanes in the biggest SCC (fully routable)
  orange = lanes outside it (reachable one way only, or isolated)
  blue x = dead ends
  red    = spawn points

It LOADS maps (that resets the world) but spawns nothing, and it reloads
the map that was open when it started.

Usage
-----
    conda activate twindrive
    python 01_compare_maps.py                 # cm2 + college map
    python 01_compare_maps.py --include-town10  # also the stock Town10HD_Opt
"""

import argparse
import sys
import time

import carla
import matplotlib

matplotlib.use("Agg")  # no display needed in WSL
import matplotlib.pyplot as plt


# ----------------------------------------------------------------- graph helpers
def lane_key(wp):
    """Identify one lane segment: (road, section, lane)."""
    return (wp.road_id, wp.section_id, wp.lane_id)


def strongly_connected_components(graph):
    """Iterative Kosaraju. graph: dict node -> set(successor nodes)."""
    nodes = set(graph)
    for succs in graph.values():
        nodes |= succs
    # pass 1: finishing order on the original graph
    visited, order = set(), []
    for start in nodes:
        if start in visited:
            continue
        stack = [(start, iter(graph.get(start, ())))]
        visited.add(start)
        while stack:
            node, it = stack[-1]
            nxt = next(it, None)
            if nxt is None:
                stack.pop()
                order.append(node)
            elif nxt not in visited:
                visited.add(nxt)
                stack.append((nxt, iter(graph.get(nxt, ()))))
    # pass 2: DFS on the reversed graph in reverse finishing order
    reverse = {n: set() for n in nodes}
    for n, succs in graph.items():
        for s in succs:
            reverse[s].add(n)
    assigned, components = set(), []
    for start in reversed(order):
        if start in assigned:
            continue
        comp, stack = [], [start]
        assigned.add(start)
        while stack:
            n = stack.pop()
            comp.append(n)
            for p in reverse[n]:
                if p not in assigned:
                    assigned.add(p)
                    stack.append(p)
        components.append(comp)
    return components


# ----------------------------------------------------------------- audit
def audit_map(world, out_png):
    cmap = world.get_map()
    spawn_points = cmap.get_spawn_points()
    topology = cmap.get_topology()
    waypoints = cmap.generate_waypoints(2.0)

    # bounding box
    xs = [w.transform.location.x for w in waypoints]
    ys = [w.transform.location.y for w in waypoints]
    width, height = max(xs) - min(xs), max(ys) - min(ys)

    # directed lane graph + dead ends
    graph, dead_ends = {}, []
    for entry_wp, exit_wp in topology:
        k = lane_key(entry_wp)
        graph.setdefault(k, set())
        successors = exit_wp.next(1.0)
        if not successors:
            dead_ends.append(exit_wp.transform.location)
        for s in successors:
            sk = lane_key(s)
            if sk != k:
                graph[k].add(sk)
    comps = sorted(strongly_connected_components(graph), key=len, reverse=True)
    biggest = set(comps[0]) if comps else set()
    n_lanes = len({k for k in graph} | {s for v in graph.values() for s in v})

    stats = {
        "map": cmap.name,
        "size_m": f"{width:.0f} x {height:.0f}",
        "spawn_points": len(spawn_points),
        "segments": len(topology),
        "lane_km": round(len(waypoints) * 2.0 / 1000.0, 1),
        "dead_ends": len(dead_ends),
        "sccs": len(comps),
        "biggest_scc_pct": round(100.0 * len(biggest) / max(n_lanes, 1), 1),
    }

    # plot
    fig, ax = plt.subplots(figsize=(10, 10))
    in_x = [w.transform.location.x for w in waypoints if lane_key(w) in biggest]
    in_y = [w.transform.location.y for w in waypoints if lane_key(w) in biggest]
    out_x = [w.transform.location.x for w in waypoints if lane_key(w) not in biggest]
    out_y = [w.transform.location.y for w in waypoints if lane_key(w) not in biggest]
    ax.scatter(in_x, in_y, s=0.5, c="grey", label="routable (biggest SCC)")
    ax.scatter(out_x, out_y, s=1.5, c="orange", label="outside biggest SCC")
    ax.scatter([d.x for d in dead_ends], [d.y for d in dead_ends],
               s=40, c="blue", marker="x", label="dead ends")
    ax.scatter([p.location.x for p in spawn_points], [p.location.y for p in spawn_points],
               s=10, c="red", label="spawn points")
    ax.set_aspect("equal")
    ax.invert_yaxis()  # CARLA is left-handed: flip Y to match a top-down view
    ax.set_title(f"{cmap.name} — routability audit")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return stats


# ----------------------------------------------------------------- main
def short(name):
    return name.rstrip("/").split("/")[-1]


def main():
    parser = argparse.ArgumentParser(description="TwinDrive Phase 0: compare campus maps")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--include-town10", action="store_true")
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(180.0)  # map loads are slow on a laptop GPU
    world = client.get_world()
    original = short(world.get_map().name)

    maps = sorted(client.get_available_maps())
    if not args.include_town10:
        maps = [m for m in maps if not short(m).startswith("Town")]
    print(f"Maps to audit: {[short(m) for m in maps]}  (currently loaded: {original})\n")

    results = []
    for m in maps:
        name = short(m)
        try:
            if name != short(world.get_map().name):
                print(f"Loading '{name}' ... (this can take a minute)")
                world = client.load_world(m)
                time.sleep(3.0)  # let the server settle after a load
            png = f"maps_{name.replace(' ', '_')}.png"
            stats = audit_map(world, png)
            stats["plot"] = png
            results.append(stats)
            print(f"  done -> {png}")
        except RuntimeError as e:
            print(f"  FAILED on '{name}': {e}")

    # restore the map that was open at the start
    if short(world.get_map().name) != original:
        print(f"\nReloading original map '{original}' ...")
        match = [m for m in client.get_available_maps() if short(m) == original]
        if match:
            client.load_world(match[0])

    print("\n================ SUMMARY ================")
    for r in results:
        print(f"\n{r['map']}")
        for k in ("size_m", "spawn_points", "segments", "lane_km",
                  "dead_ends", "sccs", "biggest_scc_pct", "plot"):
            print(f"  {k:16s}: {r[k]}")
    if not results:
        sys.exit(1)


if __name__ == "__main__":
    main()
