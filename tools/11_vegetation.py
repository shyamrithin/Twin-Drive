#!/usr/bin/env python3
# =============================================================================
# File        : tools/11_vegetation.py
# Project     : TwinDrive — Lane-Free Autonomous Driving on a Digital Twin
#               of an Indian Campus (SRM Town / cm2, CARLA 0.9.16)
# Phase       : 1 — Performance
# Runs on     : WSL2 Ubuntu 22.04 (client)  ->  Windows CarlaUE4.exe (server)
# Env         : conda activate twindrive
# =============================================================================
"""
Description
-----------
Inspect and hide/show the campus vegetation at runtime (see
twindrive/world/vegetation.py for how it works).

Commands
    list        what static objects the map has, by semantic label, and which
                ones look like vegetation (by label or by name)
    hide        disable all vegetation (stays hidden until `show` or a CARLA
                restart)
    show        enable it again
    layers      also try unloading CARLA's Foliage map layer (only works on
                layered "_Opt" maps; harmless otherwise)

Options
    --label-only   match only objects labelled Vegetation (ignore names)

Usage
-----
    python tools/11_vegetation.py list
    python tools/11_vegetation.py hide
    python tools/11_vegetation.py show
"""

import argparse

import carla

from twindrive.world.vegetation import (all_objects, find_vegetation, label_counts,
                                        name_counts, set_visible)


def main():
    p = argparse.ArgumentParser(description="TwinDrive: campus vegetation tool")
    p.add_argument("cmd", choices=["list", "hide", "show", "layers"])
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--label-only", action="store_true")
    args = p.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    world = client.get_world()
    print(f"Map: {world.get_map().name}")
    objs = all_objects(world)
    veg = find_vegetation(world, use_names=not args.label_only, objects=objs)

    if args.cmd == "list":
        print(f"\nStatic objects: {len(objs)}")
        for lbl, n in label_counts(objs).most_common():
            print(f"  {lbl:24s} {n}")
        print(f"\nLooks like vegetation: {len(veg)} objects")
        print("  labels:", dict(label_counts(veg)))
        print("  most common names:")
        for name, n in name_counts(veg):
            print(f"    {n:6d}  {name}")
        print("\nMost common names overall (to spot trees with odd names):")
        for name, n in name_counts(objs, top=20):
            print(f"    {n:6d}  {name}")
    elif args.cmd in ("hide", "show"):
        n = set_visible(world, veg, args.cmd == "show")
        world.tick() if world.get_settings().synchronous_mode else world.wait_for_tick()
        print(f"{'Enabled' if args.cmd == 'show' else 'Disabled'} {n} vegetation objects.")
    elif args.cmd == "layers":
        try:
            world.unload_map_layer(carla.MapLayer.Foliage)
            print("Requested unload of the Foliage map layer "
                  "(only has an effect on layered _Opt maps).")
        except RuntimeError as e:
            print(f"Layer unload not supported here: {e}")


if __name__ == "__main__":
    main()
