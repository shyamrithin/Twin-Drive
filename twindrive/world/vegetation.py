# =============================================================================
# File        : twindrive/world/vegetation.py
# Project     : TwinDrive — Lane-Free Autonomous Driving on a Digital Twin
#               of an Indian Campus (SRM Town / cm2, CARLA 0.9.16)
# Phase       : 1 — Performance
# =============================================================================
"""
Description
-----------
Finds the campus trees and hides/shows them at RUNTIME, without editing the
map in Unreal. On this map the dense tree foliage makes every camera render
cost ~270 ms, so hiding vegetation is the biggest single speed-up available.

How
    world.get_environment_objects() lists the map's static objects with a
    semantic label (CityObjectLabel) and a name. We pick:
      * everything labelled Vegetation, and
      * anything whose name looks like a plant (custom-imported meshes are
        often unlabelled, so the label alone may miss them).
    world.enable_environment_objects(ids, False) then disables them: they stop
    rendering and stop colliding (LiDAR no longer sees them either). The change
    lasts until enabled again or CARLA restarts; the map file is untouched.

Note: this module needs carla, so it is NOT imported by twindrive.world's
__init__ (which stays simulator-free). Import it directly.
"""

import re
from collections import Counter

import carla

# "grass" is deliberately NOT here: lawn/ground meshes are often called grass,
# and hiding them would leave holes in the terrain.
PLANT_WORDS = ("tree", "foliage", "leaf", "leaves", "plant", "bush", "shrub",
               "palm", "hedge", "veg", "branch")


def all_objects(world):
    return world.get_environment_objects(carla.CityObjectLabel.Any)


def find_vegetation(world, use_names=True, objects=None):
    objs = objects if objects is not None else all_objects(world)
    hits = []
    for o in objs:
        name = o.name.lower()
        if o.type == carla.CityObjectLabel.Vegetation or \
                (use_names and any(w in name for w in PLANT_WORDS)):
            hits.append(o)
    return hits


def set_visible(world, objects, visible):
    ids = {o.id for o in objects}
    if ids:
        world.enable_environment_objects(ids, visible)
    return len(ids)


def label_counts(objects):
    return Counter(str(o.type) for o in objects)


def name_counts(objects, top=15):
    """Most common object names, with trailing numbers stripped (Tree_12 -> Tree)."""
    base = Counter(re.sub(r"[_\-\s]*\d+$", "", o.name) for o in objects)
    return base.most_common(top)
