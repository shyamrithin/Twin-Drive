# =============================================================================
# File        : twindrive/world/landmarks.py
# Project     : TwinDrive — Lane-Free Autonomous Driving on a Digital Twin
#               of an Indian Campus (SRM Town / cm2, CARLA 0.9.16)
# Phase       : 1 — Ego vehicle and dashboard
# =============================================================================
"""
Description
-----------
Reads config/landmarks_<map>.json (written by tools/03_landmarks.py).
Returns plain dicts so this module has no CARLA dependency; callers that
need a carla.Transform build it themselves.

    load_landmarks()  ->  {"tech_park": {"label": "Tech Park", "x": .., "y": ..,
                                         "z": .., "yaw": ..}, ...}
"""

import json
from pathlib import Path

from twindrive import DEFAULT_MAP


def load_landmarks(path=None):
    path = Path(path or f"config/landmarks_{DEFAULT_MAP}.json")
    if not path.exists():
        return {}
    return json.loads(path.read_text()).get("landmarks", {})
