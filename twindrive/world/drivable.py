# =============================================================================
# File        : twindrive/world/drivable.py
# Project     : TwinDrive — Lane-Free Autonomous Driving on a Digital Twin
#               of an Indian Campus (SRM Town / cm2, CARLA 0.9.16)
# Phase       : 1 — Ego vehicle and dashboard
# =============================================================================
"""
Description
-----------
Loads the global drivable-area mask built in Phase 0
(tools/02_build_drivable_mask.py) and cuts out a LOCAL, HEADING-UP view
around the car. This is the same idea a learned driving policy will use
later as its bird's-eye-view (BEV) observation; for now it feeds the
dashboard's map panel.

No CARLA import here: everything is plain numpy, so it can be unit-tested
and reused by training code.

Frames
    world  : CARLA x/y in metres (left-handed, y grows to the "right"/south)
    local  : fwd (metres ahead of the car), right (metres to its right)
    pixel  : row/col of the local image; car sits at (car_row, car_col)
"""

import json
import math
from pathlib import Path

import numpy as np

# dark-theme colours per mask class (0 offroad, 1 soft, 2 lane, 3 routable lane)
BEV_PALETTE = np.array([
    [16, 18, 22],
    [58, 54, 44],
    [92, 66, 34],
    [58, 62, 72],
], dtype=np.uint8)


class DrivableMap:
    """Global mask + fast heading-up local crops."""

    def __init__(self, map_dir):
        map_dir = Path(map_dir)
        self.mask = np.load(map_dir / "drivable.npy")
        meta = json.loads((map_dir / "drivable_meta.json").read_text())
        self.res = meta["resolution_m"]
        self.ox = meta["origin_x"]
        self.oy = meta["origin_y"]
        self.h, self.w = self.mask.shape
        self._grid_key = None

    # ------------------------------------------------------------------ grid
    def _grid(self, size, m_per_px, car_row, car_col):
        """Local fwd/right coordinates of every output pixel (cached)."""
        key = (size, m_per_px, car_row, car_col)
        if key != self._grid_key:
            rows, cols = np.mgrid[0:size, 0:size].astype(np.float32)
            self._fwd = (car_row - rows) * m_per_px
            self._right = (cols - car_col) * m_per_px
            self._grid_key = key
        return self._fwd, self._right

    # ------------------------------------------------------------------ crop
    def classes_local(self, ex, ey, yaw_deg, size=450, m_per_px=0.25, car_row_frac=0.68):
        """Class map (size x size) of the area around the car, heading-up."""
        car_row, car_col = int(size * car_row_frac), size // 2
        fwd, right = self._grid(size, m_per_px, car_row, car_col)
        yaw = math.radians(yaw_deg)
        c, s = math.cos(yaw), math.sin(yaw)
        # world = ego + fwd * forward_vector + right * right_vector
        # forward = (cos, sin), right = (-sin, cos) in CARLA's left-handed frame
        wx = ex + fwd * c - right * s
        wy = ey + fwd * s + right * c
        col = ((wx - self.ox) / self.res).astype(np.int32)
        row = ((wy - self.oy) / self.res).astype(np.int32)
        valid = (col >= 0) & (col < self.w) & (row >= 0) & (row < self.h)
        out = np.zeros((size, size), dtype=np.uint8)
        out[valid] = self.mask[row[valid], col[valid]]
        return out

    def render_local(self, ex, ey, yaw_deg, size=450, m_per_px=0.25, car_row_frac=0.68):
        """RGB image (size, size, 3) of the local crop, for display."""
        return BEV_PALETTE[self.classes_local(ex, ey, yaw_deg, size, m_per_px, car_row_frac)]


def world_to_local(wx, wy, ex, ey, yaw_deg):
    """World point(s) -> (fwd, right) relative to the car."""
    yaw = math.radians(yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = np.asarray(wx) - ex, np.asarray(wy) - ey
    return dx * c + dy * s, -dx * s + dy * c
