# =============================================================================
# File        : twindrive/__init__.py
# Project     : TwinDrive — Lane-Free Autonomous Driving on a Digital Twin
#               of an Indian Campus (SRM Town / cm2, CARLA 0.9.16)
# =============================================================================
"""
TwinDrive core package.

Sub-packages (grow phase by phase):
    twindrive.world   — map, road graph, drivable-area mask      (Phase 0)
    twindrive.sensors — ego sensor suite                         (Phase 1)
    twindrive.traffic — Indian campus traffic module             (Phase 2)
    twindrive.perception, twindrive.planning, twindrive.learning (later)

Rule: nothing in this package imports ROS. ROS is a thin layer on top.
"""

__version__ = "0.0.1"

DEFAULT_MAP = "cm2"
