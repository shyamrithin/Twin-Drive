# =============================================================================
# File        : twindrive/world/__init__.py
# Project     : TwinDrive — Lane-Free Autonomous Driving on a Digital Twin
#               of an Indian Campus (SRM Town / cm2, CARLA 0.9.16)
# =============================================================================
"""World layer: everything about the static campus map.

None of these modules import carla themselves (road_graph only receives
carla objects as arguments), so training and visualisation code can import
them without a simulator running.
"""

from twindrive.world.drivable import DrivableMap, world_to_local  # noqa: F401
from twindrive.world.landmarks import load_landmarks  # noqa: F401
from twindrive.world.road_graph import (  # noqa: F401
    build_lane_graph,
    is_routable,
    lane_key,
    routable_lane_keys,
    strongly_connected_components,
)
