# =============================================================================
# File        : twindrive/planning/route_planner.py
# Project     : TwinDrive — Lane-Free Autonomous Driving on a Digital Twin
#               of an Indian Campus (SRM Town / cm2, CARLA 0.9.16)
# Phase       : 1 — Autonomous route following (step 2)
# =============================================================================
"""
Description
-----------
TwinDrive's own global route planner. Given where the car is and a goal
(usually a landmark), it returns a dense path: one point every `step`
metres along the lanes to follow.

How it works
    1. The lane graph from Phase 0: node = one lane segment
       (road, section, lane), edge = "driving off the end of A lands on B".
    2. Dijkstra's algorithm finds the cheapest sequence of lane segments.
       The cost of entering a segment is its length in metres, so the result
       is the shortest drivable route that obeys lane directions.
    3. The sequence is expanded into dense waypoints with CARLA's
       Waypoint.next_until_lane_end(), cut at the car on the first lane and
       at the goal on the last.

Limitations (fine for now, revisited later)
    * No lane changes: on multi-lane roads the route stays in its lane and may
      loop round a block instead of changing lane.
    * Distances only; no traffic, turn penalties or speed limits in the cost.

Output of plan(): float64 array (N, 3) of x, y, yaw (degrees) in the world frame.
"""

import heapq
import itertools
import math

import carla
import numpy as np

from twindrive.world.road_graph import build_lane_graph, lane_key


class RoutePlanner:
    def __init__(self, cmap, step=1.0):
        self.cmap = cmap
        self.step = step
        self.graph, _ = build_lane_graph(cmap)
        topo = cmap.get_topology()
        self.entry = {}
        for a, _b in topo:
            self.entry.setdefault(lane_key(a), a)
        for _a, b in topo:  # successor lanes that start mid-road
            for s in b.next(1.0):
                self.entry.setdefault(lane_key(s), s)
        self._dense = {}

    # ------------------------------------------------------------------ lanes
    def _dense_lane(self, key):
        """All waypoints of one lane segment, from its entry to its end."""
        if key not in self._dense:
            e = self.entry[key]
            self._dense[key] = [e] + list(e.next_until_lane_end(self.step))
        return self._dense[key]

    def _length(self, key):
        return max(len(self._dense_lane(key)), 1) * self.step

    def _successors(self, key, last_wp=None):
        """Lane segments reachable by driving off the end of `key`."""
        end_wp = last_wp or self._dense_lane(key)[-1]
        succ = set(self.graph.get(key, ()))
        if not succ:  # key not in the topology graph: ask the map directly
            succ = {lane_key(s) for s in end_wp.next(1.0)}
        succ.discard(key)
        for s in list(succ):  # every successor needs an entry waypoint
            if s not in self.entry:
                for n in end_wp.next(1.0):
                    if lane_key(n) == s:
                        self.entry[s] = n
                if s not in self.entry:
                    succ.discard(s)
        return succ

    # ------------------------------------------------------------------ plan
    def plan(self, start_loc, goal_loc):
        drv = carla.LaneType.Driving
        s_wp = self.cmap.get_waypoint(start_loc, project_to_road=True, lane_type=drv)
        g_wp = self.cmap.get_waypoint(goal_loc, project_to_road=True, lane_type=drv)
        if s_wp is None or g_wp is None:
            raise RuntimeError("start or goal is not near a driving lane")
        sk, gk = lane_key(s_wp), lane_key(g_wp)
        g_loc = g_wp.transform.location

        start_seq = [s_wp] + list(s_wp.next_until_lane_end(self.step))

        # goal further along the lane we are already on?
        if sk == gk:
            d = [w.transform.location.distance(g_loc) for w in start_seq]
            i = int(np.argmin(d))
            if d[i] < 3.0 and i > 2:
                return self._to_array(start_seq[: i + 1])

        # Dijkstra over lane segments
        tie = itertools.count()
        start_cost = len(start_seq) * self.step
        best = {}
        prev = {}
        heap = []
        for n in self._successors(sk, last_wp=start_seq[-1]):
            c = start_cost + self._length(n)
            if c < best.get(n, math.inf):
                best[n] = c
                prev[n] = None  # None = came straight from the start lane
                heapq.heappush(heap, (c, next(tie), n))
        found = False
        done = set()
        while heap:
            c, _, k = heapq.heappop(heap)
            if k in done:
                continue
            done.add(k)
            if k == gk:
                found = True
                break
            for n in self._successors(k):
                nc = c + self._length(n)
                if nc < best.get(n, math.inf):
                    best[n] = nc
                    prev[n] = k
                    heapq.heappush(heap, (nc, next(tie), n))
        if not found:
            raise RuntimeError("no route found (goal unreachable from here)")

        keys = []
        k = gk
        while k is not None:
            keys.append(k)
            k = prev[k]
        keys.reverse()

        wps = list(start_seq)
        for k in keys[:-1]:
            wps.extend(self._dense_lane(k))
        last = self._dense_lane(gk)
        d = [w.transform.location.distance(g_loc) for w in last]
        wps.extend(last[: int(np.argmin(d)) + 1])
        return self._to_array(wps)

    @staticmethod
    def _to_array(wps):
        pts = []
        for w in wps:
            t = w.transform
            p = (t.location.x, t.location.y, t.rotation.yaw)
            if pts and math.hypot(p[0] - pts[-1][0], p[1] - pts[-1][1]) < 0.2:
                continue  # drop duplicates where lanes join
            pts.append(p)
        return np.array(pts, dtype=np.float64)
