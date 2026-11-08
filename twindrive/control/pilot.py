# =============================================================================
# File        : twindrive/control/pilot.py
# Project     : TwinDrive — Lane-Free Autonomous Driving on a Digital Twin
#               of an Indian Campus (SRM Town / cm2, CARLA 0.9.16)
# Phase       : 1 — Autonomous route following (step 2)
# =============================================================================
"""
Description
-----------
TwinDrive's classical pilot: follows a planned route, keeps a safe gap to
the vehicle ahead and stops at the goal. It is the RULE-BASED BASELINE that
the learned (IL + RL) policy will be compared against later, and it can also
generate demonstrations for imitation learning.

No CARLA import: inputs are numbers, outputs are throttle/brake/steer.

Lateral control: pure pursuit
    Pick a target point on the route a lookahead distance Ld ahead
    (Ld grows with speed). alpha = angle from the car's heading to that
    point. The circular arc through it needs a front-wheel angle
        delta = atan(2 * L * sin(alpha) / Ld)        (L = wheelbase)
    and CARLA's steer command is delta / max_steer_angle, in [-1, 1].
    (CARLA: positive steer = turn right, and "right" is +y in the car frame.)

Longitudinal control: target speed + PI controller
    v_target = min(v_cruise,                      campus cruise speed
                   sqrt(a_lat_max / curvature),   slow down for bends
                   v_lead + k * (gap - gap_wanted),  adaptive cruise
                   sqrt(2 * a_dec * dist_to_goal))   stop at the goal
    throttle / brake from a PI loop on (v_target - v).
    Emergency brake if time-to-collision < 1.5 s or gap < 4 m.
"""

import math

import numpy as np


class Pilot:
    def __init__(self, route, wheelbase=2.9, max_steer_deg=70.0, cruise_kmh=25.0, dt=0.05):
        self.set_route(route)
        self.L = wheelbase
        self.max_steer = math.radians(max_steer_deg)
        self.v_cruise = cruise_kmh / 3.6
        self.dt = dt
        self.a_lat_max = 2.0   # m/s^2, comfortable cornering
        self.a_dec = 1.5       # m/s^2, comfortable stopping
        self.kp, self.ki, self.kb = 0.35, 0.05, 0.6
        self.integ = 0.0

    def set_route(self, route):
        self.route = np.asarray(route, dtype=np.float64)
        seg = np.hypot(np.diff(self.route[:, 0]), np.diff(self.route[:, 1]))
        self.s = np.concatenate([[0.0], np.cumsum(seg)])  # arc length along route
        self.idx = 0
        self.integ = 0.0

    # ------------------------------------------------------------------ helpers
    def _nearest(self, x, y):
        """Closest route point, searching forward from the last one."""
        lo, hi = max(self.idx - 5, 0), min(self.idx + 120, len(self.route))
        d = np.hypot(self.route[lo:hi, 0] - x, self.route[lo:hi, 1] - y)
        j = int(np.argmin(d))
        self.idx = lo + j
        return float(d[j])

    def _index_ahead(self, dist):
        target = self.s[self.idx] + dist
        k = int(np.searchsorted(self.s, target))
        return min(k, len(self.route) - 1)

    def _curvature_ahead(self, horizon=20.0):
        """Largest heading change per metre over the next `horizon` metres."""
        k_end = self._index_ahead(horizon)
        if k_end - self.idx < 3:
            return 0.0
        yaw = np.unwrap(np.radians(self.route[self.idx:k_end + 1, 2]))
        ds = np.diff(self.s[self.idx:k_end + 1])
        ds[ds < 1e-3] = 1e-3
        return float(np.max(np.abs(np.diff(yaw) / ds)))

    # ------------------------------------------------------------------ step
    def step(self, x, y, yaw_deg, v, lead=None):
        """One control step. v in m/s; lead = {gap_m, speed_kmh, ttc_s} or None."""
        off_route = self._nearest(x, y)
        remaining = float(self.s[-1] - self.s[self.idx])

        # ---- lateral: pure pursuit
        ld = float(np.clip(3.0 + 0.35 * v, 4.0, 12.0))
        tx, ty = self.route[self._index_ahead(ld), :2]
        yaw = math.radians(yaw_deg)
        dx, dy = tx - x, ty - y
        fwd = dx * math.cos(yaw) + dy * math.sin(yaw)
        right = -dx * math.sin(yaw) + dy * math.cos(yaw)
        alpha = math.atan2(right, fwd)
        delta = math.atan2(2.0 * self.L * math.sin(alpha), ld)
        steer = float(np.clip(delta / self.max_steer, -1.0, 1.0))

        # ---- longitudinal: target speed
        kappa = self._curvature_ahead()
        v_curve = math.sqrt(self.a_lat_max / kappa) if kappa > 1e-3 else math.inf
        v_goal = math.sqrt(2.0 * self.a_dec * max(remaining - 2.0, 0.0))
        v_acc, emergency, reason = math.inf, False, "cruise"
        if lead:
            gap, v_lead = lead["gap_m"], lead["speed_kmh"] / 3.6
            gap_wanted = 5.0 + 1.6 * v
            v_acc = max(0.0, v_lead + 0.4 * (gap - gap_wanted))
            ttc = lead.get("ttc_s")
            emergency = gap < 4.0 or (ttc is not None and ttc < 1.5)
        options = {"cruise": self.v_cruise, "curve": v_curve, "following": v_acc, "goal": v_goal}
        reason = min(options, key=options.get)
        v_t = options[reason]

        # ---- PI speed loop
        err = v_t - v
        if emergency:
            throttle, brake, reason = 0.0, 1.0, "EMERGENCY BRAKE"
            self.integ = 0.0
        else:
            self.integ = float(np.clip(self.integ + err * self.dt, -3.0, 3.0))
            u = self.kp * err + self.ki * self.integ
            if u >= 0.0:
                throttle, brake = float(np.clip(u, 0.0, 0.75)), 0.0
            else:
                throttle, brake = 0.0, float(np.clip(-self.kb * u, 0.0, 1.0))
                self.integ = max(self.integ, 0.0)  # no wind-up while braking
            if v_t < 0.1 and v < 0.5:
                throttle, brake = 0.0, 1.0

        arrived = remaining < 3.0 and v < 0.5
        return {"throttle": throttle, "brake": brake, "steer": steer,
                "target_kmh": v_t * 3.6, "remaining_m": remaining,
                "off_route_m": off_route, "reason": reason, "arrived": arrived,
                "lookahead": (float(tx), float(ty))}
