# =============================================================================
# File        : twindrive/viz/dashboard.py
# Project     : TwinDrive — Lane-Free Autonomous Driving on a Digital Twin
#               of an Indian Campus (SRM Town / cm2, CARLA 0.9.16)
# Phase       : 1 — Ego vehicle and dashboard
# =============================================================================
"""
Description
-----------
A driver/engineering dashboard in the style of Tesla's and Apollo's
visualisers, drawn with pygame. It is deliberately CARLA-free: it only
receives a `state` dict of numbers and numpy arrays, so it can be tested
without a simulator and later fed from recorded logs.

Layout (1600 x 930)
    +--------------------------+-------------+-----------+
    | FRONT CAMERA + GT boxes  | BEV MAP     | TELEMETRY |
    |                          | (heading-up)| speed,    |
    |                          | lidar, cars | steering, |
    |                          | landmarks   | pedals    |
    +--------------------------+------+------+-----------+
    | 3D LIDAR chase view      | LEAD | ORIENTATION      |
    | (height-coloured points, | card | compass, horizon |
    |  vehicle boxes)          +------+------------------+
    |                          | G-G  | HISTORY charts   |
    +--------------------------+------+------------------+
    | STATUS BAR                                         |

State dict (all keys optional except "ego")
    camera   uint8 (H, W, 3) RGB, same size as the camera panel
    lidar    float32 (N, 4): x fwd, y right, z up (vehicle frame), intensity
    bev      uint8 (BEV_SIZE, BEV_SIZE, 3) local drivable map
    ego      speed_kmh, x, y, yaw, pitch, roll, length, width, height
    control  throttle, brake, steer, hand_brake, reverse, gear,
             road_wheel_deg (real front-wheel angle), max_steer_deg
    imu      ax, ay, az (m/s^2, vehicle frame), gz (rad/s), compass (rad)
    vehicles list of {"corners": (8, 3) vehicle frame, "dist": m, "lead": bool}
             corner index = sx*4 + sy*2 + sz, s* in {0: -extent, 1: +extent}
    lead     None or {gap_m, rel_kmh, ttc_s, speed_kmh, type}
    landmarks list of (label, fwd, right) in vehicle frame
    status   mode, sim_time, frame, rtf, client_fps, map, collision, n_near
    camera_cfg  {"x", "z", "fov"} where the camera sits on the car
"""

import math
from collections import deque

import numpy as np
import pygame

WINDOW_SIZE = (1600, 930)
BEV_SIZE = 450
BEV_M_PER_PX = 0.25
BEV_CAR_ROW_FRAC = 0.68

R_CAM = pygame.Rect(0, 0, 800, 450)
R_3D = pygame.Rect(0, 450, 800, 450)
R_BEV = pygame.Rect(800, 0, BEV_SIZE, BEV_SIZE)
R_TELE = pygame.Rect(1250, 0, 350, 450)
R_LEAD = pygame.Rect(800, 450, 400, 210)
R_ORIENT = pygame.Rect(1200, 450, 400, 210)
R_GG = pygame.Rect(800, 660, 240, 240)
R_HIST = pygame.Rect(1040, 660, 560, 240)
R_STATUS = pygame.Rect(0, 900, 1600, 30)

C = {
    "bg": (10, 12, 15), "panel": (20, 23, 28), "edge": (44, 50, 60),
    "text": (220, 226, 232), "dim": (120, 130, 142), "faint": (60, 66, 76),
    "cyan": (0, 200, 255), "green": (60, 220, 120), "amber": (255, 180, 0),
    "red": (255, 70, 70), "magenta": (230, 90, 230), "white": (245, 245, 245),
    "ego": (40, 150, 255),
}

# 8-corner box convention: index = sx*4 + sy*2 + sz
BOX_EDGES = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7),
             (0, 4), (1, 5), (2, 6), (3, 7)]
BOX_FACES = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6),
             (0, 2, 6, 4), (1, 3, 7, 5)]
BOX_BOTTOM = (0, 2, 6, 4)


# ---------------------------------------------------------------------------- helpers
def height_colours(z, zmin=-0.2, zmax=3.5):
    """Height -> RGB. Ground is a dim slate so obstacles stand out."""
    t = np.clip((z - zmin) / (zmax - zmin), 0.0, 1.0)
    stops_t = [0.0, 0.12, 0.3, 0.55, 0.8, 1.0]
    r = np.interp(t, stops_t, [40, 50, 0, 80, 255, 255])
    g = np.interp(t, stops_t, [50, 70, 200, 230, 200, 80])
    b = np.interp(t, stops_t, [70, 95, 255, 120, 0, 60])
    return np.stack([r, g, b], axis=1).astype(np.uint8)


def project(pts, cam, pitch_deg, fov_deg, w, h):
    """Pinhole projection of vehicle-frame points from a virtual camera.

    cam: (x, y, z) camera position in the vehicle frame, looking forward and
    pitched down by pitch_deg. Returns pixel u, v and depth along the view axis.
    """
    th = math.radians(pitch_deg)
    # float64 on purpose: np.float64 is a subclass of Python float, which
    # pygame.draw accepts as a coordinate (np.float32 is rejected)
    d = np.asarray(pts, dtype=np.float64) - np.asarray(cam, dtype=np.float64)
    xc = d[:, 0] * math.cos(th) - d[:, 2] * math.sin(th)
    yc = d[:, 1]
    zc = d[:, 0] * math.sin(th) + d[:, 2] * math.cos(th)
    f = (w / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    safe = np.where(xc > 0.1, xc, 0.1)
    return w / 2.0 + f * yc / safe, h / 2.0 - f * zc / safe, xc


def ego_box(length, width, height):
    sx = np.array([-1, 1]) * length / 2.0
    sy = np.array([-1, 1]) * width / 2.0
    sz = np.array([0.0, height])
    return np.array([[x, y, z] for x in sx for y in sy for z in sz], dtype=np.float32)


def ttc_colour(ttc):
    if ttc is None:
        return C["green"]
    return C["red"] if ttc < 2.0 else C["amber"] if ttc < 4.0 else C["green"]


def heading_name(deg):
    names = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
             "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return names[int((deg % 360) / 22.5 + 0.5) % 16]


# ---------------------------------------------------------------------------- dashboard
class Dashboard:
    def __init__(self, screen, history_len=300):
        self.screen = screen
        pygame.font.init()
        face = "dejavusans,liberationsans,arial"
        self.f12 = pygame.font.SysFont(face, 12)
        self.f14 = pygame.font.SysFont(face, 14)
        self.f16b = pygame.font.SysFont(face, 16, bold=True)
        self.f20b = pygame.font.SysFont(face, 20, bold=True)
        self.f28b = pygame.font.SysFont(face, 28, bold=True)
        self.f44b = pygame.font.SysFont(face, 44, bold=True)
        self.hist = {k: deque(maxlen=history_len)
                     for k in ("speed", "steer", "throttle", "brake")}
        self.gg_trail = deque(maxlen=60)
        self.show_help = False

    # ------------------------------------------------------------------ public
    def render(self, s):
        self._record(s)
        self.screen.fill(C["bg"])
        self._camera(s)
        self._view3d(s)
        self._bev(s)
        self._telemetry(s)
        self._lead(s)
        self._orientation(s)
        self._gg(s)
        self._history()
        self._status(s)
        if self.show_help:
            self._help()

    # ------------------------------------------------------------------ utils
    def _text(self, txt, font, colour, pos, anchor="topleft"):
        surf = font.render(str(txt), True, colour)
        rect = surf.get_rect(**{anchor: pos})
        self.screen.blit(surf, rect)
        return rect

    def _panel(self, rect, title):
        pygame.draw.rect(self.screen, C["panel"], rect)
        pygame.draw.rect(self.screen, C["edge"], rect, 1)
        self._text(title, self.f12, C["dim"], (rect.x + 8, rect.y + 6))

    def _record(self, s):
        ego, ctl = s["ego"], s.get("control", {})
        self.hist["speed"].append(ego.get("speed_kmh", 0.0))
        self.hist["steer"].append(ctl.get("steer", 0.0))
        self.hist["throttle"].append(ctl.get("throttle", 0.0))
        self.hist["brake"].append(ctl.get("brake", 0.0))
        imu = s.get("imu")
        if imu:
            self.gg_trail.append((imu["ay"] / 9.81, imu["ax"] / 9.81))

    # ------------------------------------------------------------------ camera
    def _camera(self, s):
        r = R_CAM
        cam = s.get("camera")
        if cam is None:
            self._panel(r, "FRONT CAMERA")
            self._text("camera disabled", self.f16b, C["dim"], r.center, "center")
            return
        self.screen.blit(pygame.surfarray.make_surface(cam.swapaxes(0, 1)), r.topleft)
        cfg = s.get("camera_cfg", {"x": 1.5, "z": 1.6, "fov": 90.0})
        for v in s.get("vehicles", []):
            u, vv, depth = project(v["corners"], (cfg["x"], 0.0, cfg["z"]), 0.0,
                                   cfg["fov"], r.w, r.h)
            if np.any(depth < 0.5):
                continue
            x0, x1 = np.clip([u.min(), u.max()], 0, r.w - 1)
            y0, y1 = np.clip([vv.min(), vv.max()], 0, r.h - 1)
            if x1 - x0 < 3 or y1 - y0 < 3:
                continue
            lead = v.get("lead", False)
            col = C["amber"] if lead else C["cyan"]
            box = pygame.Rect(int(x0), int(y0), int(x1 - x0), int(y1 - y0))
            pygame.draw.rect(self.screen, col, box, 3 if lead else 1)
            tag = f"LEAD {v['dist']:.1f} m" if lead else f"{v['dist']:.0f} m"
            self._text(tag, self.f14 if lead else self.f12, col, (box.x, box.y - 16))
        self._text("FRONT CAMERA  ·  ground-truth boxes", self.f12, C["white"], (8, 6))
        coll = s.get("status", {}).get("collision")
        if coll:
            bar = pygame.Rect(0, r.h - 34, r.w, 34)
            pygame.draw.rect(self.screen, C["red"], bar)
            self._text(f"COLLISION  ·  {coll}", self.f20b, C["white"], bar.center, "center")

    # ------------------------------------------------------------------ 3D view
    def _view3d(self, s):
        r = R_3D
        cam, pitch, fov = (-15.0, 0.0, 7.0), 20.0, 75.0  # chase camera behind/above
        surf = pygame.Surface(r.size)
        surf.fill((8, 10, 14))

        # ground grid every 10 m
        for gx in range(-10, 71, 10):
            self._grid_line(surf, (gx, -40, 0), (gx, 40, 0), cam, pitch, fov)
        for gy in range(-40, 41, 10):
            self._grid_line(surf, (-10, gy, 0), (70, gy, 0), cam, pitch, fov)

        lidar = s.get("lidar")
        n_pts = 0
        if lidar is not None and len(lidar):
            u, v, depth = project(lidar[:, :3], cam, pitch, fov, r.w, r.h)
            ok = (depth > 1.0) & (u >= 0) & (u < r.w - 1) & (v >= 0) & (v < r.h - 1)
            ui, vi = u[ok].astype(np.int32), v[ok].astype(np.int32)
            cols = height_colours(lidar[ok, 2])
            px = pygame.surfarray.pixels3d(surf)
            px[ui, vi] = cols
            px[ui + 1, vi] = cols
            del px
            n_pts = len(lidar)

        # other vehicles: wireframes
        for veh in s.get("vehicles", []):
            lead = veh.get("lead", False)
            col = C["amber"] if lead else C["cyan"]
            u, v, depth = project(veh["corners"], cam, pitch, fov, r.w, r.h)
            for a, b in BOX_EDGES:
                if depth[a] > 1.0 and depth[b] > 1.0:
                    pygame.draw.line(surf, col, (u[a], v[a]), (u[b], v[b]), 2 if lead else 1)
            if lead and np.all(depth > 1.0):
                lbl = self.f14.render(f"{veh['dist']:.1f} m", True, col)
                surf.blit(lbl, (u.mean() - lbl.get_width() / 2, v.min() - 18))

        # ego car: shaded faces, far faces first (painter's algorithm)
        ego = s["ego"]
        box = ego_box(ego.get("length", 4.7), ego.get("width", 2.0), ego.get("height", 1.5))
        u, v, depth = project(box, cam, pitch, fov, r.w, r.h)
        faces = sorted(BOX_FACES, key=lambda f: -depth[list(f)].mean())
        for i, f in enumerate(faces):
            shade = 0.45 + 0.55 * i / (len(faces) - 1)
            col = tuple(int(c * shade) for c in C["ego"])
            pygame.draw.polygon(surf, col, [(u[k], v[k]) for k in f])
            pygame.draw.polygon(surf, C["white"], [(u[k], v[k]) for k in f], 1)

        self.screen.blit(surf, r.topleft)
        pygame.draw.rect(self.screen, C["edge"], r, 1)
        self._text(f"3D LIDAR  ·  {n_pts:,} points  ·  coloured by height",
                   self.f12, C["dim"], (r.x + 8, r.y + 6))
        # height legend
        for i in range(120):
            z = -0.2 + 3.7 * i / 119
            pygame.draw.line(self.screen, tuple(int(c) for c in height_colours(np.array([z]))[0]),
                             (r.right - 140 + i, r.y + 10), (r.right - 140 + i, r.y + 18))
        self._text("0 m", self.f12, C["dim"], (r.right - 140, r.y + 21))
        self._text("3.5 m", self.f12, C["dim"], (r.right - 20, r.y + 21), "topright")

    def _grid_line(self, surf, p0, p1, cam, pitch, fov, steps=16):
        pts = np.linspace(p0, p1, steps)
        u, v, depth = project(pts, cam, pitch, fov, surf.get_width(), surf.get_height())
        for i in range(steps - 1):
            if depth[i] > 1.0 and depth[i + 1] > 1.0:
                pygame.draw.line(surf, C["faint"], (u[i], v[i]), (u[i + 1], v[i + 1]), 1)

    # ------------------------------------------------------------------ BEV
    def _bev(self, s):
        r = R_BEV
        S, m = BEV_SIZE, BEV_M_PER_PX
        car_r, car_c = int(S * BEV_CAR_ROW_FRAC), S // 2
        bev = s.get("bev")
        img = bev.copy() if bev is not None else np.full((S, S, 3), 16, np.uint8)

        lidar = s.get("lidar")
        if lidar is not None and len(lidar):
            rows = (car_r - lidar[:, 0] / m).astype(np.int32)
            cols = (car_c + lidar[:, 1] / m).astype(np.int32)
            above = lidar[:, 2] > 0.3  # draw obstacles only; ground is the map
            ok = above & (rows >= 0) & (rows < S) & (cols >= 0) & (cols < S)
            img[rows[ok], cols[ok]] = height_colours(lidar[ok, 2])

        surf = pygame.surfarray.make_surface(img.swapaxes(0, 1))
        for rad_m in (10, 20, 30, 40):
            pygame.draw.circle(surf, C["faint"], (car_c, car_r), int(rad_m / m), 1)
            surf.blit(self.f12.render(f"{rad_m}", True, C["faint"]),
                      (car_c + 3, car_r - rad_m / m - 14))

        def to_px(fwd, right):
            return float(car_c + right / m), float(car_r - fwd / m)

        for veh in s.get("vehicles", []):
            poly = [to_px(veh["corners"][k, 0], veh["corners"][k, 1]) for k in BOX_BOTTOM]
            lead = veh.get("lead", False)
            pygame.draw.polygon(surf, C["amber"] if lead else C["cyan"], poly, 0 if lead else 2)

        for label, fwd, right in s.get("landmarks", []):
            x, y = to_px(fwd, right)
            if 0 <= x < S and 0 <= y < S:
                pygame.draw.circle(surf, C["red"], (int(x), int(y)), 5)
                surf.blit(self.f12.render(label, True, C["amber"]), (x + 7, y - 7))

        ego = s["ego"]
        L, W = ego.get("length", 4.7), ego.get("width", 2.0)
        car = pygame.Rect(0, 0, W / m, L / m)
        car.center = (car_c, car_r)
        pygame.draw.rect(surf, C["ego"], car)
        pygame.draw.polygon(surf, C["white"], [(car_c, car.top - 6), (car_c - 5, car.top + 2),
                                               (car_c + 5, car.top + 2)])

        # north arrow: world north is -y; express it in the heading-up frame
        yaw = math.radians(ego.get("yaw", 0.0))
        n_fwd, n_right = -math.sin(yaw), -math.cos(yaw)
        nx, ny = S - 30, 40
        pygame.draw.line(surf, C["red"], (nx, ny), (nx + 18 * n_right, ny - 18 * n_fwd), 3)
        surf.blit(self.f14.render("N", True, C["red"]), (nx + 24 * n_right - 5, ny - 24 * n_fwd - 8))

        self.screen.blit(surf, r.topleft)
        pygame.draw.rect(self.screen, C["edge"], r, 1)
        self._text("BEV MAP  ·  heading-up  ·  drivable area + LiDAR obstacles",
                   self.f12, C["dim"], (r.x + 8, r.y + 6))

    # ------------------------------------------------------------------ telemetry
    def _telemetry(self, s):
        r = R_TELE
        self._panel(r, "TELEMETRY")
        ctl, ego = s.get("control", {}), s["ego"]
        mode = s.get("status", {}).get("mode", "MANUAL")
        badge = pygame.Rect(0, 0, 120, 24)
        badge.topright = (r.right - 10, r.y + 6)
        pygame.draw.rect(self.screen, C["green"] if mode == "AUTOPILOT" else C["cyan"],
                         badge, border_radius=12)
        self._text(mode, self.f12, C["bg"], badge.center, "center")

        # speed gauge: 270 deg arc, 0-60 km/h
        cx, cy, rad, vmax = r.centerx, r.y + 150, 100, 60.0
        speed = ego.get("speed_kmh", 0.0)
        box = pygame.Rect(0, 0, 2 * rad, 2 * rad)
        box.center = (cx, cy)
        for lo, hi, col in ((0, 30, C["green"]), (30, 40, C["amber"]), (40, 60, C["red"])):
            a0 = math.radians(225 - hi / vmax * 270)
            a1 = math.radians(225 - lo / vmax * 270)
            pygame.draw.arc(self.screen, col, box, a0, a1, 4)
        for kmh in range(0, 61, 10):
            a = math.radians(225 - kmh / vmax * 270)
            p0 = (cx + (rad - 10) * math.cos(a), cy - (rad - 10) * math.sin(a))
            p1 = (cx + rad * math.cos(a), cy - rad * math.sin(a))
            pygame.draw.line(self.screen, C["text"], p0, p1, 2)
            self._text(kmh, self.f12, C["dim"],
                       (cx + (rad - 22) * math.cos(a), cy - (rad - 22) * math.sin(a)), "center")
        a = math.radians(225 - min(speed, vmax) / vmax * 270)
        pygame.draw.line(self.screen, C["white"], (cx, cy),
                         (cx + (rad - 14) * math.cos(a), cy - (rad - 14) * math.sin(a)), 3)
        pygame.draw.circle(self.screen, C["white"], (cx, cy), 6)
        self._text(f"{speed:4.1f}", self.f44b, C["white"], (cx, cy + 38), "center")
        self._text("km/h", self.f12, C["dim"], (cx, cy + 64), "center")
        gear = "R" if ctl.get("reverse") else f"D{max(ctl.get('gear', 1), 1)}"
        self._text(gear, self.f20b, C["amber"] if gear == "R" else C["text"],
                   (cx, cy - 40), "center")

        # steering wheel: drawn at the commanded angle (900 deg lock-to-lock)
        steer = ctl.get("steer", 0.0)
        wcx, wcy, wr = cx, r.y + 340, 52
        ang = -steer * math.radians(450)  # screen rotation, clockwise for right
        pygame.draw.circle(self.screen, C["text"], (wcx, wcy), wr, 6)
        pygame.draw.circle(self.screen, C["text"], (wcx, wcy), 10)
        for spoke in (math.pi / 2 + math.pi, math.pi / 2 + math.pi + 2.1, math.pi / 2 + math.pi - 2.1):
            a2 = spoke + ang
            pygame.draw.line(self.screen, C["text"], (wcx, wcy),
                             (wcx + wr * math.cos(a2), wcy - wr * math.sin(a2)), 5)
        top = math.pi / 2 + ang
        pygame.draw.circle(self.screen, C["red"],
                           (int(wcx + wr * math.cos(top)), int(wcy - wr * math.sin(top))), 6)
        road = ctl.get("road_wheel_deg")
        self._text(f"steer {steer:+.2f}   wheel {steer * 450:+.0f}°", self.f12, C["text"],
                   (cx, wcy + wr + 14), "center")
        if road is not None:
            self._text(f"road-wheel angle {road:+.1f}°", self.f12, C["dim"],
                       (cx, wcy + wr + 30), "center")

        # pedal bars
        for x, key, col, lbl in ((r.x + 16, "throttle", C["green"], "THR"),
                                 (r.right - 34, "brake", C["red"], "BRK")):
            bar = pygame.Rect(x, r.y + 260, 18, 150)
            pygame.draw.rect(self.screen, C["faint"], bar, 1)
            val = float(np.clip(ctl.get(key, 0.0), 0, 1))
            fill = pygame.Rect(bar.x + 1, bar.bottom - val * bar.h, bar.w - 2, val * bar.h)
            pygame.draw.rect(self.screen, col, fill)
            self._text(lbl, self.f12, C["dim"], (bar.centerx, bar.bottom + 4), "midtop")
            self._text(f"{val * 100:.0f}", self.f12, C["text"], (bar.centerx, bar.y - 4), "midbottom")
        if ctl.get("hand_brake"):
            self._text("(P) HANDBRAKE", self.f14, C["red"], (cx, r.y + 30), "center")

    # ------------------------------------------------------------------ lead card
    def _lead(self, s):
        r = R_LEAD
        self._panel(r, "LEAD VEHICLE  ·  ground truth (perception arrives in Phase 3)")
        lead = s.get("lead")
        n_near = s.get("status", {}).get("n_near", 0)
        self._text(f"vehicles within 60 m: {n_near}", self.f12, C["dim"], (r.x + 10, r.bottom - 20))
        if not lead:
            self._text("No vehicle ahead in path", self.f20b, C["green"], (r.centerx, r.y + 80), "center")
            self._text("road clear for 60 m", self.f14, C["dim"], (r.centerx, r.y + 108), "center")
            return
        ttc = lead.get("ttc_s")
        col = ttc_colour(ttc)
        self._text(f"{lead['gap_m']:.1f} m", self.f44b, col, (r.x + 12, r.y + 28))
        self._text("gap", self.f12, C["dim"], (r.x + 14, r.y + 80))
        rel = lead.get("rel_kmh", 0.0)
        trend = "closing" if rel < -0.5 else "opening" if rel > 0.5 else "steady"
        lines = [
            ("TTC", f"{ttc:.1f} s" if ttc is not None else "no closing", col),
            ("relative", f"{rel:+.1f} km/h {trend}", C["text"]),
            ("lead speed", f"{lead.get('speed_kmh', 0.0):.1f} km/h", C["text"]),
            ("type", lead.get("type", "?").replace("vehicle.", ""), C["dim"]),
        ]
        for i, (k, val, c) in enumerate(lines):
            self._text(k, self.f12, C["dim"], (r.x + 175, r.y + 30 + i * 26))
            self._text(val, self.f14 if k == "type" else self.f16b, c,
                       (r.x + 245, r.y + 28 + i * 26))
        # distance bar 0-60 m
        bar = pygame.Rect(r.x + 12, r.y + 150, r.w - 24, 10)
        pygame.draw.rect(self.screen, C["faint"], bar, border_radius=5)
        frac = min(lead["gap_m"], 60.0) / 60.0
        pygame.draw.rect(self.screen, col, (bar.x, bar.y, bar.w * frac, bar.h), border_radius=5)
        for d in (0, 20, 40, 60):
            self._text(f"{d}", self.f12, C["dim"], (bar.x + bar.w * d / 60, bar.bottom + 2), "midtop")

    # ------------------------------------------------------------------ orientation
    def _orientation(self, s):
        r = R_ORIENT
        self._panel(r, "ORIENTATION")
        ego, imu = s["ego"], s.get("imu")
        if imu and "compass" in imu:
            heading = math.degrees(imu["compass"]) % 360
        else:  # CARLA: yaw 0 points east (+x); north is -y
            heading = (ego.get("yaw", 0.0) + 90.0) % 360
        cx, cy, rad = r.x + 85, r.y + 115, 70
        pygame.draw.circle(self.screen, C["faint"], (cx, cy), rad, 2)
        for b in range(0, 360, 30):
            a = math.radians(b - heading)
            p0 = (cx + (rad - 8) * math.sin(a), cy - (rad - 8) * math.cos(a))
            p1 = (cx + rad * math.sin(a), cy - rad * math.cos(a))
            pygame.draw.line(self.screen, C["dim"], p0, p1, 2)
        for b, name in ((0, "N"), (90, "E"), (180, "S"), (270, "W")):
            a = math.radians(b - heading)
            self._text(name, self.f16b, C["red"] if name == "N" else C["text"],
                       (cx + (rad - 22) * math.sin(a), cy - (rad - 22) * math.cos(a)), "center")
        pygame.draw.polygon(self.screen, C["ego"], [(cx, cy - 26), (cx - 9, cy + 12), (cx + 9, cy + 12)])
        self._text(f"{heading:05.1f}° {heading_name(heading)}", self.f16b, C["white"],
                   (cx, r.bottom - 14), "center")

        # artificial horizon
        hz = pygame.Rect(r.x + 180, r.y + 30, 110, 80)
        hsurf = pygame.Surface(hz.size)
        hsurf.fill((40, 90, 160))
        pitch, roll = ego.get("pitch", 0.0), ego.get("roll", 0.0)
        off = pitch * 2.0
        ca, sa = math.cos(math.radians(roll)), math.sin(math.radians(roll))
        w2, h2 = hz.w / 2, hz.h / 2 + off
        ground = [(w2 - 200 * ca, h2 + 200 * sa), (w2 + 200 * ca, h2 - 200 * sa),
                  (w2 + 200 * ca + 200 * sa, h2 - 200 * sa + 200 * ca),
                  (w2 - 200 * ca + 200 * sa, h2 + 200 * sa + 200 * ca)]
        pygame.draw.polygon(hsurf, (120, 80, 40), ground)
        pygame.draw.line(hsurf, C["amber"], (hz.w / 2 - 25, hz.h / 2), (hz.w / 2 + 25, hz.h / 2), 3)
        self.screen.blit(hsurf, hz.topleft)
        pygame.draw.rect(self.screen, C["edge"], hz, 1)

        rows = [("pitch", f"{pitch:+.1f}°"), ("roll", f"{roll:+.1f}°"),
                ("yaw (map)", f"{ego.get('yaw', 0.0):+.1f}°"),
                ("yaw rate", f"{math.degrees(imu['gz']):+.1f}°/s" if imu else "—"),
                ("position", f"{ego.get('x', 0):.1f}, {ego.get('y', 0):.1f}")]
        for i, (k, val) in enumerate(rows):
            self._text(k, self.f12, C["dim"], (r.x + 300, r.y + 30 + i * 30))
            self._text(val, self.f14, C["text"], (r.x + 300, r.y + 44 + i * 30))
        self._text("horizon", self.f12, C["dim"], (hz.centerx, hz.bottom + 4), "midtop")
        if imu:
            self._text(f"az {imu['az']:.2f} m/s²", self.f12, C["dim"], (hz.centerx, hz.bottom + 22), "midtop")

    # ------------------------------------------------------------------ g-g diagram
    def _gg(self, s):
        r = R_GG
        self._panel(r, "ACCELERATION  ·  g-g")
        cx, cy, full = r.centerx, r.centery + 6, 0.6  # radius = 0.6 g
        rad = 90
        for g in (0.2, 0.4, 0.6):
            pygame.draw.circle(self.screen, C["faint"], (cx, cy), int(rad * g / full), 1)
            self._text(f"{g}", self.f12, C["faint"], (cx + 3, cy - rad * g / full - 14))
        pygame.draw.line(self.screen, C["faint"], (cx - rad, cy), (cx + rad, cy))
        pygame.draw.line(self.screen, C["faint"], (cx, cy - rad), (cx, cy + rad))
        pts = list(self.gg_trail)
        for i, (lat, lon) in enumerate(pts):
            k = (i + 1) / max(len(pts), 1)
            col = (int(255 * k), int(180 * k), 0)
            x = cx + np.clip(lat / full, -1, 1) * rad
            y = cy - np.clip(lon / full, -1, 1) * rad
            pygame.draw.circle(self.screen, col, (int(x), int(y)), 2)
        if pts:
            lat, lon = pts[-1]
            pygame.draw.circle(self.screen, C["white"],
                               (int(cx + np.clip(lat / full, -1, 1) * rad),
                                int(cy - np.clip(lon / full, -1, 1) * rad)), 5)
            self._text(f"long {lon:+.2f} g   lat {lat:+.2f} g", self.f12, C["text"],
                       (r.centerx, r.bottom - 16), "center")
        self._text("accel", self.f12, C["dim"], (cx + 4, r.y + 22))
        self._text("right", self.f12, C["dim"], (r.right - 8, cy + 2), "topright")

    # ------------------------------------------------------------------ history
    def _history(self):
        r = R_HIST
        self._panel(r, "HISTORY  ·  last 15 s")
        top = pygame.Rect(r.x + 40, r.y + 26, r.w - 52, 90)
        bot = pygame.Rect(r.x + 40, r.y + 136, r.w - 52, 90)
        sp = list(self.hist["speed"])
        vmax = max(30.0, max(sp) * 1.1) if sp else 30.0
        self._chart(top, [(sp, C["cyan"])], 0.0, vmax, "km/h", f"{vmax:.0f}")
        steer = list(self.hist["steer"])
        pedal = [t - b for t, b in zip(self.hist["throttle"], self.hist["brake"])]
        self._chart(bot, [(pedal, C["green"]), (steer, C["magenta"])], -1.0, 1.0, "", "+1")
        self._text("steer", self.f12, C["magenta"], (bot.right - 110, bot.y - 14))
        self._text("thr − brk", self.f12, C["green"], (bot.right - 60, bot.y - 14))

    def _chart(self, rect, series, lo, hi, unit, hi_lbl):
        pygame.draw.rect(self.screen, (14, 16, 20), rect)
        pygame.draw.rect(self.screen, C["faint"], rect, 1)
        if lo < 0 < hi:
            y0 = rect.bottom - (0 - lo) / (hi - lo) * rect.h
            pygame.draw.line(self.screen, C["faint"], (rect.x, y0), (rect.right, y0))
        self._text(hi_lbl, self.f12, C["dim"], (rect.x - 4, rect.y), "topright")
        self._text(f"{lo:.0f}", self.f12, C["dim"], (rect.x - 4, rect.bottom), "bottomright")
        if unit:
            self._text(unit, self.f12, C["dim"], (rect.x + 4, rect.y + 2))
        for data, col in series:
            if len(data) < 2:
                continue
            n = self.hist["speed"].maxlen
            pts = [(rect.x + rect.w * (i + n - len(data)) / (n - 1),
                    rect.bottom - (np.clip(d, lo, hi) - lo) / (hi - lo) * rect.h)
                   for i, d in enumerate(data)]
            pygame.draw.lines(self.screen, col, False, pts, 2)

    # ------------------------------------------------------------------ status / help
    def _status(self, s):
        r = R_STATUS
        pygame.draw.rect(self.screen, (16, 18, 22), r)
        pygame.draw.line(self.screen, C["edge"], r.topleft, r.topright)
        st = s.get("status", {})
        left = (f"SIM {st.get('sim_time', 0):7.1f} s   |   frame {st.get('frame', 0)}   |   "
                f"real-time factor {st.get('rtf', 0):.2f}x   |   {st.get('client_fps', 0):4.1f} FPS   |   "
                f"map {st.get('map', '?')}")
        self._text(left, self.f12, C["text"], (r.x + 10, r.centery), "midleft")
        self._text("W/S thr/brk   A/D steer   Q reverse   SPACE handbrake   "
                   "P autopilot   H help   ESC quit", self.f12, C["dim"],
                   (r.right - 10, r.centery), "midright")

    def _help(self):
        lines = ["CONTROLS", "", "W / UP        throttle", "S / DOWN      brake",
                 "A / D         steer left / right", "Q             toggle reverse",
                 "SPACE         handbrake", "P             toggle autopilot (Traffic Manager)",
                 "H             toggle this help", "ESC           quit and clean up"]
        box = pygame.Rect(0, 0, 420, 40 + 24 * len(lines))
        box.center = (WINDOW_SIZE[0] // 2, WINDOW_SIZE[1] // 2)
        overlay = pygame.Surface(box.size, pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 215))
        self.screen.blit(overlay, box.topleft)
        pygame.draw.rect(self.screen, C["cyan"], box, 1)
        for i, ln in enumerate(lines):
            self._text(ln, self.f16b if i == 0 else self.f14, C["text"],
                       (box.x + 24, box.y + 20 + 24 * i))
