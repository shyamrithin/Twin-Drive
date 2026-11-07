# =============================================================================
# File        : twindrive/sensors/suite.py
# Project     : TwinDrive — Lane-Free Autonomous Driving on a Digital Twin
#               of an Indian Campus (SRM Town / cm2, CARLA 0.9.16)
# Phase       : 1 — Ego vehicle and dashboard
# =============================================================================
"""
Description
-----------
Attaches TwinDrive's sensor suite to the ego vehicle and returns
FRAME-SYNCHRONISED data every tick.

Sensors
    front camera   RGB 800x450, FOV 90, mounted behind the windscreen
    3D LiDAR       32 channels, 60 m range, +10/-25 deg vertical FOV, one full
                   360 deg sweep per tick (rotation_frequency = 1/dt)
    IMU            accelerometer, gyroscope, compass
    collision      event sensor (only fires on impact)

Why synchronous mode matters
    In synchronous mode the simulator only advances when we call
    world.tick(). Each sensor callback pushes (frame, data) into its own
    queue; get(frame) pops until every sensor has data for exactly that
    frame. So camera, LiDAR and IMU always describe the same instant,
    which is essential later when we record training data.

Output of get(frame)
    {"camera": uint8 (H, W, 3) RGB,
     "lidar":  float32 (N, 4) x fwd, y right, z up (VEHICLE frame, z ~ height
               above ground), intensity,
     "imu":    dict ax, ay, az (m/s^2), gx, gy, gz (rad/s), compass (rad)}
"""

import queue

import carla
import numpy as np

CAMERA_CFG = {"x": 1.5, "y": 0.0, "z": 1.6, "fov": 90.0, "w": 800, "h": 450}
LIDAR_CFG = {"x": 0.0, "y": 0.0, "z": 2.4, "channels": 32, "range": 60.0,
             "upper_fov": 10.0, "lower_fov": -25.0, "pps": 300000}

# CARLA LiDAR points come in the sensor's UE frame (x fwd, y right, z up).
# If the 3D view ever looks mirrored left/right against the camera, set True.
LIDAR_FLIP_Y = False


class SensorSuite:
    def __init__(self, world, vehicle, dt, lidar_pps=None, camera=True):
        self.world = world
        self.vehicle = vehicle
        self.actors = []
        self.queues = {}
        self.last_collision = None  # (sim_time, other_actor_type)
        bp_lib = world.get_blueprint_library()

        if camera:
            bp = bp_lib.find("sensor.camera.rgb")
            bp.set_attribute("image_size_x", str(CAMERA_CFG["w"]))
            bp.set_attribute("image_size_y", str(CAMERA_CFG["h"]))
            bp.set_attribute("fov", str(CAMERA_CFG["fov"]))
            bp.set_attribute("sensor_tick", "0.0")  # every tick
            self._spawn("camera", bp, CAMERA_CFG)

        bp = bp_lib.find("sensor.lidar.ray_cast")
        bp.set_attribute("channels", str(LIDAR_CFG["channels"]))
        bp.set_attribute("range", str(LIDAR_CFG["range"]))
        bp.set_attribute("upper_fov", str(LIDAR_CFG["upper_fov"]))
        bp.set_attribute("lower_fov", str(LIDAR_CFG["lower_fov"]))
        bp.set_attribute("points_per_second", str(lidar_pps or LIDAR_CFG["pps"]))
        bp.set_attribute("rotation_frequency", str(1.0 / dt))
        bp.set_attribute("sensor_tick", "0.0")
        self._spawn("lidar", bp, LIDAR_CFG)

        bp = bp_lib.find("sensor.other.imu")
        bp.set_attribute("sensor_tick", "0.0")
        self._spawn("imu", bp, {"x": 0.0, "y": 0.0, "z": 0.0})

        # collision is event-based, so it is NOT part of the per-frame sync
        bp = bp_lib.find("sensor.other.collision")
        col = world.spawn_actor(bp, carla.Transform(), attach_to=vehicle)
        col.listen(self._on_collision)
        self.actors.append(col)

    # ------------------------------------------------------------------ setup
    def _spawn(self, name, bp, cfg):
        tf = carla.Transform(carla.Location(x=cfg["x"], y=cfg["y"], z=cfg["z"]))
        actor = self.world.spawn_actor(bp, tf, attach_to=self.vehicle)
        q = queue.Queue()
        actor.listen(lambda data, q=q: q.put((data.frame, data)))
        self.actors.append(actor)
        self.queues[name] = q

    def _on_collision(self, event):
        self.last_collision = (event.timestamp, event.other_actor.type_id)

    # ------------------------------------------------------------------ data
    def get(self, frame, timeout=2.0):
        """Data from every synced sensor for exactly this frame."""
        raw = {}
        for name, q in self.queues.items():
            while True:
                f, data = q.get(timeout=timeout)
                if f == frame:
                    raw[name] = data
                    break
                if f > frame:  # should not happen in sync mode; keep newest
                    raw[name] = data
                    break
        return self._convert(raw)

    @staticmethod
    def _convert(raw):
        out = {}
        if "camera" in raw:
            img = raw["camera"]
            arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)
            out["camera"] = arr[:, :, 2::-1].copy()  # BGRA -> RGB
        if "lidar" in raw:
            pts = np.frombuffer(raw["lidar"].raw_data, dtype=np.float32).reshape(-1, 4).copy()
            if LIDAR_FLIP_Y:
                pts[:, 1] *= -1.0
            pts[:, 0] += LIDAR_CFG["x"]  # sensor frame -> vehicle frame
            pts[:, 2] += LIDAR_CFG["z"]
            out["lidar"] = pts
        if "imu" in raw:
            m = raw["imu"]
            out["imu"] = {"ax": m.accelerometer.x, "ay": m.accelerometer.y,
                          "az": m.accelerometer.z, "gx": m.gyroscope.x,
                          "gy": m.gyroscope.y, "gz": m.gyroscope.z,
                          "compass": m.compass}
        return out

    # ------------------------------------------------------------------ teardown
    def destroy(self):
        for a in self.actors:
            try:
                if a.is_listening:
                    a.stop()
                a.destroy()
            except RuntimeError:
                pass
        self.actors.clear()
