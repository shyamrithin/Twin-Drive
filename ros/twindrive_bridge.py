#!/usr/bin/env python3
# =============================================================================
# File        : ros/twindrive_bridge.py
# Project     : TwinDrive — Lane-Free Autonomous Driving on a Digital Twin
#               of an Indian Campus (SRM Town / cm2, CARLA 0.9.16)
# Phase       : 1 — ROS 2 integration (SLAM, then Nav2)
# Runs on     : WSL2 Ubuntu 22.04, ROS 2 Humble, SYSTEM python3 (conda OFF)
# =============================================================================
"""
Description
-----------
TwinDrive's own CARLA <-> ROS 2 bridge. It owns the synchronous simulation
loop (like tools/10_drive_dashboard.py) and exposes the car to ROS 2:

Publishes
    /clock                 rosgraph_msgs/Clock      simulation time
    /tf                    odom -> base_link        car pose
    /tf_static             base_link -> lidar, laser, camera, imu
    /odom                  nav_msgs/Odometry        (ground truth, see below)
    /scan                  sensor_msgs/LaserScan    2D scan cut from the 3D LiDAR
    /points                sensor_msgs/PointCloud2  full 3D LiDAR cloud
    /imu                   sensor_msgs/Imu
    /camera/image          sensor_msgs/Image        rgb8
    /twindrive/route       nav_msgs/Path            route the pilot follows

Subscribes
    /goal_pose             geometry_msgs/PoseStamped  RViz "2D Goal Pose":
                           our Dijkstra planner + pilot drive there (--mode pilot)
    /cmd_vel               geometry_msgs/Twist        velocity commands, e.g.
                           from Nav2 (--mode cmd_vel)

Modes
    pilot     TwinDrive pilot drives: to clicked goals, or along --tour
    cmd_vel   an external ROS controller (Nav2) drives via /cmd_vel

Coordinate frames (the important part)
    CARLA is LEFT-handed: x forward, y RIGHT, z up, yaw clockwise.
    ROS (REP 103) is RIGHT-handed: x forward, y LEFT, z up, yaw anticlockwise.
    So everything crossing the bridge gets y -> -y and yaw -> -yaw.
    The odom frame is placed at the CARLA world origin, so map/odom
    coordinates are simply CARLA world coordinates with y flipped.

Odometry
    /odom comes from CARLA's exact pose ("perfect odometry"). Real robots
    integrate noisy wheel/IMU data; --odom-noise adds drift to mimic that.

Why not the official carla-ros-bridge? It does not support CARLA 0.9.16 on
ROS 2 Humble cleanly, and writing our own keeps every frame conversion visible.

Usage (conda deactivated, ROS sourced)
-----
    python3 ros/twindrive_bridge.py --tour all --hide-vegetation        # map campus
    python3 ros/twindrive_bridge.py --hide-vegetation                   # click goals
    python3 ros/twindrive_bridge.py --mode cmd_vel --hide-vegetation    # for Nav2
"""

import argparse
import math
import queue
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root -> twindrive

import carla  # noqa: E402
import numpy as np  # noqa: E402
import rclpy  # noqa: E402
from builtin_interfaces.msg import Time  # noqa: E402
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist  # noqa: E402
from nav_msgs.msg import Odometry, Path as PathMsg  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy  # noqa: E402
from rosgraph_msgs.msg import Clock  # noqa: E402
from sensor_msgs.msg import Image, Imu, LaserScan, PointCloud2, PointField  # noqa: E402
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster  # noqa: E402

from twindrive import DEFAULT_MAP  # noqa: E402
from twindrive.control import Pilot  # noqa: E402
from twindrive.planning import RoutePlanner  # noqa: E402
from twindrive.sensors import CAMERA_CFG, LIDAR_CFG, SensorSuite  # noqa: E402
from twindrive.world import load_landmarks  # noqa: E402
from twindrive.world.vegetation import find_vegetation, set_visible  # noqa: E402

SCAN_BINS = 1080            # 1/3 degree resolution
SCAN_Z = (0.4, 2.5)         # metres above ground kept for the 2D scan
SCAN_MIN_R = 2.6            # ignore returns from our own car body
LASER_Z = 1.0               # height of the virtual 2D laser frame
CMD_TIMEOUT_S = 0.5         # stop if /cmd_vel goes quiet


# ============================================================================ helpers
def stamp(sim_t):
    s = int(sim_t)
    return Time(sec=s, nanosec=int((sim_t - s) * 1e9))


def yaw_quat(yaw_rad):
    return 0.0, 0.0, math.sin(yaw_rad / 2.0), math.cos(yaw_rad / 2.0)


def static_tf(parent, child, x, y, z, t):
    m = TransformStamped()
    m.header.stamp = t
    m.header.frame_id = parent
    m.child_frame_id = child
    m.transform.translation.x, m.transform.translation.y, m.transform.translation.z = x, y, z
    m.transform.rotation.w = 1.0
    return m


def find_lead(ego, others):
    """Nearest vehicle ahead overlapping the ego's width (ground truth)."""
    tf = ego.get_transform()
    inv = np.array(tf.get_inverse_matrix())
    fwd = tf.get_forward_vector()
    v = ego.get_velocity()
    v_ego = v.x * fwd.x + v.y * fwd.y
    ext = ego.bounding_box.extent
    best = None
    for a in others:
        loc = a.get_location()
        if loc.distance(tf.location) > 60.0:
            continue
        p = inv @ np.array([loc.x, loc.y, loc.z, 1.0])
        aext = a.bounding_box.extent
        if p[0] > 0 and abs(p[1]) < ext.y + aext.y + 0.3:
            gap = max(p[0] - ext.x - aext.x, 0.0)
            if best is None or gap < best["gap_m"]:
                av = a.get_velocity()
                closing = v_ego - (av.x * fwd.x + av.y * fwd.y)
                best = {"gap_m": gap, "speed_kmh": math.hypot(av.x, av.y) * 3.6,
                        "ttc_s": gap / closing if closing > 0.3 else None}
    return best


# ============================================================================ node
class Bridge(Node):
    def __init__(self, args):
        super().__init__("twindrive_bridge")
        self.args = args
        rel = QoSProfile(depth=10)
        best_effort = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_clock = self.create_publisher(Clock, "/clock", rel)
        self.pub_odom = self.create_publisher(Odometry, "/odom", rel)
        self.pub_scan = self.create_publisher(LaserScan, "/scan", rel)
        self.pub_pts = self.create_publisher(PointCloud2, "/points", best_effort)
        self.pub_imu = self.create_publisher(Imu, "/imu", rel)
        self.pub_img = self.create_publisher(Image, "/camera/image", best_effort)
        self.pub_route = self.create_publisher(PathMsg, "/twindrive/route", latched)
        self.tf_dyn = TransformBroadcaster(self)
        self.tf_static = StaticTransformBroadcaster(self)
        self.create_subscription(PoseStamped, "/goal_pose", self.on_goal, rel)
        self.create_subscription(Twist, "/cmd_vel", self.on_cmd_vel, rel)
        self.pending_goal = None
        self.cmd = None          # (linear_x, angular_z, sim_time_received)
        self.sim_t = 0.0
        self.odom_err = np.zeros(3)  # x, y, yaw drift for --odom-noise

    # ------------------------------------------------------------------ callbacks
    def on_goal(self, msg):
        # ROS map/odom frame -> CARLA world: flip y
        self.pending_goal = (msg.pose.position.x, -msg.pose.position.y)
        self.get_logger().info(f"Goal received: x={msg.pose.position.x:.1f} "
                               f"y={msg.pose.position.y:.1f} (ROS frame)")

    def on_cmd_vel(self, msg):
        self.cmd = (msg.linear.x, msg.angular.z, self.sim_t)

    # ------------------------------------------------------------------ publishing
    def publish_static(self, t):
        self.tf_static.sendTransform([
            static_tf("base_link", "lidar", LIDAR_CFG["x"], 0.0, LIDAR_CFG["z"], t),
            static_tf("base_link", "laser", 0.0, 0.0, LASER_Z, t),
            static_tf("base_link", "camera", CAMERA_CFG["x"], 0.0, CAMERA_CFG["z"], t),
            static_tf("base_link", "imu", 0.0, 0.0, 0.0, t),
        ])

    def publish_state(self, ego, data, t, dt):
        tf = ego.get_transform()
        x, y = tf.location.x, -tf.location.y                 # CARLA -> ROS
        yaw = -math.radians(tf.rotation.yaw)
        v = ego.get_velocity()
        fwd = tf.get_forward_vector()
        v_fwd = v.x * fwd.x + v.y * fwd.y + v.z * fwd.z
        yaw_rate = -math.radians(ego.get_angular_velocity().z)

        if self.args.odom_noise > 0.0:  # random-walk drift, grows with distance
            k = self.args.odom_noise
            self.odom_err += np.random.normal(0.0, 1.0, 3) * np.array([k, k, k * 0.02]) \
                * abs(v_fwd) * dt
            x, y, yaw = x + self.odom_err[0], y + self.odom_err[1], yaw + self.odom_err[2]
        qx, qy, qz, qw = yaw_quat(yaw)

        ts = TransformStamped()
        ts.header.stamp = t
        ts.header.frame_id = "odom"
        ts.child_frame_id = "base_link"
        ts.transform.translation.x, ts.transform.translation.y = x, y
        ts.transform.rotation.x, ts.transform.rotation.y = qx, qy
        ts.transform.rotation.z, ts.transform.rotation.w = qz, qw
        self.tf_dyn.sendTransform(ts)

        od = Odometry()
        od.header.stamp = t
        od.header.frame_id = "odom"
        od.child_frame_id = "base_link"
        od.pose.pose.position.x, od.pose.pose.position.y = x, y
        od.pose.pose.orientation.z, od.pose.pose.orientation.w = qz, qw
        od.twist.twist.linear.x = v_fwd
        od.twist.twist.angular.z = yaw_rate
        self.pub_odom.publish(od)

        if "lidar" in data:
            self.publish_lidar(data["lidar"], t)
        if "imu" in data:
            self.publish_imu(data["imu"], t, qz, qw)
        if "camera" in data:
            self.publish_image(data["camera"], t)

    def publish_lidar(self, pts, t):
        # vehicle frame (CARLA) -> base_link (ROS): flip y; z stays height above ground
        x, y, z = pts[:, 0], -pts[:, 1], pts[:, 2]

        cloud = np.stack([x, y, z, pts[:, 3]], axis=1).astype(np.float32)
        pc = PointCloud2()
        pc.header.stamp, pc.header.frame_id = t, "base_link"
        pc.height, pc.width = 1, len(cloud)
        pc.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                     for i, n in enumerate(("x", "y", "z", "intensity"))]
        pc.is_bigendian, pc.point_step = False, 16
        pc.row_step, pc.is_dense = 16 * len(cloud), True
        pc.data = cloud.tobytes()
        self.pub_pts.publish(pc)

        # 2D scan: keep obstacles at body height, take the nearest return per angle bin
        keep = (z > SCAN_Z[0]) & (z < SCAN_Z[1])
        r = np.hypot(x[keep], y[keep])
        a = np.arctan2(y[keep], x[keep])
        ok = r > SCAN_MIN_R
        r, a = r[ok], a[ok]
        ranges = np.full(SCAN_BINS, np.inf, dtype=np.float32)
        idx = ((a + math.pi) / (2 * math.pi) * SCAN_BINS).astype(np.int32) % SCAN_BINS
        np.minimum.at(ranges, idx, r.astype(np.float32))
        sc = LaserScan()
        sc.header.stamp, sc.header.frame_id = t, "laser"
        sc.angle_min, sc.angle_max = -math.pi, math.pi - 2 * math.pi / SCAN_BINS
        sc.angle_increment = 2 * math.pi / SCAN_BINS
        sc.scan_time = self.args.dt
        sc.range_min, sc.range_max = SCAN_MIN_R, LIDAR_CFG["range"]
        sc.ranges = ranges.tolist()
        self.pub_scan.publish(sc)

    def publish_imu(self, m, t, qz, qw):
        im = Imu()
        im.header.stamp, im.header.frame_id = t, "imu"
        im.orientation.z, im.orientation.w = qz, qw
        # reflection y -> -y: vectors flip y; angular velocity (a pseudo-vector) flips x and z
        im.linear_acceleration.x, im.linear_acceleration.y = m["ax"], -m["ay"]
        im.linear_acceleration.z = m["az"]
        im.angular_velocity.x, im.angular_velocity.y = -m["gx"], m["gy"]
        im.angular_velocity.z = -m["gz"]
        self.pub_imu.publish(im)

    def publish_image(self, rgb, t):
        img = Image()
        img.header.stamp, img.header.frame_id = t, "camera"
        img.height, img.width = rgb.shape[:2]
        img.encoding, img.is_bigendian, img.step = "rgb8", 0, rgb.shape[1] * 3
        img.data = rgb.tobytes()
        self.pub_img.publish(img)

    def publish_route(self, route, t):
        p = PathMsg()
        p.header.stamp, p.header.frame_id = t, "odom"
        for wx, wy, wyaw in route[::2]:
            ps = PoseStamped()
            ps.header = p.header
            ps.pose.position.x, ps.pose.position.y = float(wx), float(-wy)
            _, _, qz, qw = yaw_quat(-math.radians(wyaw))
            ps.pose.orientation.z, ps.pose.orientation.w = qz, qw
            p.poses.append(ps)
        self.pub_route.publish(p)


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser(description="TwinDrive CARLA <-> ROS 2 bridge")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--tm-port", type=int, default=8000)
    ap.add_argument("--mode", choices=["pilot", "cmd_vel"], default="pilot")
    ap.add_argument("--spawn", default="main_gate")
    ap.add_argument("--tour", default="", help="'all' or comma list of landmarks to visit in order")
    ap.add_argument("--cruise", type=float, default=20.0, help="pilot cruise km/h")
    ap.add_argument("--traffic", type=int, default=0, help="moving cars smear SLAM maps; keep 0 to map")
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--camera-hz", type=float, default=2.0)
    ap.add_argument("--no-camera", action="store_true")
    ap.add_argument("--hide-vegetation", action="store_true")
    ap.add_argument("--odom-noise", type=float, default=0.0,
                    help="odometry drift per metre travelled (e.g. 0.02); 0 = perfect")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    world = client.get_world()
    cmap = world.get_map()
    if cmap.name.split("/")[-1] != DEFAULT_MAP:
        raise SystemExit(f"Loaded map is {cmap.name}; load '{DEFAULT_MAP}' first.")
    landmarks = load_landmarks(Path(__file__).resolve().parents[1] / f"config/landmarks_{DEFAULT_MAP}.json")
    if args.spawn not in landmarks:
        raise SystemExit(f"Unknown spawn '{args.spawn}'")
    tour = [n for n in landmarks if n != args.spawn] if args.tour == "all" else \
        [n.strip() for n in args.tour.split(",") if n.strip()]
    for n in tour:
        if n not in landmarks:
            raise SystemExit(f"Unknown landmark in --tour: {n}")

    rclpy.init()
    node = Bridge(args)
    log = node.get_logger()
    log.info("Building route planner ...")
    planner = RoutePlanner(cmap)

    original = world.get_settings()
    tm = client.get_trafficmanager(args.tm_port)
    ego, suite, traffic, hidden = None, None, [], []
    try:
        if args.hide_vegetation:
            hidden = find_vegetation(world)
            set_visible(world, hidden, False)
            log.info(f"Vegetation: {len(hidden)} objects hidden")
        s = world.get_settings()
        s.synchronous_mode, s.fixed_delta_seconds = True, args.dt
        world.apply_settings(s)
        tm.set_synchronous_mode(True)

        # ---------------------------------------------------- ego
        lm = landmarks[args.spawn]
        bp = world.get_blueprint_library().filter("vehicle.tesla.model3")[0]
        bp.set_attribute("role_name", "hero")
        ego = world.try_spawn_actor(bp, carla.Transform(
            carla.Location(lm["x"], lm["y"], lm["z"] + 0.5), carla.Rotation(yaw=lm["yaw"])))
        if ego is None:
            raise SystemExit("Could not spawn the ego car (spot occupied?)")
        world.tick()

        # ---------------------------------------------------- optional traffic
        if args.traffic > 0:
            pts = [p for p in cmap.get_spawn_points() if p.location.distance(ego.get_location()) > 15]
            rng.shuffle(pts)
            bps = [b for b in world.get_blueprint_library().filter("vehicle.*")
                   if int(b.get_attribute("number_of_wheels")) == 4]
            batch = [carla.command.SpawnActor(rng.choice(bps), p).then(
                carla.command.SetAutopilot(carla.command.FutureActor, True, tm.get_port()))
                for p in pts[:args.traffic]]
            traffic = [r.actor_id for r in client.apply_batch_sync(batch, True) if not r.error]
            log.info(f"Traffic: {len(traffic)} vehicles")

        suite = SensorSuite(world, ego, args.dt, camera=not args.no_camera,
                            cam_w=640, cam_h=360, cam_hz=args.camera_hz)
        world.tick()
        wheels = ego.get_physics_control().wheels
        max_steer = wheels[0].max_steer_angle
        wb = math.dist((wheels[0].position.x, wheels[0].position.y),
                       (wheels[2].position.x, wheels[2].position.y)) / 100.0
        wheelbase = wb if 1.5 < wb < 5.0 else 2.9
        others = [a for a in world.get_actors().filter("vehicle.*") if a.id != ego.id]

        pilot, tour_i = None, 0
        integ = 0.0  # PI state for cmd_vel speed tracking

        def plan_to(x, y, label):
            nonlocal pilot
            route = planner.plan(ego.get_location(), carla.Location(x, y, 0.0))
            if pilot is None:
                pilot = Pilot(route, wheelbase, max_steer, args.cruise, args.dt)
            else:
                pilot.set_route(route)
            node.publish_route(route, stamp(node.sim_t))
            log.info(f"Route to {label}: {len(route)} points")

        if args.mode == "pilot" and tour:
            plan_to(landmarks[tour[0]]["x"], landmarks[tour[0]]["y"], landmarks[tour[0]]["label"])

        log.info(f"Bridge running in '{args.mode}' mode. Ctrl+C to stop.")
        first = True
        while rclpy.ok():
            frame = world.tick()
            try:
                data = suite.get(frame)
            except queue.Empty:
                log.warn(f"sensor timeout at frame {frame}")
                continue
            node.sim_t = world.get_snapshot().timestamp.elapsed_seconds
            t = stamp(node.sim_t)
            node.pub_clock.publish(Clock(clock=t))
            if first:
                node.publish_static(t)
                first = False
            node.publish_state(ego, data, t, args.dt)
            rclpy.spin_once(node, timeout_sec=0.0)  # handle /goal_pose and /cmd_vel

            tf = ego.get_transform()
            v = ego.get_velocity()
            speed = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)

            if args.mode == "pilot":
                if node.pending_goal is not None:
                    gx, gy = node.pending_goal
                    node.pending_goal = None
                    try:
                        plan_to(gx, gy, f"clicked goal ({gx:.0f}, {-gy:.0f})")
                        tour = []  # a click overrides the tour
                    except RuntimeError as e:
                        log.warn(f"Planning failed: {e}")
                if pilot is not None:
                    out = pilot.step(tf.location.x, tf.location.y, tf.rotation.yaw, speed,
                                     find_lead(ego, others) if others else None)
                    ego.apply_control(carla.VehicleControl(
                        throttle=out["throttle"], brake=out["brake"], steer=out["steer"]))
                    if out["arrived"]:
                        log.info("Arrived.")
                        tour_i += 1
                        if tour_i < len(tour):
                            n = tour[tour_i]
                            plan_to(landmarks[n]["x"], landmarks[n]["y"], landmarks[n]["label"])
                        else:
                            pilot = None
                            if tour:
                                log.info("Tour complete. Save the map now (see README).")
                    elif out["off_route_m"] > 10.0:
                        log.warn("Off route; stopping. Click a new goal.")
                        pilot = None
                else:
                    ego.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
            else:  # cmd_vel: follow (v, w) like a car
                if node.cmd and node.sim_t - node.cmd[2] < CMD_TIMEOUT_S:
                    v_t, w_t = max(min(node.cmd[0], 8.0), 0.0), node.cmd[1]
                else:
                    v_t, w_t = 0.0, 0.0
                curvature = w_t / max(v_t, 0.5)
                steer = math.atan(wheelbase * curvature) / math.radians(max_steer)
                err = v_t - speed
                integ = float(np.clip(integ + err * args.dt, -3.0, 3.0))
                u = 0.35 * err + 0.05 * integ
                c = carla.VehicleControl(steer=float(np.clip(-steer, -1.0, 1.0)))  # ROS left+ -> CARLA right+
                if v_t < 0.05 and speed < 0.3:
                    c.brake, integ = 1.0, 0.0
                elif u >= 0:
                    c.throttle = float(min(u, 0.75))
                else:
                    c.brake, integ = float(min(-0.6 * u, 1.0)), max(integ, 0.0)
                ego.apply_control(c)

    except KeyboardInterrupt:
        pass
    finally:
        print("Cleaning up ...")
        if suite:
            suite.destroy()
        ids = list(traffic) + ([ego.id] if ego else [])
        if ids:
            client.apply_batch_sync([carla.command.DestroyActor(i) for i in ids], True)
        tm.set_synchronous_mode(False)
        world.apply_settings(original)
        if hidden:
            set_visible(world, hidden, True)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("Done.")


if __name__ == "__main__":
    main()
