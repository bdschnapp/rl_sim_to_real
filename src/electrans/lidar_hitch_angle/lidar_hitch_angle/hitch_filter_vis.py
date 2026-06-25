"""
hitch_filter_vis.py
-------------------
Hitch-angle estimator with:
  - L-shape rectangle fitting  (robust to partial front + side face visibility)
  - 10-sample moving-average filter
  - Kalman filter  (constant-angular-velocity model, with innovation gating)
  - RViz2 visualization: 3-D ROI bounding box, filtered direction arrow, text overlay

All ROI geometry is configurable via ROS2 parameters – no recompile needed:

  ros2 run lidar_hitch_angle hitchangle_filtered --ros-args \
    -p roi_x_min:=-4.0  -p roi_x_max:=-1.5 \
    -p roi_y_half:=0.9  -p roi_z_min:=-0.3  -p roi_z_max:=2.5

The truck/trailer dimensions default to the measured geometry used by
autoware_truck_trailer_freespace_planner. They are used as priors during
rectangle fitting and heading-ambiguity resolution.

Topics published
----------------
  /hitch_angle_raw        (std_msgs/Float64)  – raw L-shape estimate  (deg)
  /hitch_angle_moving_avg (std_msgs/Float64)  – moving-average output (deg)
  /hitch_angle_kalman     (std_msgs/Float64)  – Kalman-filter output  (deg)
  /vehicle/trailer_state  (autoware_vehicle_msgs/TrailerState) – Autoware input (rad)
  /hitch_visualization    (visualization_msgs/MarkerArray)
"""

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from autoware_vehicle_msgs.msg import TrailerState
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import Float64

import numpy as np
from sensor_msgs_py import point_cloud2
from collections import deque

# Marker lifetime: stale markers disappear after this many seconds
_MARKER_LIFETIME = Duration(seconds=0.5)


class HitchAngleEstimatorFiltered(Node):
    """
    L-shape rectangle fitting + temporal filters + RViz visualisation.

    L-shape fitting (closeness criterion, Zhang et al. 2017):
      For each candidate rectangle orientation θ ∈ [0, π):
        rotate points by θ, compute axis-aligned bounding box,
        score = Σ  min(dist_to_x_edge, dist_to_y_edge)²
      The θ that minimises the score aligns the bounding rectangle with the
      trailer's true face(s), regardless of whether one or two faces are visible.
    """

    def __init__(self):
        super().__init__('hitch_angle_estimator_filtered')

        # ── ROI parameters  ───────────────────────────────────────────────────
        # Tune these to put the green box on the trailer front face.
        # Use RViz's "Publish Point" tool to click on the trailer and read
        # the coordinates, then pass them with --ros-args -p roi_x_min:=...
        #
        # For a trailer 1-1.5 m wide whose front face is ~2-4 m behind the LiDAR:
        #   roi_x_min  = -(hitch_to_face + margin)   e.g. -4.0
        #   roi_x_max  = -(hitch_distance - margin)  e.g. -1.5
        #   roi_y_half = half trailer width + margin  e.g.  0.9
        self.declare_parameter('roi_x_min',  -4.0)
        self.declare_parameter('roi_x_max',  -1.5)
        self.declare_parameter('roi_y_half',  0.9)   # half-width of ROI in Y (metres)
        self.declare_parameter('roi_z_min',  -0.3)
        self.declare_parameter('roi_z_max',   2.5)
        # Maximum ROI tracking shift in Y when the trailer turns
        self.declare_parameter('roi_max_y_shift', 1.2)

        # Geometry priors, matched to truck_trailer_freespace_planner.param.yaml.
        self.declare_parameter('tractor_base2hinge', 0.20)
        self.declare_parameter('trailer_front_oh', 0.39)
        self.declare_parameter('trailer_wheel_base', 2.75)
        self.declare_parameter('trailer_rear_oh', 0.51)
        self.declare_parameter('trailer_width', 0.62)
        self.declare_parameter('use_dimension_prior', True)
        self.declare_parameter('dimension_prior_weight', 1.0)
        self.declare_parameter('dimension_heading_weight', 3.0)
        self.declare_parameter('dimension_width_tolerance_m', 0.20)
        self.declare_parameter('dimension_length_margin_m', 0.50)

        # Autoware TrailerState output.
        # hitch_angle_sign lets field calibration match Autoware's convention:
        #   hitch_angle = theta_truck - theta_trailer, in radians.
        self.declare_parameter('publish_trailer_state', True)
        self.declare_parameter('trailer_state_topic', '/vehicle/trailer_state')
        self.declare_parameter('hitch_angle_sign', 1.0)
        self.declare_parameter('hitch_angle_offset_deg', 0.0)
        self.declare_parameter('max_abs_hitch_angle_deg', 90.0)

        # ── subscriptions / publications ──────────────────────────────────────
        self.subscription = self.create_subscription(
            PointCloud2, '/rslidar_points', self.pointcloud_callback, 10)

        self.marker_pub = self.create_publisher(
            MarkerArray, '/hitch_visualization', 10)
        self.angle_raw_pub = self.create_publisher(
            Float64, '/hitch_angle_raw', 10)
        self.angle_moving_avg_pub = self.create_publisher(
            Float64, '/hitch_angle_moving_avg', 10)
        self.angle_kalman_pub = self.create_publisher(
            Float64, '/hitch_angle_kalman', 10)
        trailer_state_topic = str(
            self.get_parameter('trailer_state_topic').value)
        self.trailer_state_pub = self.create_publisher(
            TrailerState, trailer_state_topic, 10)

        # ── state ─────────────────────────────────────────────────────────────
        self.hitch_angle = 0.0          # last accepted estimate (drives ambiguity resolution)

        # ── moving-average filter ─────────────────────────────────────────────
        self.ma_window = 10
        self.angle_buffer: deque = deque(maxlen=self.ma_window)

        # ── Kalman filter ─────────────────────────────────────────────────────
        # State:  x = [angle (deg), angular_velocity (deg/s)]
        # Model:  constant angular-velocity  (angle_{k+1} = angle_k + vel_k * dt)
        self.kf_x = np.array([[0.0], [0.0]])
        self.kf_P = np.diag([100.0, 10.0])   # initial covariance (deg², (deg/s)²)
        self.kf_H = np.array([[1.0, 0.0]])    # observe angle only
        self.kf_R = np.array([[4.0]])         # measurement noise (deg²)  – ↑ = trust model more
        self.kf_Q = np.diag([0.05, 0.5])      # process noise             – ↑ = faster tracking

        # Innovation gating: reject measurement if Mahalanobis distance > gate
        # chi2(1) 99.7% confidence = 9.0  →  ~3σ gate on the innovation
        self.kf_innovation_gate = 9.0

        self.last_stamp_ns: int | None = None

    # ══════════════════════════════════════════════════════════════════════════
    #  ROS callback
    # ══════════════════════════════════════════════════════════════════════════

    def pointcloud_callback(self, msg: PointCloud2) -> None:
        # Compute dt for Kalman prediction step
        stamp_msg = msg.header.stamp
        stamp_ns = self._stamp_to_nanoseconds(stamp_msg)
        if stamp_ns <= 0:
            stamp_msg = self.get_clock().now().to_msg()
            stamp_ns = self._stamp_to_nanoseconds(stamp_msg)

        dt = ((stamp_ns - self.last_stamp_ns) * 1e-9
              if self.last_stamp_ns is not None else 0.1)
        dt = float(np.clip(dt, 1e-3, 1.0))
        self.last_stamp_ns = stamp_ns

        points = self._cloud_to_numpy(msg)
        if points.shape[0] == 0:
            self.get_logger().warn("Empty point cloud")
            return

        roi_points, roi_bounds = self._filter_roi(points)
        n_roi = roi_points.shape[0]
        if n_roi < 10:
            self.get_logger().warn(
                f"Only {n_roi} points in ROI – check roi_x/y/z parameters")
            return

        raw_angle = self._compute_hitch_angle_lshape(roi_points)
        self.hitch_angle = raw_angle

        # ── filters ──────────────────────────────────────────────────────────
        self.angle_buffer.append(raw_angle)
        moving_avg = float(np.mean(self.angle_buffer))
        kalman_angle = self._kalman_update(raw_angle, dt)

        # ── publish ───────────────────────────────────────────────────────────
        self._pub_float(self.angle_raw_pub,        raw_angle)
        self._pub_float(self.angle_moving_avg_pub, moving_avg)
        self._pub_float(self.angle_kalman_pub,     kalman_angle)
        self._publish_trailer_state(kalman_angle, stamp_msg)
        self._publish_markers(roi_bounds, roi_points,
                              raw_angle, moving_avg, kalman_angle, msg.header)

        self.get_logger().info(
            f"pts={n_roi:3d}  "
            f"Raw:{raw_angle:+6.1f}°  "
            f"MA:{moving_avg:+6.1f}°  "
            f"KF:{kalman_angle:+6.1f}°")

    # ══════════════════════════════════════════════════════════════════════════
    #  Point-cloud helper
    # ══════════════════════════════════════════════════════════════════════════

    def _cloud_to_numpy(self, cloud_msg: PointCloud2) -> np.ndarray:
        return np.array([
            [p[0], p[1], p[2]]
            for p in point_cloud2.read_points(
                cloud_msg, field_names=("x", "y", "z"), skip_nans=True)
        ])

    # ══════════════════════════════════════════════════════════════════════════
    #  ROI filter  (uses ROS2 parameters + trig-based Y tracking)
    # ══════════════════════════════════════════════════════════════════════════

    def _filter_roi(self, points: np.ndarray):
        """
        Return (roi_points, bounds) where bounds = (x_min, x_max, y_min, y_max, z_min, z_max).

        The ROI Y-centre tracks the trailer face as it turns.
        The shift is computed geometrically:
          y_shift = -D_face * sin(hitch_angle)
        where D_face = half the X span of the ROI (rough distance pivot→face).
        The Kalman estimate is used (smoother than raw) to avoid feedback loops.
        """
        x_min = float(self.get_parameter('roi_x_min').value)
        x_max = float(self.get_parameter('roi_x_max').value)
        y_half = float(self.get_parameter('roi_y_half').value)
        z_min = float(self.get_parameter('roi_z_min').value)
        z_max = float(self.get_parameter('roi_z_max').value)
        max_shift = float(self.get_parameter('roi_max_y_shift').value)

        # Geometric Y-tracking: use the Kalman angle (smoothest estimate)
        kf_angle_rad = np.radians(float(self.kf_x[0, 0]))
        d_face = abs(x_max - x_min) / 2.0          # approx distance pivot → face
        y_shift = float(np.clip(-d_face * np.sin(kf_angle_rad), -max_shift, max_shift))

        y_min = -y_half + y_shift
        y_max =  y_half + y_shift

        mask = (
            (points[:, 0] > x_min) & (points[:, 0] < x_max) &
            (points[:, 1] > y_min) & (points[:, 1] < y_max) &
            (points[:, 2] > z_min) & (points[:, 2] < z_max)
        )
        return points[mask], (x_min, x_max, y_min, y_max, z_min, z_max)

    # ══════════════════════════════════════════════════════════════════════════
    #  L-shape rectangle fitting
    # ══════════════════════════════════════════════════════════════════════════

    def _compute_hitch_angle_lshape(self, points: np.ndarray) -> float:
        """
        Fit a minimum-closeness bounding rectangle to the XY projection.

        Closeness criterion (Zhang et al. 2017):
          For a candidate angle θ, rotate the 2-D cloud and compute the
          axis-aligned bounding box.  For each point, the "closeness" to the
          box is min(dist_to_near_x_edge, dist_to_near_y_edge).
          We minimise the sum of squared closeness values over all θ.

        Handles:
          – front face only  (straight / small hitch)
          – front + side     (L-shape, moderate hitch)
          – side face only   (large hitch)

        Heading ambiguity (four 90° candidates) resolved by picking the one
        closest to the previous Kalman estimate.
        """
        pts2d = points[:, :2]

        # ── coarse search: 1° steps over [0, π) ──────────────────────────────
        thetas = np.linspace(0.0, np.pi, 180, endpoint=False)
        best_score = np.inf
        best_theta = 0.0

        for theta in thetas:
            score = self._rectangle_fit_score(pts2d, theta)

            if score < best_score:
                best_score = score
                best_theta = theta

        # ── fine search: 0.1° steps in a ±1° window ──────────────────────────
        for theta in np.linspace(best_theta - np.radians(1.0),
                                 best_theta + np.radians(1.0), 20):
            score = self._rectangle_fit_score(pts2d, theta)

            if score < best_score:
                best_score = score
                best_theta = theta

        # ── resolve heading ambiguity with Kalman estimate ────────────────────
        candidates = np.array([
            best_theta,
            best_theta + np.pi / 2.0,
            best_theta + np.pi,
            best_theta + 3.0 * np.pi / 2.0,
        ]) % (2.0 * np.pi)

        # Use Kalman state for the reference (more stable than raw hitch_angle)
        prev_heading = np.radians(float(self.kf_x[0, 0])) % (2.0 * np.pi)

        def _adiff(a: float, b: float) -> float:
            d = abs(a - b) % (2.0 * np.pi)
            return float(min(d, 2.0 * np.pi - d))

        best_heading = float(min(candidates,
                                 key=lambda c: (
                                     _adiff(c, prev_heading) +
                                     self._heading_prior_penalty(
                                         pts2d, best_theta, c))))

        # ── signed hitch angle: +X is truck-forward, CCW positive ─────────────
        h_vec = np.array([np.cos(best_heading), np.sin(best_heading)])
        truck_fwd = np.array([1.0, 0.0])
        dot = float(np.dot(truck_fwd, h_vec))
        det = float(truck_fwd[0] * h_vec[1] - truck_fwd[1] * h_vec[0])
        return float(np.degrees(np.arctan2(det, dot)))

    def _rectangle_fit_score(self, pts2d: np.ndarray, theta: float) -> float:
        c, s = np.cos(theta), np.sin(theta)
        u = pts2d[:, 0] * c + pts2d[:, 1] * s
        v = -pts2d[:, 0] * s + pts2d[:, 1] * c

        du = np.minimum(np.abs(u - u.min()), np.abs(u - u.max()))
        dv = np.minimum(np.abs(v - v.min()), np.abs(v - v.max()))
        closeness_score = float(np.sum(np.minimum(du, dv) ** 2))
        prior_score = self._dimension_prior_score(
            float(u.max() - u.min()), float(v.max() - v.min()), pts2d.shape[0])
        return closeness_score + prior_score

    def _dimension_prior_score(
        self, extent_u: float, extent_v: float, point_count: int
    ) -> float:
        if not bool(self.get_parameter('use_dimension_prior').value):
            return 0.0

        weight = float(self.get_parameter('dimension_prior_weight').value)
        if weight <= 0.0:
            return 0.0

        trailer_width = float(self.get_parameter('trailer_width').value)
        trailer_length = (
            float(self.get_parameter('trailer_front_oh').value) +
            float(self.get_parameter('trailer_wheel_base').value) +
            float(self.get_parameter('trailer_rear_oh').value)
        )
        width_tol = float(
            self.get_parameter('dimension_width_tolerance_m').value)
        length_margin = float(
            self.get_parameter('dimension_length_margin_m').value)
        if trailer_width <= 0.0 or trailer_length <= 0.0:
            return 0.0

        width_error = min(
            abs(extent_u - trailer_width), abs(extent_v - trailer_width))
        width_error = max(0.0, width_error - width_tol)
        length_excess = max(
            0.0, max(extent_u, extent_v) - (trailer_length + length_margin))
        return float(point_count) * weight * (width_error ** 2 + length_excess ** 2)

    def _heading_prior_penalty(
        self, pts2d: np.ndarray, rectangle_theta: float, heading: float
    ) -> float:
        if not bool(self.get_parameter('use_dimension_prior').value):
            return 0.0

        weight = float(self.get_parameter('dimension_heading_weight').value)
        if weight <= 0.0:
            return 0.0

        trailer_width = float(self.get_parameter('trailer_width').value)
        width_tol = float(
            self.get_parameter('dimension_width_tolerance_m').value)
        if trailer_width <= 0.0:
            return 0.0

        extent_u, extent_v = self._rectangle_extents(pts2d, rectangle_theta)
        err_u = abs(extent_u - trailer_width)
        err_v = abs(extent_v - trailer_width)
        if abs(err_u - err_v) < width_tol:
            return 0.0

        lateral_axis = rectangle_theta if err_u < err_v else rectangle_theta + np.pi / 2.0
        longitudinal_axes = [
            (lateral_axis + np.pi / 2.0) % (2.0 * np.pi),
            (lateral_axis + 3.0 * np.pi / 2.0) % (2.0 * np.pi),
        ]
        axis_error = min(self._angle_diff(heading, axis) for axis in longitudinal_axes)
        return weight * axis_error

    @staticmethod
    def _rectangle_extents(pts2d: np.ndarray, theta: float) -> tuple[float, float]:
        c, s = np.cos(theta), np.sin(theta)
        u = pts2d[:, 0] * c + pts2d[:, 1] * s
        v = -pts2d[:, 0] * s + pts2d[:, 1] * c
        return float(u.max() - u.min()), float(v.max() - v.min())

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        d = abs(a - b) % (2.0 * np.pi)
        return float(min(d, 2.0 * np.pi - d))

    # ══════════════════════════════════════════════════════════════════════════
    #  Kalman filter  –  constant angular-velocity model
    # ══════════════════════════════════════════════════════════════════════════

    def _kalman_update(self, measurement: float, dt: float) -> float:
        F = np.array([[1.0, dt],
                      [0.0, 1.0]])

        # Predict
        x_pred = F @ self.kf_x
        P_pred = F @ self.kf_P @ F.T + self.kf_Q

        # Innovation and Mahalanobis gating
        z = np.array([[measurement]])
        innovation = z - self.kf_H @ x_pred
        S = self.kf_H @ P_pred @ self.kf_H.T + self.kf_R
        mahal = float(innovation.T @ np.linalg.inv(S) @ innovation)

        if mahal > self.kf_innovation_gate:
            # Outlier: only propagate prediction, skip the measurement update
            self.get_logger().warn(
                f"KF gate reject: innovation={float(innovation[0,0]):+.1f}°  "
                f"mahal={mahal:.1f}")
            self.kf_x = x_pred
            self.kf_P = P_pred
        else:
            K = P_pred @ self.kf_H.T @ np.linalg.inv(S)
            self.kf_x = x_pred + K @ innovation
            self.kf_P = (np.eye(2) - K @ self.kf_H) @ P_pred

        return float(self.kf_x[0, 0])

    # ══════════════════════════════════════════════════════════════════════════
    #  Visualisation
    # ══════════════════════════════════════════════════════════════════════════

    def _publish_markers(self, roi_bounds, roi_points: np.ndarray,
                         raw_angle: float, moving_avg: float,
                         kalman_angle: float, header) -> None:
        markers = MarkerArray()
        lifetime_msg = _MARKER_LIFETIME.to_msg()
        x_min, x_max, y_min, y_max, z_min, z_max = roi_bounds

        # ── Marker 0: 3-D ROI bounding box (green wire-frame) ────────────────
        bbox = Marker()
        bbox.header = header
        bbox.ns = 'hitch_roi'
        bbox.id = 0
        bbox.type = Marker.LINE_LIST
        bbox.action = Marker.ADD
        bbox.lifetime = lifetime_msg
        bbox.scale.x = 0.03          # line width  (m)
        bbox.color.r = 0.0
        bbox.color.g = 1.0
        bbox.color.b = 0.0
        bbox.color.a = 0.9

        corners = [
            (x_min, y_min, z_min), (x_max, y_min, z_min),   # 0 1
            (x_min, y_max, z_min), (x_max, y_max, z_min),   # 2 3
            (x_min, y_min, z_max), (x_max, y_min, z_max),   # 4 5
            (x_min, y_max, z_max), (x_max, y_max, z_max),   # 6 7
        ]
        edges = [
            (0, 1), (2, 3), (4, 5), (6, 7),
            (0, 2), (1, 3), (4, 6), (5, 7),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        for a, b in edges:
            for idx in (a, b):
                cx, cy, cz = corners[idx]
                bbox.points.append(Point(x=cx, y=cy, z=cz))
        markers.markers.append(bbox)

        # ── Marker 1: Kalman-filtered heading arrow (orange) ─────────────────
        if roi_points.shape[0] >= 3:
            mean_pt = np.mean(roi_points, axis=0)
            heading_rad = np.radians(kalman_angle)
            arrow_len = abs(x_max - x_min) * 0.8   # scale with ROI size

            arrow = Marker()
            arrow.header = header
            arrow.ns = 'hitch_direction'
            arrow.id = 1
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            arrow.lifetime = lifetime_msg
            arrow.scale.x = 0.06    # shaft diameter
            arrow.scale.y = 0.12    # head diameter
            arrow.scale.z = 0.15    # head length
            arrow.color.r = 1.0
            arrow.color.g = 0.5
            arrow.color.b = 0.0
            arrow.color.a = 1.0
            arrow.points = [
                Point(x=float(mean_pt[0]),
                      y=float(mean_pt[1]),
                      z=float(mean_pt[2])),
                Point(x=float(mean_pt[0] + np.cos(heading_rad) * arrow_len),
                      y=float(mean_pt[1] + np.sin(heading_rad) * arrow_len),
                      z=float(mean_pt[2])),
            ]
            markers.markers.append(arrow)

        # ── Marker 2: text overlay ────────────────────────────────────────────
        text = Marker()
        text.header = header
        text.ns = 'hitch_text'
        text.id = 2
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.lifetime = lifetime_msg
        text.pose.position.x = float((x_min + x_max) / 2.0)
        text.pose.position.y = float((y_min + y_max) / 2.0)
        text.pose.position.z = float(z_max + 0.2)
        text.scale.z = 0.25
        text.color.r = 1.0
        text.color.g = 1.0
        text.color.b = 1.0
        text.color.a = 1.0
        text.text = (f"Raw:{raw_angle:+.1f}\n"
                     f"MA:{moving_avg:+.1f}\n"
                     f"KF:{kalman_angle:+.1f}")
        markers.markers.append(text)

        self.marker_pub.publish(markers)

    # ══════════════════════════════════════════════════════════════════════════
    #  Utility
    # ══════════════════════════════════════════════════════════════════════════

    def _pub_float(self, publisher, value: float) -> None:
        msg = Float64()
        msg.data = float(value)
        publisher.publish(msg)

    @staticmethod
    def _stamp_to_nanoseconds(stamp) -> int:
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def _publish_trailer_state(self, angle_deg: float, stamp) -> None:
        if not bool(self.get_parameter('publish_trailer_state').value):
            return

        sign = float(self.get_parameter('hitch_angle_sign').value)
        offset_deg = float(self.get_parameter('hitch_angle_offset_deg').value)
        max_abs_deg = float(self.get_parameter('max_abs_hitch_angle_deg').value)

        calibrated_deg = sign * float(angle_deg) + offset_deg
        if max_abs_deg > 0.0:
            calibrated_deg = float(
                np.clip(calibrated_deg, -max_abs_deg, max_abs_deg))

        msg = TrailerState()
        msg.stamp = stamp
        msg.hitch_angle = float(np.radians(calibrated_deg))
        msg.hitch_rate = float(np.radians(sign * float(self.kf_x[1, 0])))
        self.trailer_state_pub.publish(msg)


# ─────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = HitchAngleEstimatorFiltered()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
