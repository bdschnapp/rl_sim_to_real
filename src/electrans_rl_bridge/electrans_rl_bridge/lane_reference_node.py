"""
Lane reference node.

Loads the lanelet2 map from disk (the binary topic /map/vector_map cannot be
deserialized from Python — fromBinMsg is C++-only — so we use the same OSM
file the map_loader was given). Subscribes to ego odometry + an optional goal
pose. Publishes:

  /planning/lane_reference/centerline   nav_msgs/Path
  /planning/lane_reference/drive_enabled std_msgs/Bool

Active-lane selection:
  1. getCurrentLanelets(ego_xy) -> candidates
  2. If multiple, pick the one whose tangent yaw matches ego yaw best
     (handles bidirectional overlap).
  3. If a goal is set: prefer a candidate that has a routing path to the goal.

Centerline output is the active lanelet concatenated with N forward successors
(default 50 m), refined with utilities.generateFineCenterline(resolution=0.5 m).

drive_enabled goes True when: a goal is set, the goal is far enough (> 2 m),
an active lanelet is found, and a path was successfully published.
"""

from __future__ import annotations

import math
import os
from typing import Optional, List

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Bool


def _quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def _angle_diff(a: float, b: float) -> float:
    """Signed difference a - b wrapped to [-pi, pi]."""
    d = (a - b + math.pi) % (2 * math.pi) - math.pi
    return d


class LaneReferenceNode(Node):
    def __init__(self):
        super().__init__("lane_reference_node")

        # ----- params -----
        self.declare_parameter("map_path", "")
        self.declare_parameter("lanelet2_map_file", "lanelet2_map.osm")
        self.declare_parameter("forward_horizon_m", 50.0)
        self.declare_parameter("centerline_resolution_m", 0.5)
        self.declare_parameter("goal_reached_distance_m", 2.0)
        self.declare_parameter("publish_rate_hz", 10.0)

        self.map_path = self.get_parameter("map_path").value
        self.map_file = self.get_parameter("lanelet2_map_file").value
        self.forward_horizon_m = float(self.get_parameter("forward_horizon_m").value)
        self.centerline_resolution_m = float(self.get_parameter("centerline_resolution_m").value)
        self.goal_reached_distance_m = float(self.get_parameter("goal_reached_distance_m").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)

        # ----- load lanelet2 map from disk -----
        # Imports are inside __init__ so a missing lanelet2 stack fails this node
        # at construction with a clear message rather than at module-import time.
        import lanelet2
        from autoware_lanelet2_extension_python.projection import MGRSProjector
        import autoware_lanelet2_extension_python.utility.query as query
        import autoware_lanelet2_extension_python.utility.utilities as utilities

        self._lanelet2 = lanelet2
        self._query = query
        self._utilities = utilities

        map_file_full = os.path.join(self.map_path, self.map_file) if self.map_path else self.map_file
        self.get_logger().info(f"Loading lanelet2 map from {map_file_full}")
        # MGRSProjector is the only projector exposed by the Python bindings. For
        # Local-projection maps (lat="" lon="" + local_x/local_y tags, as produced
        # by VMB) every point projects to (0,0); we then apply the local_x/local_y
        # overrides ourselves, mirroring autoware_map_loader's LocalProjector path
        # (lanelet2_map_loader_node.cpp:140-175).
        projector = MGRSProjector(lanelet2.io.Origin(0.0, 0.0))
        self.lanelet_map = lanelet2.io.load(map_file_full, projector)

        # Mirrors autoware_map_loader's local-projection fixup
        # (lanelet2_map_loader_node.cpp:157-165). The C++ loader also calls
        # lanelet2.geometry.align on each lanelet's bounds afterwards, but the
        # align() helper is not exposed in the Python bindings; in practice the
        # bounds authored by VMB are already consistent so skipping it is fine.
        n_overridden = 0
        for point in self.lanelet_map.pointLayer:
            if "local_x" in point.attributes:
                point.x = float(point.attributes["local_x"])
                n_overridden += 1
            if "local_y" in point.attributes:
                point.y = float(point.attributes["local_y"])
        self.get_logger().info(f"Applied local_x/local_y to {n_overridden} points")

        traffic_rules = lanelet2.traffic_rules.create(
            lanelet2.traffic_rules.Locations.Germany,
            lanelet2.traffic_rules.Participants.Vehicle,
        )
        self.routing_graph = lanelet2.routing.RoutingGraph(self.lanelet_map, traffic_rules)
        self.all_lanelets = self._query.laneletLayer(self.lanelet_map)
        self.get_logger().info(f"Loaded {len(self.all_lanelets)} lanelets")

        # ----- state -----
        self._ego_pose: Optional[tuple] = None  # (x, y, yaw)
        self._goal_pose: Optional[tuple] = None  # (x, y)
        self._last_active_id: Optional[int] = None

        # ----- pub / sub -----
        transient = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )

        self.pub_path = self.create_publisher(Path, "/planning/lane_reference/centerline", transient)
        self.pub_drive = self.create_publisher(Bool, "/planning/lane_reference/drive_enabled", transient)

        self.create_subscription(Odometry, "/localization/kinematic_state", self._on_odom, 10)
        # autoware.rviz wires the SetGoal tool to /planning/mission_planning/goal
        # (see rviz/autoware.rviz tool block "rviz_default_plugins/SetGoal").
        self.create_subscription(PoseStamped, "/planning/mission_planning/goal", self._on_goal, 1)

        self.create_timer(1.0 / self.publish_rate_hz, self._on_timer)

    # --------------------------------------------------------------- inputs
    def _on_odom(self, msg: Odometry):
        p = msg.pose.pose
        yaw = _quat_to_yaw(p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w)
        self._ego_pose = (p.position.x, p.position.y, yaw)

    def _on_goal(self, msg: PoseStamped):
        self._goal_pose = (msg.pose.position.x, msg.pose.position.y)
        self.get_logger().info(f"Goal set: ({self._goal_pose[0]:.2f}, {self._goal_pose[1]:.2f})")

    # --------------------------------------------------------- active lane
    def _pick_active_lanelet(self):
        if self._ego_pose is None:
            return None
        ex, ey, eyaw = self._ego_pose
        from geometry_msgs.msg import Point as RosPoint
        ego_point = RosPoint(x=ex, y=ey, z=0.0)
        candidates = self._query.getCurrentLanelets(self.all_lanelets, ego_point)
        if not candidates:
            # Fall back to nearest
            from geometry_msgs.msg import Pose as RosPose
            from geometry_msgs.msg import Quaternion
            pose = RosPose()
            pose.position.x, pose.position.y = ex, ey
            pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            try:
                return self._query.getClosestLanelet(self.all_lanelets, pose)
            except Exception:
                return None

        # Tangent vs ego-yaw alignment error per candidate.
        scored = []
        for ll in candidates:
            try:
                tangent = self._utilities.getLaneletAngle(ll, ego_point)
            except Exception:
                continue
            err = abs(_angle_diff(eyaw, tangent))
            scored.append((err, ll))
        if not scored:
            return None
        scored.sort(key=lambda x: x[0])
        best_err, best = scored[0]

        # Hysteresis: with two parallel opposite-direction lanelets the
        # alignment error is nearly identical near yaw=±π/2 and noise flips
        # the choice. If the previously-active lanelet is still a candidate
        # and its tangent-error is within a margin of the new best, keep it.
        # Without this the ref path direction flips every tick and the policy
        # commands max steer.
        margin = math.radians(45.0)
        if self._last_active_id is not None:
            prev = next((ll for _, ll in scored if ll.id == self._last_active_id), None)
            if prev is not None:
                prev_err = next(e for e, ll in scored if ll.id == self._last_active_id)
                if prev_err <= best_err + margin:
                    return prev
        return best

    def _concat_centerline(self, active_lanelet) -> List[tuple]:
        # Build seq = [N predecessors] + [active] + [as many successors as
        # needed to cover forward_horizon_m, looping back to active if the
        # network is a closed ring]. The MVSL map's lanelets are ~3 m each
        # and the policy was trained on long paths; if we publish only the
        # active lanelet, the truck drives off the end almost immediately
        # and the env's projection-to-path obs goes to infinity.
        seq: List = []
        visited = set()

        # One predecessor so the path covers behind the truck (e_y_t depends
        # on trailer position, which sits behind the tractor).
        try:
            prevs = self.routing_graph.previous(active_lanelet)
        except Exception:
            prevs = []
        if prevs:
            seq.append(prevs[0])
            visited.add(prevs[0].id)

        seq.append(active_lanelet)
        visited.add(active_lanelet.id)

        # Walk forward until total length >= forward_horizon_m or we loop.
        total_len = self._approx_length(active_lanelet)
        current = active_lanelet
        max_steps = 64  # safety cap; lanelet rings shouldn't be longer
        for _ in range(max_steps):
            if total_len >= self.forward_horizon_m:
                break
            try:
                nexts = self.routing_graph.following(current)
            except Exception:
                nexts = []
            if not nexts:
                break
            nxt = nexts[0]
            seq.append(nxt)
            total_len += self._approx_length(nxt)
            if nxt.id in visited:
                # closed ring; include this duplicate to bridge the seam and stop
                break
            visited.add(nxt.id)
            current = nxt

        try:
            combined = self._utilities.combineLaneletsShape(seq)
            fine = self._utilities.generateFineCenterline(combined, self.centerline_resolution_m)
            return [(pt.x, pt.y) for pt in fine]
        except Exception:
            # Fall back to raw centerline points
            pts = []
            for ll in seq:
                for p in ll.centerline:
                    pts.append((p.x, p.y))
            return pts

    def _approx_length(self, lanelet) -> float:
        """Sum of segment lengths of the lanelet centerline. Used to decide
        when we have enough successors stacked up."""
        pts = list(lanelet.centerline)
        if len(pts) < 2:
            return 0.0
        total = 0.0
        for a, b in zip(pts[:-1], pts[1:]):
            total += math.hypot(b.x - a.x, b.y - a.y)
        return total

    # ------------------------------------------------------------- publish
    def _on_timer(self):
        if self._ego_pose is None:
            self._publish_drive(False)
            return

        active = self._pick_active_lanelet()
        if active is None:
            self._publish_drive(False)
            return

        if active.id != self._last_active_id:
            self.get_logger().info(f"Active lanelet -> {active.id}")
            self._last_active_id = active.id

        polyline = self._concat_centerline(active)
        if len(polyline) < 2:
            self._publish_drive(False)
            return

        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = "map"
        for x, y in polyline:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        self.pub_path.publish(path)

        drive = self._goal_pose is not None
        if drive:
            ex, ey, _ = self._ego_pose
            gx, gy = self._goal_pose
            dist = math.hypot(ex - gx, ey - gy)
            drive = dist > self.goal_reached_distance_m
        self._publish_drive(drive)

    def _publish_drive(self, val: bool):
        msg = Bool()
        msg.data = bool(val)
        self.pub_drive.publish(msg)


def main():
    rclpy.init()
    node = LaneReferenceNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
