"""
Lane reference node.

Loads the lanelet2 map from disk (the binary topic /map/vector_map cannot be
deserialized from Python — fromBinMsg is C++-only — so we use the same OSM
file the map_loader was given). Subscribes to ego odometry + an optional goal
pose. Publishes:

  /planning/lane_reference/centerline       nav_msgs/Path   (latched, transient_local)
  /planning/lane_reference/drive_enabled    std_msgs/Bool   (latched)
  /planning/lane_reference/drive_direction  std_msgs/Bool   (latched; True = reverse)

Active-lane selection (post-bidirectional-collapse: each point is contained in
at most one lanelet, so yaw-tangent disambiguation is no longer required):
  1. getCurrentLanelets(ego_xy) -> candidates.
  2. If a goal is set, prefer the candidate that ALSO contains the goal.
     This is the lanelet on which getArcCoordinates gives meaningful direction
     inference (s_goal < s_ego => reverse). Goals that fall in a different
     lanelet or in empty space are flagged once via a console warning; the
     bridge still drives on the ego candidate.
  3. Hysteresis on lanelet id stops short-lived flapping at boundaries.

Centerline output is the active lanelet concatenated with N forward successors
(default 50 m), refined with utilities.generateFineCenterline(resolution=0.5 m).

drive_enabled goes True when: a goal is set, the goal is far enough (> 2 m),
an active lanelet is found, and a path was successfully published.

drive_direction goes True (reverse) when the truck must physically drive in
reverse to reach the goal — i.e. the lane direction needed to reach the goal
disagrees with the truck's current heading. Computed from
goal_downstream (s_goal vs s_ego on canonical) XOR heading_canonical_aligned
(dot product of ego heading with canonical tangent at ego).
"""

from __future__ import annotations

import math
import os
from typing import Optional, List

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Bool


def _pose_at(x: float, y: float) -> Pose:
    """Build a yaw-agnostic Pose at (x, y, 0). Arc-length / containment
    queries on the lanelet2 utility API only read .position."""
    p = Pose()
    p.position = Point(x=float(x), y=float(y), z=0.0)
    p.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    return p


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
        # Drive direction is latched at goal-set time and held until the next
        # goal arrives. Recomputing every tick produces flapping near the goal
        # (s_goal - s_ego crosses zero) and near lanelet seams (tangent jumps
        # ± across segment boundaries when the projection switches segments).
        # None means "no goal yet, or first tick since a new goal was set —
        # decide on the next tick where we have an active lanelet".
        self._goal_drive_reverse: Optional[bool] = None
        # Track whether we've already logged the "goal not on ego's lanelet"
        # warning so we don't spam the console every tick while the user
        # leaves a stale goal in a non-adjacent lanelet.
        self._warned_goal_off_lanelet: bool = False

        # ----- pub / sub -----
        transient = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )

        self.pub_path = self.create_publisher(Path, "/planning/lane_reference/centerline", transient)
        self.pub_drive = self.create_publisher(Bool, "/planning/lane_reference/drive_enabled", transient)
        # True = drive in REVERSE along the canonical lanelet direction (goal is
        # upstream of the ego on the active lanelet's arc-length parameterisation).
        # Latched so the bridge picks up the last decision even if it subscribes late.
        self.pub_drive_direction = self.create_publisher(
            Bool, "/planning/lane_reference/drive_direction", transient
        )

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
        # New goal -> reset the off-lanelet warning latch so the user gets a
        # fresh warning if this goal also lands in a non-adjacent lanelet.
        self._warned_goal_off_lanelet = False
        # New goal -> drop the cached direction decision so the next timer
        # tick recomputes it against this goal's geometry.
        self._goal_drive_reverse = None
        self.get_logger().info(f"Goal set: ({self._goal_pose[0]:.2f}, {self._goal_pose[1]:.2f})")

    # --------------------------------------------------------- active lane
    def _pick_active_lanelet(self):
        """Pick the lanelet that should drive the centerline + direction-inference
        logic. After the bidirectional-collapse Milestone 2, each point is
        contained in at most one lanelet, so the yaw-vs-tangent disambiguation
        we used to need is moot. New policy:

          1. Find lanelets containing the ego point.
          2. If a goal is set, find lanelets containing the goal and prefer
             the ego-candidate that also contains the goal — that's the
             lanelet on which `getArcCoordinates` will give a meaningful
             `s_ego < s_goal` (or `>`, indicating reverse).
          3. If no overlap with the goal's lanelet, log once and fall back to
             the ego candidate (the bridge will still drive correctly thanks
             to the local-tangent flip in ros_env_adapter, but direction
             inference will be undefined until the goal moves onto the
             ego's lanelet — multi-lanelet routing is out of scope here).
          4. Hysteresis on ID still applies as a tiebreaker.
        """
        if self._ego_pose is None:
            return None
        ex, ey, _ = self._ego_pose
        ego_point = Point(x=ex, y=ey, z=0.0)
        ego_candidates = list(
            self._query.getCurrentLanelets(self.all_lanelets, ego_point)
        )
        if not ego_candidates:
            # Off-lanelet ego: fall back to nearest lanelet so the bridge has
            # something to project onto.
            try:
                return self._query.getClosestLanelet(self.all_lanelets, _pose_at(ex, ey))
            except Exception:
                return None

        # Goal-aware narrowing: prefer the candidate that also contains the goal.
        if self._goal_pose is not None:
            gx, gy = self._goal_pose
            goal_point = Point(x=gx, y=gy, z=0.0)
            goal_candidates = list(
                self._query.getCurrentLanelets(self.all_lanelets, goal_point)
            )
            goal_ids = {g.id for g in goal_candidates}
            overlap = [ll for ll in ego_candidates if ll.id in goal_ids]
            if overlap:
                ego_candidates = overlap
                self._warned_goal_off_lanelet = False
            elif not self._warned_goal_off_lanelet:
                if goal_candidates:
                    self.get_logger().warn(
                        f"Goal lanelet(s) {goal_ids} are not the ego lanelet(s) "
                        f"{[ll.id for ll in ego_candidates]}; multi-lanelet routing "
                        "is out of scope. Direction inference is undefined until "
                        "ego and goal are on the same lanelet."
                    )
                else:
                    self.get_logger().warn(
                        f"Goal pose ({gx:.2f}, {gy:.2f}) is not inside any "
                        "lanelet; direction inference is undefined until the "
                        "goal is moved onto a drivable lane."
                    )
                self._warned_goal_off_lanelet = True

        # Hysteresis: stick with the previously-active lanelet if it's still
        # in the candidate set, to avoid id flapping on lanelet boundaries.
        if self._last_active_id is not None:
            prev = next((ll for ll in ego_candidates if ll.id == self._last_active_id), None)
            if prev is not None:
                return prev
        return ego_candidates[0]

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

        # Direction inference: latch once per goal. Recomputing every tick
        # produces flapping as the truck approaches the goal (s_goal - s_ego
        # crosses zero) or when its projected segment on the active lanelet
        # changes (tangent jumps). With one goal -> one direction, the bridge
        # commits to that direction and goal_reached_distance_m stops it.
        if self._goal_pose is None:
            # No goal: clear latch so the next goal gets a fresh decision,
            # and default to forward for subscribers' initial state.
            self._goal_drive_reverse = None
            drive_reverse = False
        else:
            if self._goal_drive_reverse is None:
                self._goal_drive_reverse = self._infer_drive_reverse(active)
                self.get_logger().info(
                    f"Drive direction latched: "
                    f"{'REVERSE' if self._goal_drive_reverse else 'FORWARD'}"
                )
            drive_reverse = self._goal_drive_reverse
        self._publish_drive_direction(drive_reverse)

    def _infer_drive_reverse(self, active_lanelet) -> bool:
        """Return True iff the truck should physically drive in REVERSE (gear
        reversed, vehicle moves opposite to its heading) to reach the goal.

        Logic: motion direction needed is "toward the goal along the lane";
        truck heading may agree or disagree with the lane's canonical
        direction at the ego. Combining the two:
          - goal_downstream = s_goal > s_ego (goal further along canonical)
          - heading_canonical_aligned = ego heading · canonical tangent at ego > 0
          - drive_reverse = goal_downstream XOR heading_canonical_aligned
        which is True iff motion direction != ego heading direction.
        """
        if self._goal_pose is None or self._ego_pose is None:
            return False
        ex, ey, eyaw = self._ego_pose
        gx, gy = self._goal_pose
        try:
            s_ego = self._arc_length_on(active_lanelet, ex, ey)
            s_goal = self._arc_length_on(active_lanelet, gx, gy)
            tangent = self._utilities.getLaneletAngle(
                active_lanelet, Point(x=ex, y=ey, z=0.0)
            )
        except Exception as e:
            self.get_logger().warn(f"Arc-length inference failed: {e}")
            return False
        goal_downstream = s_goal > s_ego
        heading_canonical_aligned = math.cos(eyaw - tangent) > 0.0
        return goal_downstream != heading_canonical_aligned

    def _arc_length_on(self, lanelet, x: float, y: float) -> float:
        """Project (x, y) onto `lanelet` and return its arc-length distance
        from the lanelet's canonical start."""
        arc = self._utilities.getArcCoordinates([lanelet], _pose_at(x, y))
        return float(arc.length)

    def _publish_drive(self, val: bool):
        msg = Bool()
        msg.data = bool(val)
        self.pub_drive.publish(msg)

    def _publish_drive_direction(self, reverse: bool):
        msg = Bool()
        msg.data = bool(reverse)
        self.pub_drive_direction.publish(msg)


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
