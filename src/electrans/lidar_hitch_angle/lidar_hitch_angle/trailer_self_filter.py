import array
import math

import numpy as np
import rclpy
from autoware_vehicle_msgs.msg import TrailerState
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool


class TrailerSelfFilter(Node):
    def __init__(self):
        super().__init__("trailer_self_filter")

        self.declare_parameter("enabled", True)
        self.declare_parameter("expected_frame", "base_link")
        self.declare_parameter("trailer_state_timeout_s", 1.0)
        self.declare_parameter("require_connected_topic", True)
        self.declare_parameter("connected_timeout_s", 1.0)

        # Geometry matched to truck_trailer_freespace_planner.param.yaml.
        self.declare_parameter("tractor_base2hinge", 0.20)
        self.declare_parameter("trailer_front_oh", 0.39)
        self.declare_parameter("trailer_wheel_base", 2.75)
        self.declare_parameter("trailer_rear_oh", 0.51)
        self.declare_parameter("trailer_width", 0.62)
        self.declare_parameter("trailer_min_z", -0.30)
        self.declare_parameter("trailer_max_z", 1.30)
        self.declare_parameter("filter_margin_m", 0.05)

        self.latest_trailer_state = None
        self.last_trailer_state_time = None
        self.latest_connected = None
        self.last_connected_time = None

        self.create_subscription(
            TrailerState, "~/input/trailer_state", self.on_trailer_state, 10)
        self.create_subscription(
            Bool, "~/input/is_trailer_connected", self.on_is_connected, 10)
        self.pointcloud_sub = self.create_subscription(
            PointCloud2, "~/input/pointcloud", self.on_pointcloud, qos_profile_sensor_data)
        self.pointcloud_pub = self.create_publisher(
            PointCloud2, "~/output/pointcloud", qos_profile_sensor_data)

    def on_trailer_state(self, msg: TrailerState) -> None:
        self.latest_trailer_state = msg
        self.last_trailer_state_time = self.get_clock().now()

    def on_is_connected(self, msg: Bool) -> None:
        self.latest_connected = bool(msg.data)
        self.last_connected_time = self.get_clock().now()

    def on_pointcloud(self, msg: PointCloud2) -> None:
        if not bool(self.get_parameter("enabled").value):
            self.pointcloud_pub.publish(msg)
            return

        if not self._trailer_state_is_fresh():
            self.pointcloud_pub.publish(msg)
            return

        if not self._connected_state_allows_filtering():
            self.pointcloud_pub.publish(msg)
            return

        expected_frame = str(self.get_parameter("expected_frame").value)
        if msg.header.frame_id != expected_frame:
            self.get_logger().warn(
                f"Trailer self filter expects {expected_frame}, got "
                f"{msg.header.frame_id}; passing pointcloud through.",
                throttle_duration_sec=1.0,
            )
            self.pointcloud_pub.publish(msg)
            return

        filtered_msg, removed_count = self._remove_trailer_points(msg)
        self.get_logger().debug(
            f"Removed {removed_count} trailer points from {msg.width * msg.height} input points")
        self.pointcloud_pub.publish(filtered_msg)

    def _trailer_state_is_fresh(self) -> bool:
        if self.latest_trailer_state is None or self.last_trailer_state_time is None:
            return False

        timeout_s = float(self.get_parameter("trailer_state_timeout_s").value)
        age_s = (self.get_clock().now() - self.last_trailer_state_time).nanoseconds * 1e-9
        return age_s <= timeout_s

    def _connected_state_allows_filtering(self) -> bool:
        if not bool(self.get_parameter("require_connected_topic").value):
            return True

        if self.latest_connected is None or self.last_connected_time is None:
            return False

        timeout_s = float(self.get_parameter("connected_timeout_s").value)
        age_s = (self.get_clock().now() - self.last_connected_time).nanoseconds * 1e-9
        return self.latest_connected and age_s <= timeout_s

    def _remove_trailer_points(self, cloud_msg: PointCloud2) -> tuple[PointCloud2, int]:
        points = point_cloud2.read_points(cloud_msg)
        if len(points) == 0:
            return cloud_msg, 0

        x = points["x"].astype(np.float64, copy=False)
        y = points["y"].astype(np.float64, copy=False)
        z = points["z"].astype(np.float64, copy=False)

        hitch_angle = float(self.latest_trailer_state.hitch_angle)
        trailer_yaw = -hitch_angle

        tractor_base2hinge = float(self.get_parameter("tractor_base2hinge").value)
        front = float(self.get_parameter("trailer_front_oh").value)
        rear = (
            float(self.get_parameter("trailer_wheel_base").value) +
            float(self.get_parameter("trailer_rear_oh").value)
        )
        half_width = 0.5 * float(self.get_parameter("trailer_width").value)
        min_z = float(self.get_parameter("trailer_min_z").value)
        max_z = float(self.get_parameter("trailer_max_z").value)
        margin = float(self.get_parameter("filter_margin_m").value)

        hitch_x = -tractor_base2hinge
        hitch_y = 0.0
        dx = x - hitch_x
        dy = y - hitch_y

        cos_yaw = math.cos(trailer_yaw)
        sin_yaw = math.sin(trailer_yaw)
        longitudinal = dx * cos_yaw + dy * sin_yaw
        lateral = -dx * sin_yaw + dy * cos_yaw

        finite = np.isfinite(longitudinal) & np.isfinite(lateral) & np.isfinite(z)
        inside_trailer = (
            finite &
            (longitudinal >= -(rear + margin)) &
            (longitudinal <= front + margin) &
            (np.abs(lateral) <= half_width + margin) &
            (z >= min_z - margin) &
            (z <= max_z + margin)
        )
        keep = ~inside_trailer

        if np.all(keep):
            return cloud_msg, 0

        filtered_points = points[keep].copy()
        output = self._create_cloud_like(cloud_msg, filtered_points)
        return output, int(np.count_nonzero(inside_trailer))

    @staticmethod
    def _create_cloud_like(template: PointCloud2, points: np.ndarray) -> PointCloud2:
        output = PointCloud2()
        output.header = template.header
        output.height = 1
        output.width = int(points.shape[0])
        output.fields = template.fields
        output.is_bigendian = template.is_bigendian
        output.point_step = template.point_step
        output.row_step = output.point_step * output.width
        output.is_dense = template.is_dense

        contiguous = np.ascontiguousarray(points)
        payload = array.array("B")
        payload.frombytes(memoryview(contiguous).cast("B"))
        output.data = payload
        return output


def main(args=None):
    rclpy.init(args=args)
    node = TrailerSelfFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
