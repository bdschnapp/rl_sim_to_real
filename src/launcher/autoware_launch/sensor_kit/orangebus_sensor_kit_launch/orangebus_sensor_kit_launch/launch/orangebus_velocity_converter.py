#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistWithCovarianceStamped, TransformStamped
from tf2_ros import TransformBroadcaster
from builtin_interfaces.msg import Time

class VelocityAndTfNode(Node):
    def __init__(self):
        super().__init__('orangebus_velocity_converter')
        self.vel_pub = self.create_publisher(
            TwistWithCovarianceStamped,
            '/sensing/vehicle_velocity_converter/twist_with_covariance',
            1
        )
        self.tf_broadcaster = TransformBroadcaster(self)
        self.create_subscription(
            Odometry,
            '/localization/kinematic_state',
            self.odometry_callback,
            1
        )
        self.get_logger().info('Velocity/TF converter node started.')

    def odometry_callback(self, msg: Odometry):
        # 1️⃣ Publish TwistWithCovarianceStamped using same header
        t = TwistWithCovarianceStamped()
        t.header = msg.header
        t.twist = msg.twist  # includes covariance
        self.vel_pub.publish(t)

        # 2️⃣ Broadcast map -> base_link transform
        tf = TransformStamped()
        tf.header = msg.header
        # tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = msg.header.frame_id  # typically "map"
        tf.child_frame_id = msg.child_frame_id    # typically "base_link"
        tf.transform.translation.x = msg.pose.pose.position.x
        tf.transform.translation.y = msg.pose.pose.position.y
        tf.transform.translation.z = msg.pose.pose.position.z
        tf.transform.rotation = msg.pose.pose.orientation
        self.tf_broadcaster.sendTransform(tf)

def main(args=None):
    rclpy.init(args=args)
    node = VelocityAndTfNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('Shutting down.')
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()