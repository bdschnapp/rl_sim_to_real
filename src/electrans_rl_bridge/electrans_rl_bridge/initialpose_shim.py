"""Bridge RViz's /initialpose topic to autoware's pose-initializer service.

RViz's "2D Pose Estimate" tool publishes a PoseWithCovarianceStamped on
/initialpose. autoware's pose_initializer doesn't subscribe — it exposes a
service. Upstream, two nodes handled this: autoware_adapi_adaptors'
InitialPoseAdaptor (/initialpose -> /api/localization/initialize) and
autoware_default_adapi (serves /api/localization/initialize, forwards to the
core /localization/initialize). Both were removed when the lifecycle/ADAPI
stack was deleted from autoware.launch.xml.

This node restores the operator workflow without reintroducing that stack: it
subscribes to /initialpose and calls the CORE service directly —
/localization/initialize (autoware_localization_msgs/srv/InitializeLocalization),
which is what the pose_initializer node actually serves. The previous version
called /api/localization/initialize (autoware_adapi_v1_msgs), but nothing serves
that name once default_adapi is gone, so every pose was silently dropped at
wait_for_service.

method=AUTO runs NDT alignment using the provided pose as the initial guess
(real scan matching), as opposed to DIRECT which would trust the pose blindly.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
from autoware_localization_msgs.srv import InitializeLocalization

SERVICE_NAME = "/localization/initialize"


class InitialPoseShim(Node):
    def __init__(self):
        super().__init__("initialpose_shim")
        self.cli = self.create_client(InitializeLocalization, SERVICE_NAME)
        self.create_subscription(PoseWithCovarianceStamped, "/initialpose", self._on_pose, 1)
        self.get_logger().info(f"initialpose_shim ready: /initialpose → {SERVICE_NAME}")

    def _on_pose(self, msg: PoseWithCovarianceStamped):
        if not self.cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn(f"{SERVICE_NAME} unavailable; dropping pose")
            return
        req = InitializeLocalization.Request()
        req.pose_with_covariance = [msg]
        req.method = InitializeLocalization.Request.AUTO  # NDT alignment from this guess
        fut = self.cli.call_async(req)
        fut.add_done_callback(self._on_response)

    def _on_response(self, fut):
        try:
            res = fut.result()
            if res.status.success:
                self.get_logger().info("pose initialized OK")
            else:
                self.get_logger().warn(f"pose initialize failed: {res.status.message}")
        except Exception as e:
            self.get_logger().warn(f"service call exception: {e}")


def main():
    rclpy.init()
    rclpy.spin(InitialPoseShim())
    rclpy.shutdown()
