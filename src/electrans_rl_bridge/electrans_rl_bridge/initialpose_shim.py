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

We use method=DIRECT (trust the provided pose) rather than AUTO, then activate
the ONGOING ndt_scan_matcher ourselves. AUTO would make the pose_initializer
call the blocking ndt_align service, which deadlocks on this robot: ndt_align
needs the pointcloud map, NDT's dynamic map loader only loads it around the live
EKF pose, but the pose_initializer deactivates the EKF during init -> no pose ->
no map -> ndt_align hangs forever. DIRECT activates the EKF immediately with the
user pose; we then call /localization/pose_estimator/trigger_node to start the
ongoing NDT, which (with the EKF now live) loads the map around the pose and
corrects continuously. So NDT still runs map-relative correction — just without
the init-time deadlock.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
from autoware_localization_msgs.srv import InitializeLocalization
from std_srvs.srv import SetBool

SERVICE_NAME = "/localization/initialize"
NDT_TRIGGER_SERVICE = "/localization/pose_estimator/trigger_node"


class InitialPoseShim(Node):
    def __init__(self):
        super().__init__("initialpose_shim")
        self.cli = self.create_client(InitializeLocalization, SERVICE_NAME)
        self.ndt_trig = self.create_client(SetBool, NDT_TRIGGER_SERVICE)
        self.create_subscription(PoseWithCovarianceStamped, "/initialpose", self._on_pose, 1)
        self.get_logger().info(f"initialpose_shim ready: /initialpose → {SERVICE_NAME}")

    def _on_pose(self, msg: PoseWithCovarianceStamped):
        if not self.cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn(f"{SERVICE_NAME} unavailable; dropping pose")
            return
        req = InitializeLocalization.Request()
        req.pose_with_covariance = [msg]
        # DIRECT (trust the pose) — NOT AUTO. AUTO calls the blocking ndt_align,
        # which deadlocks here (needs map ← needs live EKF ← needs ndt_align).
        req.method = InitializeLocalization.Request.DIRECT
        fut = self.cli.call_async(req)
        fut.add_done_callback(self._on_response)

    def _on_response(self, fut):
        try:
            res = fut.result()
            if res.status.success:
                self.get_logger().info("pose initialized OK (DIRECT) → activating NDT")
                self._activate_ndt()
            else:
                self.get_logger().warn(f"pose initialize failed: {res.status.message}")
        except Exception as e:
            self.get_logger().warn(f"service call exception: {e}")

    def _activate_ndt(self):
        """Activate the ONGOING ndt_scan_matcher. With the EKF now live (DIRECT
        init), NDT's dynamic map loader gets a pose, loads the map, and corrects
        continuously — no blocking ndt_align, so no init deadlock."""
        if not self.ndt_trig.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn(f"{NDT_TRIGGER_SERVICE} unavailable; NDT not activated")
            return
        fut = self.ndt_trig.call_async(SetBool.Request(data=True))
        fut.add_done_callback(
            lambda f: self.get_logger().info("NDT scan matcher activated")
            if (f.result() and f.result().success)
            else self.get_logger().warn("NDT activation call returned failure")
        )


def main():
    rclpy.init()
    rclpy.spin(InitialPoseShim())
    rclpy.shutdown()
