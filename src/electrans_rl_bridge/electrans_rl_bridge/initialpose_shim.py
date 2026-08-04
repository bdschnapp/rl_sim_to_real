"""Bridge RViz's /initialpose topic to autoware's pose-initializer service.

RViz's "2D Pose Estimate" tool publishes a PoseWithCovarianceStamped on
/initialpose. autoware's pose_initializer doesn't subscribe — it exposes a
service. Upstream, two nodes handled this: autoware_adapi_adaptors'
InitialPoseAdaptor (/initialpose -> /api/localization/initialize) and
autoware_default_adapi (serves /api/localization/initialize, forwards to the
core /localization/initialize). Both were removed when the lifecycle/ADAPI
stack was deleted from autoware.launch.xml.

This node restores the operator workflow without reintroducing that stack, and
it now REPLICATES exactly what InitialPoseAdaptor::on_initial_pose did (verified
against the upstream source), which is the piece that was silently lost:

  1. HEIGHT-FIT: snap the clicked pose's z onto the pcd-map ground via the
     map_height_fitter service (the adaptor's `fitter_.fit(position, frame_id)`).
     RViz hands us z=0, but the lab pcd ground spans z=-0.35..1.5 m, so an
     un-fitted seed starts ndt_align off-ground and it fails to converge. The
     pose_initializer's OWN fitter only runs for GNSS init (gnss_module), never
     for a user/RViz pose — so this fit has to happen here.
  2. COVARIANCE: overwrite with the RViz particle covariance (the adaptor's
     `initial_pose_particle_covariance`: diag 4, 4, 0.01, 0.01, 0.01, 1.0).
  3. INITIALIZE: call the CORE /localization/initialize with method=AUTO. AUTO
     runs ndt_align seeded by the (now height-fitted) pose — the same path the
     working full-Autoware stack uses. (The previous version used DIRECT + a
     manual NDT trigger as a workaround for an apparent ndt_align deadlock; that
     deadlock was a symptom of the missing height-fit / bad seed, not a real
     chicken-and-egg, so we go back to the upstream AUTO flow.)

The height-fit degrades gracefully: if /map/map_height_fitter/service isn't up
yet we forward the un-fitted pose rather than dropping it.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
from autoware_localization_msgs.srv import InitializeLocalization
from autoware_internal_localization_msgs.srv import (
    PoseWithCovarianceStamped as FitService,
)

INIT_SERVICE = "/localization/initialize"
FIT_SERVICE = "/map/map_height_fitter/service"

# Initial-pose covariance handed to /localization/initialize. ndt_align derives
# the TPE particle SEARCH spread from sqrt(cov) per axis:
#   x,y = 0.25 -> std 0.5 m  -> search ~±1 m   (trust the RViz click to ~0.5 m)
#   yaw = 0.12 -> std ~20 deg -> search ~±30-40 deg around the clicked heading
# This is the tightened search. The yaw term only bites because the TPE was
# patched to sample yaw ~ normal(initial_yaw, std) instead of uniform(-pi,pi)
# (autoware_localization_util/tree_structured_parzen_estimator.cpp +
# ndt_scan_matcher_core.cpp passing a 6th yaw element).
#
# Two variants, selected by the `localization_mode` ROS param (plumbed from the
# map folder's map_metadata.yaml via start_robot.sh -> launch chain):
#   2d — walls-only slab map, flat floor: height-fit + flat floor pin
#        z/roll/pitch, keep them tight (0.01 -> std 0.1 m / 0.1 rad).
#   3d — full map with real elevation (ramps): the height-fit is a downhill-
#        biased "lowest point within ~1 m" on slopes and the floor is NOT flat,
#        so give the TPE room to search z (0.25 -> std 0.5 m) and roll/pitch
#        (0.0025 -> std 0.05 rad ~ 3 deg, covers the ~2-3 deg real grades).
RVIZ_PARTICLE_COVARIANCE_2D = [
    0.25, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.25, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.01, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.01, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.01, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.12,
]
RVIZ_PARTICLE_COVARIANCE_3D = [
    0.25, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.25, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.25, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0025, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0025, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.12,
]


class InitialPoseShim(Node):
    def __init__(self):
        super().__init__("initialpose_shim")
        mode = self.declare_parameter("localization_mode", "2d").value
        if mode not in ("2d", "3d"):
            self.get_logger().error(f"localization_mode '{mode}' invalid; using 2d")
            mode = "2d"
        self.mode = mode
        self.covariance = (
            RVIZ_PARTICLE_COVARIANCE_3D if mode == "3d" else RVIZ_PARTICLE_COVARIANCE_2D
        )
        self.cli = self.create_client(InitializeLocalization, INIT_SERVICE)
        self.fit = self.create_client(FitService, FIT_SERVICE)
        self.create_subscription(PoseWithCovarianceStamped, "/initialpose", self._on_pose, 1)
        self.get_logger().info(
            f"initialpose_shim ready ({mode} mode): /initialpose → height-fit "
            f"({FIT_SERVICE}) → {INIT_SERVICE} (AUTO)"
        )

    def _on_pose(self, msg: PoseWithCovarianceStamped):
        # 1. Height-fit the clicked pose onto the map ground (adaptor step 1).
        if self.fit.wait_for_service(timeout_sec=1.0):
            req = FitService.Request()
            req.pose_with_covariance = msg
            fut = self.fit.call_async(req)
            fut.add_done_callback(lambda f: self._after_fit(f, msg))
        else:
            self.get_logger().warn(
                f"{FIT_SERVICE} unavailable; initializing with un-fitted z"
            )
            self._initialize(msg)

    def _after_fit(self, fut, original: PoseWithCovarianceStamped):
        pose = original
        try:
            res = fut.result()
            if res is not None and res.success:
                pose = res.pose_with_covariance
                z = pose.pose.pose.position.z
                self.get_logger().info(f"height-fit OK (z → {z:.3f} m)")
            else:
                self.get_logger().warn("height-fit failed; using un-fitted z")
        except Exception as e:
            self.get_logger().warn(f"height-fit exception ({e}); using un-fitted z")
        self._initialize(pose)

    def _initialize(self, pose: PoseWithCovarianceStamped):
        if not self.cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn(f"{INIT_SERVICE} unavailable; dropping pose")
            return
        # 2. Overwrite covariance with the RViz particle covariance (adaptor step 2).
        pose.pose.covariance = self.covariance
        req = InitializeLocalization.Request()
        req.pose_with_covariance = [pose]
        # 3. AUTO → ndt_align seeded by the height-fitted pose (adaptor step 3).
        req.method = InitializeLocalization.Request.AUTO
        fut = self.cli.call_async(req)
        fut.add_done_callback(self._on_response)

    def _on_response(self, fut):
        try:
            res = fut.result()
            if res.status.success:
                self.get_logger().info("pose initialized OK (AUTO / ndt_align)")
            else:
                self.get_logger().warn(f"pose initialize failed: {res.status.message}")
        except Exception as e:
            self.get_logger().warn(f"service call exception: {e}")


def main():
    rclpy.init()
    rclpy.spin(InitialPoseShim())
    rclpy.shutdown()
