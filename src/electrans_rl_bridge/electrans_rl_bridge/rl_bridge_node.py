"""
RL bridge node.

Subscribes:
  /localization/kinematic_state        nav_msgs/Odometry
  /vehicle/status/steering_status      autoware_vehicle_msgs/SteeringReport
  /vehicle/trailer_state               autoware_vehicle_msgs/TrailerState
  /planning/lane_reference/centerline  nav_msgs/Path
  /planning/lane_reference/drive_enabled std_msgs/Bool

Publishes:
  /control/command/control_cmd         autoware_control_msgs/Control
  /control/command/gear_cmd            autoware_vehicle_msgs/GearCommand
  /rl_bridge/bev_image                 sensor_msgs/Image   [debug, 32x32 mono8]
  /rl_bridge/state_vector              std_msgs/Float32MultiArray [debug, len=8]

Loads a TD3 model once at startup. On a 10 Hz timer, packs the latest ROS state
into the e2e_rl env adapter, computes the observation via the training-time
pipeline, predicts an action, integrates steering_rate to a target steering angle,
and publishes the Control message. Gear is published at 1 Hz (DRIVE).
"""

from __future__ import annotations

import math
import os
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray

from autoware_control_msgs.msg import Control
from autoware_vehicle_msgs.msg import GearCommand, SteeringReport, TrailerState

from electrans_rl_bridge.ros_env_adapter import install_e2e_rl_on_path, ROSLineFollowingAdapter


def _quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


class RLBridgeNode(Node):
    def __init__(self):
        super().__init__("rl_bridge_node")

        # ----- params -----
        self.declare_parameter("e2e_rl_path", "/home/ben/Ben/Thesis/e2e_rl")
        self.declare_parameter("td3_model_path", "")
        self.declare_parameter("action_space", "fixed_speed")  # or 'variable_speed'
        self.declare_parameter("control_rate_hz", 10.0)
        self.declare_parameter("max_steering_rad", math.pi / 4.0)
        self.declare_parameter("default_velocity_mps", 5.0)
        # Ratio of training-truck length to actual vehicle length. The policy
        # was trained at semi-truck scale (~6 m trailer wheelbase). For an
        # AgileX-class 1/8 lab robot, set this to ~8 so the policy sees
        # training-scale obs. Bridge then divides the policy's velocity
        # output by world_scale before commanding the small vehicle.
        self.declare_parameter("world_scale", 1.0)

        self.e2e_rl_path = str(self.get_parameter("e2e_rl_path").value)
        self.model_path = str(self.get_parameter("td3_model_path").value)
        self.action_space = str(self.get_parameter("action_space").value)
        self.control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.max_steering = float(self.get_parameter("max_steering_rad").value)
        self.default_velocity = float(self.get_parameter("default_velocity_mps").value)
        self.world_scale = float(self.get_parameter("world_scale").value)

        if not self.model_path:
            raise RuntimeError("rl_bridge_node: td3_model_path parameter is required")

        # ----- import e2e_rl + load model -----
        install_e2e_rl_on_path(self.e2e_rl_path)
        # CNNFeatureExtractor must be importable in scope before TD3 is built
        # (it's referenced in the saved policy_kwargs).
        from Models.CNNFeatureExtractor import CNNFeatureExtractor  # noqa: F401
        from stable_baselines3 import TD3
        import pickle
        import torch

        # The adapter owns the gymnasium env that defines action/observation
        # spaces -- build it first so we can construct a fresh TD3 directly
        # against those spaces, bypassing the SB3 .zip cloudpickle path (which
        # is fragile across numpy major versions). The portable .pth is produced
        # by scripts/re_export_td3.py.
        meta_path = os.path.splitext(self.model_path)[0] + ".policy_kwargs.pkl"
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        # Env class + kwargs are recorded into the meta at re-export time so we
        # can swap models (BEV / state / lidar+state) without touching code.
        # Old checkpoints that don't carry this metadata fall back to BEV.
        env_class_module = meta.get("env_class_module", "Environments.LineFollowing")
        env_class_name = meta.get("env_class_name", "BevObservationLineFollowingEnv")
        env_kwargs = dict(meta.get("env_kwargs", {}))
        env_kwargs.setdefault("fixed_speed", self.action_space == "fixed_speed")
        self.get_logger().info(
            f"Instantiating env {env_class_module}.{env_class_name} kwargs={env_kwargs}"
        )
        self.adapter = ROSLineFollowingAdapter(
            env_class_module=env_class_module,
            env_class_name=env_class_name,
            env_kwargs=env_kwargs,
            world_scale=self.world_scale,
        )

        # MultiInputPolicy works for Dict obs (BEV); for flat Box obs use MlpPolicy.
        from gymnasium import spaces
        if isinstance(self.adapter.env.observation_space, spaces.Dict):
            sb3_policy_class = "MultiInputPolicy"
        else:
            sb3_policy_class = "MlpPolicy"

        self.get_logger().info(f"Loading TD3 policy state from {self.model_path}")
        self.model = TD3(
            policy=sb3_policy_class,
            env=self.adapter.env,
            policy_kwargs=meta["policy_kwargs"],
            buffer_size=1,
            device="auto",
        )
        state_dict = torch.load(self.model_path, map_location=self.model.device)
        self.model.policy.load_state_dict(state_dict)

        # ----- state caches -----
        self._ego: Optional[tuple] = None        # (x, y, yaw, xd)
        self._steering: float = 0.0              # measured tire angle (rad)
        self._hitch_angle: float = 0.0           # rad
        self._drive_enabled: bool = False
        self._target_steering: float = 0.0       # integrated from action[0]
        self._dt = 1.0 / self.control_rate_hz

        # ----- pub / sub -----
        self.pub_control = self.create_publisher(Control, "/control/command/control_cmd", 1)
        self.pub_gear = self.create_publisher(GearCommand, "/control/command/gear_cmd", 1)
        self.pub_bev = self.create_publisher(Image, "/rl_bridge/bev_image", 1)
        self.pub_vec = self.create_publisher(Float32MultiArray, "/rl_bridge/state_vector", 1)

        self.create_subscription(Odometry, "/localization/kinematic_state", self._on_odom, 10)
        self.create_subscription(SteeringReport, "/vehicle/status/steering_status", self._on_steering, 10)
        # Simulator's sim_model_delay_steer_acc_geared_trailer publishes here;
        # autoware_trailer_state_visualizer reads the same topic. The real
        # vehicle bridge (e.g. canbridge) publishes the same topic name.
        self.create_subscription(TrailerState, "/vehicle/trailer_state", self._on_trailer, 10)
        self.create_subscription(Path, "/planning/lane_reference/centerline", self._on_centerline, 1)
        self.create_subscription(Bool, "/planning/lane_reference/drive_enabled", self._on_drive, 1)

        self.create_timer(self._dt, self._on_control_tick)
        self.create_timer(1.0, self._on_gear_tick)

        self.get_logger().info(
            f"RL bridge up — action_space={self.action_space}, rate={self.control_rate_hz} Hz, "
            f"world_scale={self.world_scale}"
        )

    # --------------------------------------------------------------- inputs
    def _on_odom(self, msg: Odometry):
        p = msg.pose.pose
        yaw = _quat_to_yaw(p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w)
        xd = msg.twist.twist.linear.x
        self._ego = (p.position.x, p.position.y, yaw, xd)

    def _on_steering(self, msg: SteeringReport):
        self._steering = float(msg.steering_tire_angle)

    def _on_trailer(self, msg: TrailerState):
        self._hitch_angle = float(msg.hitch_angle)

    def _on_centerline(self, msg: Path):
        if len(msg.poses) < 2:
            return
        xs = np.fromiter((ps.pose.position.x for ps in msg.poses), dtype=np.float32, count=len(msg.poses))
        ys = np.fromiter((ps.pose.position.y for ps in msg.poses), dtype=np.float32, count=len(msg.poses))
        try:
            self.adapter.set_reference_path(xs, ys)
        except Exception as e:
            self.get_logger().warn(f"set_reference_path failed: {e}")

    def _on_drive(self, msg: Bool):
        self._drive_enabled = bool(msg.data)

    # --------------------------------------------------------------- tick
    def _on_control_tick(self):
        if self._ego is None or not self.adapter.has_path():
            return

        x, y, yaw, xd = self._ego
        # Seed target steering with the measured tire angle on first tick so we
        # don't slam the actuator on startup.
        if abs(self._target_steering) < 1e-6 and abs(self._steering) > 1e-6:
            self._target_steering = self._steering

        self.adapter.set_ego_state(x, y, yaw, self._target_steering, xd)
        self.adapter.set_trailer_state_from_hitch(self._hitch_angle)

        try:
            obs = self.adapter.get_observation()
        except Exception as e:
            self.get_logger().warn(f"observation failed: {e}")
            return

        action, _ = self.model.predict(obs, deterministic=True)
        action = np.asarray(action).flatten()

        steering_rate = float(action[0])
        if self.action_space == "variable_speed" and action.size >= 2:
            velocity_cmd = float(action[1])
        else:
            velocity_cmd = self.default_velocity

        if self._drive_enabled:
            self._target_steering = float(
                np.clip(self._target_steering + steering_rate * self._dt, -self.max_steering, self.max_steering)
            )
        else:
            # No goal yet -- don't pre-commit the steering. Track the measured
            # tire angle so the first commanded angle on drive-enable is the
            # actual current angle, avoiding a step change.
            self._target_steering = float(self._steering)
            velocity_cmd = 0.0

        # The policy's velocity is in env-space (training-scale truck);
        # divide by world_scale so the small vehicle moves at the
        # physically-corresponding speed.
        velocity_cmd = velocity_cmd / self.world_scale

        # The autoware DELAY_STEER_ACC_GEARED* sim vehicle models (and the real
        # vehicle's underlying acc-tracking loop) read .longitudinal.acceleration
        # and ignore .longitudinal.velocity, so we need a closed-loop accel
        # signal that tracks the target velocity. Simple P controller; bridge
        # runs at control_rate_hz so this stays stable.
        current_v = self._ego[3] if self._ego is not None else 0.0
        kp = 1.0  # accel gain in 1/s; tuned so 1 m/s error => 1 m/s^2 accel
        accel_lim = 2.0  # m/s^2, comfortable
        accel_cmd = float(np.clip(kp * (velocity_cmd - current_v), -accel_lim, accel_lim))

        # ---- publish Control ----
        ctl = Control()
        ctl.stamp = self.get_clock().now().to_msg()
        ctl.lateral.stamp = ctl.stamp
        ctl.lateral.steering_tire_angle = float(self._target_steering)
        ctl.lateral.steering_tire_rotation_rate = float(steering_rate)
        ctl.lateral.is_defined_steering_tire_rotation_rate = True
        ctl.longitudinal.stamp = ctl.stamp
        ctl.longitudinal.velocity = float(velocity_cmd)
        ctl.longitudinal.acceleration = accel_cmd
        ctl.longitudinal.is_defined_acceleration = True
        self.pub_control.publish(ctl)

        # ---- debug publishes ----
        # Always publish the full observation as a flat float vector on
        # /rl_bridge/state_vector. For BEV (dict) obs, additionally publish
        # the 32x32 image on /rl_bridge/bev_image. Lidar+state and state-only
        # obs skip the image publisher (obs is already flat).
        if isinstance(obs, dict):
            vec = obs["vector"].astype(np.float32).flatten()
            img = obs.get("image")
        else:
            vec = np.asarray(obs).astype(np.float32).flatten()
            img = None
        vmsg = Float32MultiArray()
        vmsg.data = vec.tolist()
        self.pub_vec.publish(vmsg)

        if img is not None:
            img_msg = Image()
            img_msg.header.stamp = ctl.stamp
            img_msg.header.frame_id = "base_link"
            img_msg.height = int(img.shape[0])
            img_msg.width = int(img.shape[1])
            img_msg.encoding = "mono8"
            img_msg.is_bigendian = 0
            img_msg.step = int(img.shape[1])
            img_msg.data = img.reshape(-1).tobytes()
            self.pub_bev.publish(img_msg)

    def _on_gear_tick(self):
        msg = GearCommand()
        msg.stamp = self.get_clock().now().to_msg()
        msg.command = GearCommand.DRIVE
        self.pub_gear.publish(msg)


def main():
    rclpy.init()
    node = RLBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
