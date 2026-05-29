"""
RL bridge node.

Subscribes:
  /localization/kinematic_state        nav_msgs/Odometry
  /vehicle/status/steering_status      autoware_vehicle_msgs/SteeringReport
  /vehicle/trailer_state               autoware_vehicle_msgs/TrailerState
  /planning/lane_reference/centerline    nav_msgs/Path
  /planning/lane_reference/drive_enabled std_msgs/Bool
  /planning/lane_reference/drive_direction std_msgs/Bool  [True = reverse]

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
        # Optional reverse-trained checkpoint. If empty, the bridge runs in
        # forward-only mode and ignores /planning/lane_reference/drive_direction.
        # If set, the bridge loads BOTH policies at startup and picks each
        # tick based on the drive_direction topic.
        self.declare_parameter("td3_reverse_model_path", "")
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
        self.reverse_model_path = str(self.get_parameter("td3_reverse_model_path").value)
        self.action_space = str(self.get_parameter("action_space").value)
        self.control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.max_steering = float(self.get_parameter("max_steering_rad").value)
        self.default_velocity = float(self.get_parameter("default_velocity_mps").value)
        self.world_scale = float(self.get_parameter("world_scale").value)

        if not self.model_path:
            raise RuntimeError("rl_bridge_node: td3_model_path parameter is required")

        # ----- import e2e_rl + load model -----
        install_e2e_rl_on_path(self.e2e_rl_path)

        # Override e2e_rl config to use AgileX lab measurements BEFORE
        # importing env classes. Environments.TractorTrailer reads these at
        # module load (WINDOW_WIDTH = config.window_width_px etc.), and the
        # remaining lane/BEV knobs are read at runtime via getattr(config, …)
        # — so this assignment is enough to drive both rendering and the
        # occupancy grid / lidar at the real lab scale instead of the
        # training-time semi-truck scale.
        from e2erl_utils import config as e2erl_config
        # Pygame world: 25 m × 20 m at 0.05 m/px = 500 × 400 px. Bounding box
        # of the MVSL map plus ~5 m margin on each side.
        e2erl_config.window_width_px = 500
        e2erl_config.window_height_px = 400
        e2erl_config.meters_per_pixel = 0.05
        # Vehicle rendering: actual AgileX lab measurements. TRAILER_LENGTH
        # is conflated with kinematic wheelbase in the env, so we use the
        # wheelbase value (2.0 m).
        e2erl_config.tractor_length_m = 1.0
        e2erl_config.tractor_width_m = 0.65
        e2erl_config.trailer_length_m = 2.0
        e2erl_config.trailer_width_m = 0.5
        # Lane corridor: LL7's bounds span y∈[-1.81, +1.0] = 2.81 m wide,
        # so half-width 1.41 m. Small shoulder so lidar starts hitting the
        # boundary just past the painted edge.
        e2erl_config.lane_centerline_half_width_m = 1.41
        e2erl_config.lane_shoulder_m = 0.20
        e2erl_config.grid_res_m = 0.05
        e2erl_config.lane_sample_ds_m = 0.10
        # BEV crop layout for the AgileX lab. World canvas is rotated so the
        # truck's heading points "up" in the BEV (top of frame = ahead of
        # truck, bottom = behind = trailer side). Anchor=center centres the
        # crop on (rear_axle + bev_offset_x_m). Positive offset pushes the
        # anchor (and hence the truck) DOWN in the BEV, freeing more pixels
        # for the lane ahead.
        #   - bev_offset_x_m=+1.0 puts the anchor 1 m AHEAD of the rear
        #     axle, so the rear axle ends up 1 m below BEV centre. Truck
        #     occupies the lower-centre of the frame, ~5 m of lane ahead
        #     are visible above it, and the trailer drops into the bottom
        #     ~2 m of the crop.
        #   - bev_obs_crop_m=8.0 gives ~8 m × 8 m visible world area.
        # This override is local to the bridge — training-time defaults in
        # e2e_rl/e2erl_utils/config.py remain unchanged.
        e2erl_config.bev_obs_crop_anchor_forward = "center"
        e2erl_config.bev_obs_crop_m = 8.0
        e2erl_config.bev_zoom_scale = 1.0
        e2erl_config.bev_offset_x_m = 1.0
        # Vehicle CG-to-axle distances: AgileX wheelbase 0.65 m split ~half.
        e2erl_config.tesla_model_s_vehicle_params = dict(
            e2erl_config.tesla_model_s_vehicle_params,
            lf=0.33, lr=0.32,
        )

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

        # Optional reverse policy. If the path is provided AND the file
        # exists, build a second TD3 with the same env-frame so we can swap
        # between forward and reverse per tick. Their policy_kwargs may
        # differ (e.g. different features_extractor_kwargs), so we load the
        # reverse meta independently.
        self.reverse_model = None
        if self.reverse_model_path and os.path.exists(self.reverse_model_path):
            reverse_meta_path = (
                os.path.splitext(self.reverse_model_path)[0] + ".policy_kwargs.pkl"
            )
            with open(reverse_meta_path, "rb") as f:
                reverse_meta = pickle.load(f)
            self.get_logger().info(
                f"Loading reverse TD3 policy state from {self.reverse_model_path}"
            )
            self.reverse_model = TD3(
                policy=sb3_policy_class,
                env=self.adapter.env,
                policy_kwargs=reverse_meta["policy_kwargs"],
                buffer_size=1,
                device="auto",
            )
            rev_state_dict = torch.load(
                self.reverse_model_path, map_location=self.reverse_model.device
            )
            self.reverse_model.policy.load_state_dict(rev_state_dict)
        elif self.reverse_model_path:
            self.get_logger().warn(
                f"td3_reverse_model_path is set to '{self.reverse_model_path}' but "
                "the file does not exist — running forward-only."
            )

        # ----- state caches -----
        self._ego: Optional[tuple] = None        # (x, y, yaw, xd)
        self._steering: float = 0.0              # measured tire angle (rad)
        self._hitch_angle: float = 0.0           # rad
        self._drive_enabled: bool = False
        # Set from /planning/lane_reference/drive_direction; chooses which
        # policy + adapter mode to run on each tick. Ignored when only the
        # forward checkpoint is loaded.
        self._drive_reverse: bool = False
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
        self.create_subscription(
            Bool, "/planning/lane_reference/drive_direction", self._on_drive_direction, 1
        )

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

    def _on_drive_direction(self, msg: Bool):
        new_reverse = bool(msg.data)
        if new_reverse != self._drive_reverse:
            self.get_logger().info(
                f"drive_direction changed → {'REVERSE' if new_reverse else 'FORWARD'}"
            )
        self._drive_reverse = new_reverse

    # --------------------------------------------------------------- tick
    def _on_control_tick(self):
        if self._ego is None or not self.adapter.has_path():
            return

        x, y, yaw, xd = self._ego
        # Seed target steering with the measured tire angle on first tick so we
        # don't slam the actuator on startup.
        if abs(self._target_steering) < 1e-6 and abs(self._steering) > 1e-6:
            self._target_steering = self._steering

        # Pick policy + adapter frame for this tick. Only honour drive_reverse
        # if we actually loaded a reverse checkpoint; otherwise stay forward.
        is_reverse = self._drive_reverse and self.reverse_model is not None
        self.adapter.set_reverse_mode(is_reverse)
        active_model = self.reverse_model if is_reverse else self.model

        self.adapter.set_ego_state(x, y, yaw, self._target_steering, xd)
        self.adapter.set_trailer_state_from_hitch(self._hitch_angle)

        try:
            obs = self.adapter.get_observation()
        except Exception as e:
            self.get_logger().warn(f"observation failed: {e}")
            return

        action, _ = active_model.predict(obs, deterministic=True)
        action = np.asarray(action).flatten()

        steering_rate = float(action[0])
        if self.action_space == "variable_speed" and action.size >= 2:
            # variable_speed policies emit the signed longitudinal velocity
            # directly; reverse-trained policies already produce vx<0, so we
            # trust the action.
            velocity_cmd = float(action[1])
        else:
            # fixed_speed policies don't emit a velocity; we apply the
            # configured magnitude with a sign flip in reverse mode.
            velocity_cmd = -self.default_velocity if is_reverse else self.default_velocity

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

        # The sim's set_input flips the acceleration sign in REVERSE gear:
        # combined_acc = -acc_by_cmd. simple_planning_simulator_core.cpp:637-638.
        # Then sim_model_delay_steer_acc_geared_trailer.cpp:211-213 forces
        # VX=0 when REVERSE-gear and VX>0. So if we sent the raw negative
        # accel_cmd in reverse, sim would flip it positive and immediately
        # zero the velocity every tick. Pre-flip here so the sim's flip
        # cancels and combined_acc keeps the sign the P-controller wanted.
        if is_reverse:
            accel_cmd = -accel_cmd

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
        # /rl_bridge/state_vector. Always publish the 32x32 BEV image on
        # /rl_bridge/bev_image regardless of the policy's obs pipeline -- the
        # adapter spins up a debug BEV renderer on construction so we can
        # visualise what the env sees even when the policy uses state-only or
        # lidar+state observations.
        if isinstance(obs, dict):
            vec = obs["vector"].astype(np.float32).flatten()
        else:
            vec = np.asarray(obs).astype(np.float32).flatten()
        vmsg = Float32MultiArray()
        vmsg.data = vec.tolist()
        self.pub_vec.publish(vmsg)

        img = self.adapter.get_debug_bev_image()

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
        # The sim's DELAY_STEER_ACC_GEARED_TRAILER vehicle model uses the gear
        # command to decide the sign convention on acceleration; the real
        # vehicle's gear interface follows the same convention. Mirror the
        # active drive direction so both stay consistent.
        is_reverse = self._drive_reverse and self.reverse_model is not None
        msg = GearCommand()
        msg.stamp = self.get_clock().now().to_msg()
        msg.command = GearCommand.REVERSE if is_reverse else GearCommand.DRIVE
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
