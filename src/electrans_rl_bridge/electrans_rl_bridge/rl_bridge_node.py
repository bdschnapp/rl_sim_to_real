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

import faulthandler
import math
import os
import sys

faulthandler.enable()
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
        # Trailer attached? Selects which model PAIR the bridge loads and, with
        # it, the observation space — the tractor-only models were trained
        # against the TractorOnly env (5-dim state, 29-D lidar_24 obs) while the
        # trailer models use the 8-dim state (32-D). The obs space itself is
        # read from each checkpoint's .policy_kwargs.pkl meta (env_class_*), so
        # all this flag does is pick the right pair of checkpoints + assert the
        # loaded model actually matches (guards against a trailer model being
        # deployed with trailer:=false and vice-versa). Default true = trailer
        # models (back-compat with existing deployments).
        self.declare_parameter("trailer", True)
        # Tractor-only checkpoints, used when trailer:=false. Separate params so
        # both pairs can be configured once and switched by the flag alone.
        self.declare_parameter("td3_model_path_tractor_only", "")
        self.declare_parameter("td3_reverse_model_path_tractor_only", "")
        # action_space:
        #   fixed_speed    → 1-D action [steer_rate]; constant velocity.
        #   variable_speed → 2-D action [steer_rate, velocity].
        #   stop_signal    → 2-D action [steer_rate, stop_signal]; constant
        #                    velocity, but the policy can decide to STOP via the
        #                    second dim (the bridge zeroes velocity when it
        #                    exceeds stop_threshold). The stop is LEARNED by the
        #                    model — the bridge just listens for it.
        self.declare_parameter("action_space", "fixed_speed")
        # stop_signal threshold: action[1] > this ⇒ STOP (velocity → 0). Matches
        # the training-time default (stop_signal_patch threshold 0.0).
        self.declare_parameter("stop_threshold", 0.5)
        self.declare_parameter("control_rate_hz", 10.0)
        # The debug BEV image (/rl_bridge/bev_image, relayed into RViz) is for
        # human verification, not control, so it renders on its own slower
        # timer instead of every control tick. Rendering + occupancy build for
        # the debug env is the single most expensive thing the bridge does;
        # decoupling it keeps the 10 Hz control loop cheap on the Jetson. Set
        # 0 to disable BEV publishing entirely.
        self.declare_parameter("bev_publish_rate_hz", 4.0)
        self.declare_parameter("max_steering_rad", math.pi / 4.0)
        self.declare_parameter("default_velocity_mps", 5.0)
        # Ratio of training-truck length to actual vehicle length. The policy
        # was trained at semi-truck scale (~6 m trailer wheelbase). For an
        # AgileX-class 1/8 lab robot, set this to ~8 so the policy sees
        # training-scale obs. Bridge then divides the policy's velocity
        # output by world_scale before commanding the small vehicle.
        self.declare_parameter("world_scale", 1.0)
        # Deployment-side velocity clamp applied AFTER the policy's chosen
        # velocity (variable-speed only). The lab-trained policy can output
        # up to 3 m/s, which is faster than the AgileX hardware should run
        # in the MVSL space. Clamping to ~0.8 m/s here keeps the policy's
        # *intent* (slow for curves, faster on straights) but capped to a
        # safe absolute speed for the real robot / sim. Min clamp prevents
        # the policy from idling below a useful crawl speed.
        self.declare_parameter("bridge_velocity_min", 0.1)
        self.declare_parameter("bridge_velocity_max", 0.8)
        # Reverse driving is harder to control (trailer leads, non-minimum-phase
        # dynamics) and feels visibly faster in the MVSL space, so the reverse
        # cap defaults lower than forward.
        self.declare_parameter("bridge_velocity_max_reverse", 0.4)
        # Multiplier applied to the reverse policy's commanded steer rate
        # (and the integrated target tire angle) before publication. 1.0 =
        # pass-through. Slightly >1 compensates for ROS actuator lag that
        # pygame training didn't include.
        self.declare_parameter("reverse_steer_rate_gain", 1.0)
        # EMA smoothing factor for the reverse policy's raw steer_rate output.
        # Reverse policy output exhibits 5 Hz alternating bang-bang rates that
        # pygame (zero actuator lag) can execute but ROS's 50-150 ms actuator
        # delay turns into destructive oscillation. alpha=1.0 = no smoothing
        # (pass through). alpha=0.3 = each tick is 30% new + 70% previous,
        # giving ~3-tick effective averaging. Forward driving is left
        # unfiltered.
        # EMA: 1.0 = OFF. The 0.3 lag was tuned for the OLD v22 models' 5 Hz
        # bang-bang output; the new lab models steer smoothly, and 0.3 s of
        # steering lag destabilises the (non-minimum-phase) reverse loop. Default
        # OFF for the new models; raise toward 0.3 only if oscillation appears.
        self.declare_parameter("reverse_steer_rate_ema_alpha", 1.0)
        # Sign of the reverse steering-rate command. MUST be -1.0. The adapter
        # leaves obs[0]=-measured_steering (mirrored); with -1.0 the steering
        # integrates ∫(-action)dt, so obs[0]=-measured=+∫action = the native
        # steering state, AND the physical turn direction matches what the
        # policy intends for the un-mirrored lateral obs. (+1.0 inverts the turn
        # -> divergence.) Pairs with reverse_native_obs=True.
        self.declare_parameter("reverse_steer_rate_sign", -1.0)
        # Reverse kinematic steer-rate down-scaling (scale = v_clamped/v_intent).
        # STALE: it assumes the policy trained at v in [-3,-0.5] m/s (old semi-
        # truck v22 models) and over-steers at the clamped lab speed. The new lab
        # models trained at ~0.5-0.8 m/s, so this just robs steering authority.
        # Default OFF for the new models.
        self.declare_parameter("reverse_kinematic_scaling", False)
        # Lateral-obs sign for REVERSE TRACTOR-ONLY (multiplies e_y AND e_psi).
        # MUST be -1.0. The reverse env-frame transform (-yaw+pi rotation + Y-flip)
        # inverts the lateral error sign vs the policy's training convention, giving
        # a positive-feedback lateral loop: with +1.0 the truck reverses straight
        # briefly then veers off and wanders in big arcs (sim 2026-06-28). -1.0
        # closes the loop: it reverses in a near-straight line and reaches the goal
        # (drive disengages within ~2 m). NOTE: a small residual lateral drift
        # (~0.9 m over ~3 m) remains under -1.0 — stable, reaches goal; refine
        # later (possibly e_psi wants separate handling). This is the SECOND half
        # of the reverse fix; the first is reverse_steer_rate_sign=-1. Live-tunable.
        # Recover the native training obs for REVERSE TRACTOR-ONLY by fully
        # un-mirroring the env-frame obs (negate the 5-D state + reverse lidar).
        # Verified deploy==native build_observation. Fixes curvature sign + lidar
        # order that the piecemeal e_y/e_psi flips left wrong. Live-tunable.
        self.declare_parameter("reverse_native_obs", True)
        # FORWARD tractor-only native obs (full un-mirror incl steer). Makes
        # forward behave like reverse (works both lane directions) instead of the
        # fragile mirror convention. Pairs with forward_steer_rate_sign=+1.0
        # (apply the action directly, since obs[0]=steer is also un-mirrored).
        self.declare_parameter("forward_native_obs", True)
        self.declare_parameter("forward_steer_rate_sign", 1.0)

        # Smith Predictor: predictor-feedback shim that lets a delay-free-
        # trained policy (e.g. v15) deploy on a delayed sim/robot. When
        # enabled, the bridge runs a shadow `StateSpaceTractorTrailer` in
        # parallel; each tick, the shadow is re-synced from the measured
        # ROS state and rolled forward `smith_delay_steps` ticks through
        # the pending action queue. The predicted future state is what's
        # fed to the adapter / policy, so the policy effectively sees a
        # delay-free virtual env. See `scripts/smith_predictor.py`.
        self.declare_parameter("use_smith_predictor", False)
        self.declare_parameter("smith_steer_tau", 0.05)
        self.declare_parameter("smith_velocity_tau", 0.10)
        self.declare_parameter("smith_delay_steps", 2)

        self.e2e_rl_path = str(self.get_parameter("e2e_rl_path").value)
        self.trailer = bool(self.get_parameter("trailer").value)
        # Pick the active model pair from the trailer flag. trailer:=false → the
        # tractor-only checkpoints (29-D obs); trailer:=true → the trailer
        # checkpoints (32-D obs). The obs space follows from the chosen model's
        # meta in _load_model; the post-load guard asserts they agree.
        if self.trailer:
            self.model_path = str(self.get_parameter("td3_model_path").value)
            self.reverse_model_path = str(self.get_parameter("td3_reverse_model_path").value)
        else:
            self.model_path = str(self.get_parameter("td3_model_path_tractor_only").value)
            self.reverse_model_path = str(
                self.get_parameter("td3_reverse_model_path_tractor_only").value
            )
        self.action_space = str(self.get_parameter("action_space").value)
        self.stop_threshold = float(self.get_parameter("stop_threshold").value)
        self._stop_active = False   # True while the policy is commanding a stop
        self.control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.bev_publish_rate_hz = float(self.get_parameter("bev_publish_rate_hz").value)
        self.max_steering = float(self.get_parameter("max_steering_rad").value)
        self.default_velocity = float(self.get_parameter("default_velocity_mps").value)
        self.world_scale = float(self.get_parameter("world_scale").value)
        self.bridge_v_min = float(self.get_parameter("bridge_velocity_min").value)
        self.bridge_v_max = float(self.get_parameter("bridge_velocity_max").value)
        self.bridge_v_max_reverse = float(self.get_parameter("bridge_velocity_max_reverse").value)
        self.reverse_steer_rate_gain = float(self.get_parameter("reverse_steer_rate_gain").value)
        self.reverse_steer_rate_ema_alpha = float(self.get_parameter("reverse_steer_rate_ema_alpha").value)
        self.reverse_steer_rate_sign = float(self.get_parameter("reverse_steer_rate_sign").value)
        self.reverse_kinematic_scaling = bool(self.get_parameter("reverse_kinematic_scaling").value)
        self.reverse_native_obs = bool(self.get_parameter("reverse_native_obs").value)
        self.forward_native_obs = bool(self.get_parameter("forward_native_obs").value)
        self.forward_steer_rate_sign = float(self.get_parameter("forward_steer_rate_sign").value)
        self._reverse_steer_rate_ema = 0.0
        self.use_smith_predictor = bool(self.get_parameter("use_smith_predictor").value)
        self.smith_steer_tau = float(self.get_parameter("smith_steer_tau").value)
        self.smith_velocity_tau = float(self.get_parameter("smith_velocity_tau").value)
        self.smith_delay_steps = int(self.get_parameter("smith_delay_steps").value)
        self.smith_predictor = None  # built after env+adapter setup below
        self._last_kine_scale = 1.0

        if not self.model_path:
            which = "td3_model_path" if self.trailer else "td3_model_path_tractor_only"
            raise RuntimeError(
                f"rl_bridge_node: {which} parameter is required "
                f"(trailer={self.trailer}). With trailer:=false the tractor-only "
                f"checkpoints must be exported first "
                f"(re_export_td3.py --tractor-only)."
            )

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
        e2erl_config.trailer_length_m = 2.8
        # Trailer width matches the tractor width — the real AgileX trailer
        # is roughly the same width as the truck, not a narrower box.
        e2erl_config.trailer_width_m = 0.65
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

        # If the loaded policy was trained variable-speed (2-D action),
        # the env it was built against had its action_space overridden to
        # [steer_rate, velocity] with velocity bounded to [v_min, v_max].
        # LidarStateObservationLineFollowingEnv has no fixed_speed kwarg
        # so the adapter silently drops it; without this patch the env
        # defaults to fixed_speed=True → 1-D action space → state_dict
        # load fails with a shape mismatch on actor.mu.4. Mirrors
        # train_lab_model._patch_variable_speed_action exactly.
        if self.action_space == "variable_speed":
            # v_min/v_max here must match the policy's training action
            # space exactly — the state_dict shape depends on it via
            # SB3's TanhSquasher. v13+: 0.5/3.0. Earlier checkpoints
            # (v9-v12) trained with 0.1/3.0; override at launch if loading
            # one of those.
            self._patch_variable_speed_envs(v_min=0.5, v_max=3.0)

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

        # Guard: the loaded checkpoint's env class must agree with the trailer
        # flag. A TractorOnly model fed 32-D trailer obs (or a trailer model fed
        # 29-D tractor obs) silently produces garbage actions, so fail loudly at
        # startup instead. env_class_module is "Environments.TractorOnly" for
        # tractor-only checkpoints.
        is_tractor_only_model = "TractorOnly" in env_class_module
        if self.trailer and is_tractor_only_model:
            raise RuntimeError(
                f"trailer:=true but the forward checkpoint ({self.model_path}) is "
                f"a TractorOnly model ({env_class_module}.{env_class_name}). Set "
                f"trailer:=false or point td3_model_path at a trailer model."
            )
        if not self.trailer and not is_tractor_only_model:
            raise RuntimeError(
                f"trailer:=false but the forward checkpoint ({self.model_path}) is "
                f"NOT a tractor-only model ({env_class_module}.{env_class_name}). "
                f"Re-export it with re_export_td3.py --tractor-only, or set "
                f"trailer:=true."
            )

        env_kwargs = dict(meta.get("env_kwargs", {}))
        # Constant speed for both fixed_speed AND stop_signal (the env's velocity
        # handling is unused at deploy anyway — the env is never stepped, only
        # its action/obs spaces are read — but keep the semantics honest).
        env_kwargs.setdefault("fixed_speed", self.action_space != "variable_speed")
        self.get_logger().info(
            f"Instantiating env {env_class_module}.{env_class_name} kwargs={env_kwargs}"
        )
        self.adapter = ROSLineFollowingAdapter(
            env_class_module=env_class_module,
            env_class_name=env_class_name,
            env_kwargs=env_kwargs,
            world_scale=self.world_scale,
        )
        self.adapter.set_reverse_native_obs(self.reverse_native_obs)
        self.adapter.set_forward_native_obs(self.forward_native_obs)

        # Stop-signal action mode: override the env's action_space to 2-D
        # [steer_rate, stop_signal], stop_signal ∈ [-1, 1]. Set directly on the
        # instance (not via an __init__ monkey-patch) so it works for every env
        # class including the TractorOnly subclasses. TD3 is built against this
        # space below, and predict() CLIPS the stop output to these bounds — so
        # they must be [-1,1], not a velocity range.
        if self.action_space == "stop_signal":
            self.adapter.env.action_space = self._stop_signal_action_space()
            self.get_logger().info(
                f"stop_signal mode: forward action_space → {self.adapter.env.action_space} "
                f"(stop when action[1] > {self.stop_threshold})"
            )

        # Smith Predictor: instantiate AFTER adapter so the e2e_rl path is
        # on sys.path (adapter does that). The predictor's shadow vehicle
        # uses the same kinematic model the env wraps.
        if self.use_smith_predictor:
            from electrans_rl_bridge.smith_predictor import SmithPredictorState
            action_dim = 2 if self.action_space == "variable_speed" else 1
            self.smith_predictor = SmithPredictorState(
                steer_tau=self.smith_steer_tau,
                velocity_tau=self.smith_velocity_tau,
                dt=1.0 / self.control_rate_hz,
                delay_steps=self.smith_delay_steps,
                action_dim=action_dim,
            )
            self.get_logger().info(
                f"Smith Predictor (dual-shadow + residual) ENABLED  "
                f"τ_steer={self.smith_steer_tau}s  τ_vel={self.smith_velocity_tau}s  "
                f"action_dim={action_dim}"
            )
            # Action issued on the previous tick (currently in the
            # actuator pipeline). Initialise to zero; gets overwritten
            # after the first policy call.
            self._last_smith_action = np.zeros(action_dim, dtype=np.float32)

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
        # Clear error if the checkpoint's action dim disagrees with action_space
        # (otherwise load_state_dict throws a cryptic "size mismatch for
        # actor.mu.4.weight"). stop_signal + variable_speed need a 2-D actor;
        # fixed_speed needs 1-D. MODE=fixed training produces 1-D models.
        fwd_action_dim = int(state_dict["actor.mu.4.weight"].shape[0])
        expected_dim = 1 if self.action_space == "fixed_speed" else 2
        if fwd_action_dim != expected_dim:
            raise RuntimeError(
                f"action_space='{self.action_space}' expects a {expected_dim}-D "
                f"actor but the forward checkpoint {self.model_path} is "
                f"{fwd_action_dim}-D. For stop_signal you need models trained "
                f"with MODE=stop_signal; a MODE=fixed model is 1-D — either "
                f"retrain, or launch with action_space:=fixed_speed."
            )
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

            # Detect the reverse checkpoint's action dim by peeking at the
            # final actor layer in the state dict. The forward env was
            # patched to 2-D variable-speed above; the reverse env may
            # need the same (v15+) or stay 1-D (legacy v4). Mismatch =>
            # state_dict load fails with size mismatch on actor.mu.4.
            rev_state_dict = torch.load(self.reverse_model_path, map_location="cpu")
            rev_action_dim = int(rev_state_dict["actor.mu.4.weight"].shape[0])
            self.get_logger().info(
                f"Reverse checkpoint action_dim = {rev_action_dim} "
                f"({'variable-speed' if rev_action_dim == 2 else 'fixed-speed'})"
            )
            if rev_action_dim == 2 and self.action_space == "variable_speed":
                # Patch the reverse env class to variable-speed bounds
                # matching forward training (must agree with how the
                # reverse model was trained). Skipped for stop_signal — that
                # mode overrides the reverse env's action_space directly below.
                self._patch_reverse_env_variable_speed(v_min=0.5, v_max=3.0)

            # Build a SEPARATE env for the reverse policy. Its action_space
            # must match the reverse checkpoint's actor.mu output dim. The
            # reverse env is NEVER stepped — TD3 only inspects its
            # action_space and observation_space at construction. Per-tick
            # the bridge feeds obs from the forward adapter to
            # reverse_model.predict().
            import importlib, inspect
            rev_env_class_name = reverse_meta.get(
                "env_class_name", "ReverseStateObservationLineFollowingEnv"
            )
            rev_env_class_module = reverse_meta.get(
                "env_class_module", "Environments.LineFollowing"
            )
            rev_env_kwargs = dict(reverse_meta.get("env_kwargs", {}))
            rev_mod = importlib.import_module(rev_env_class_module)
            rev_env_cls = getattr(rev_mod, rev_env_class_name)
            rev_init_kwargs = {"render_mode": None, "reward_mode": "dense"}
            rev_init_kwargs.update(rev_env_kwargs)
            rev_sig = inspect.signature(rev_env_cls.__init__)
            rev_init_kwargs = {
                k: v for k, v in rev_init_kwargs.items() if k in rev_sig.parameters
            }
            reverse_env = rev_env_cls(**rev_init_kwargs)
            if self.action_space == "stop_signal":
                reverse_env.action_space = self._stop_signal_action_space()

            # CRITICAL: build reverse-mode OBSERVATIONS with this Reverse* env.
            # Its get_errors wraps reverse heading to ~0; the forward adapter env
            # would give e_psi ~ pi and full-lock the reverse policy. Previously
            # this env was only used to size the TD3 and obs came from the
            # forward env — the reverse wall-slam bug.
            self.adapter.set_reverse_env(reverse_env)

            self.get_logger().info(
                f"Loading reverse TD3 policy state from {self.reverse_model_path} "
                f"(env action_space={reverse_env.action_space})"
            )
            self.reverse_model = TD3(
                policy=sb3_policy_class,
                env=reverse_env,
                policy_kwargs=reverse_meta["policy_kwargs"],
                buffer_size=1,
                device="auto",
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
        # Raw policy action (BEFORE gain/clamps): [steer_rate, velocity_intent]
        # for diagnosing whether the policy itself is under-actuating or
        # whether bridge-side clamping is.
        self.pub_raw_action = self.create_publisher(Float32MultiArray, "/rl_bridge/raw_action", 1)

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

        # Runtime-tunable params so we can A/B-test without restarting the
        # launch. Only the two reverse-deployment knobs we expect to iterate
        # on are honoured here; the others (e.g. world_scale) need a restart.
        from rcl_interfaces.msg import SetParametersResult
        def _on_set_params(params):
            for p in params:
                if p.name == "bridge_velocity_max_reverse":
                    self.bridge_v_max_reverse = float(p.value)
                    self.get_logger().info(f"bridge_velocity_max_reverse -> {self.bridge_v_max_reverse}")
                elif p.name == "reverse_steer_rate_gain":
                    self.reverse_steer_rate_gain = float(p.value)
                    self.get_logger().info(f"reverse_steer_rate_gain -> {self.reverse_steer_rate_gain}")
                elif p.name == "reverse_steer_rate_ema_alpha":
                    self.reverse_steer_rate_ema_alpha = float(p.value)
                    self.get_logger().info(f"reverse_steer_rate_ema_alpha -> {self.reverse_steer_rate_ema_alpha}")
                elif p.name == "reverse_steer_rate_sign":
                    self.reverse_steer_rate_sign = float(p.value)
                    self.get_logger().info(f"reverse_steer_rate_sign -> {self.reverse_steer_rate_sign}")
                elif p.name == "reverse_kinematic_scaling":
                    self.reverse_kinematic_scaling = bool(p.value)
                    self.get_logger().info(f"reverse_kinematic_scaling -> {self.reverse_kinematic_scaling}")
                elif p.name == "reverse_native_obs":
                    self.reverse_native_obs = bool(p.value)
                    self.adapter.set_reverse_native_obs(self.reverse_native_obs)
                    self.get_logger().info(f"reverse_native_obs -> {self.reverse_native_obs}")
                elif p.name == "forward_native_obs":
                    self.forward_native_obs = bool(p.value)
                    self.adapter.set_forward_native_obs(self.forward_native_obs)
                    self.get_logger().info(f"forward_native_obs -> {self.forward_native_obs}")
                elif p.name == "forward_steer_rate_sign":
                    self.forward_steer_rate_sign = float(p.value)
                    self.get_logger().info(f"forward_steer_rate_sign -> {self.forward_steer_rate_sign}")
            return SetParametersResult(successful=True)
        self.add_on_set_parameters_callback(_on_set_params)

        self.create_timer(self._dt, self._on_control_tick)
        self.create_timer(1.0, self._on_gear_tick)
        # Debug BEV renders on its own slow timer, decoupled from control.
        # Single-threaded executor (rclpy.spin) serializes this with the
        # control tick, so there's no race on the shared adapter state the
        # control tick writes via the setters.
        if self.bev_publish_rate_hz > 0.0:
            self.create_timer(1.0 / self.bev_publish_rate_hz, self._on_bev_tick)

        self.get_logger().info(
            f"RL bridge up — action_space={self.action_space}, rate={self.control_rate_hz} Hz, "
            f"world_scale={self.world_scale}"
        )

    # ------------------------------------------------------------- patches
    def _patch_variable_speed_envs(self, *, v_min: float, v_max: float) -> None:
        """Mirror of train_lab_model._patch_variable_speed_action — wraps the
        Lidar env __init__'s so they end up with self.fixed_speed=False AND
        a 2-D action space [steer_rate, velocity] with velocity ∈ [v_min,
        v_max] (forward) or [-v_max, -v_min] (reverse). The adapter constructs
        the env afterwards and SB3 reads action_space from it; the loaded
        .pth state_dict then matches by shape."""
        import numpy as np
        from gymnasium import spaces
        import Environments.LineFollowing as lf
        import Environments.ObstacleAvoidance as oa

        try:
            from e2erl_utils import config as c
            steering_deg = float(c.steering_action)
        except Exception:
            steering_deg = 25.0
        max_steer_rate = np.deg2rad(steering_deg)

        orig_forward_init = oa.LidarStateObservationLineFollowingEnv.__init__

        def forward_init(self, render_mode="human", max_episode_steps=1000,
                         lidar_beams=16, reward_mode: str = "dense"):
            orig_forward_init(
                self,
                render_mode=render_mode,
                max_episode_steps=max_episode_steps,
                lidar_beams=lidar_beams,
                reward_mode=reward_mode,
            )
            self.fixed_speed = False
            self.action_space = spaces.Box(
                low=np.array([-max_steer_rate, v_min], dtype=np.float32),
                high=np.array([max_steer_rate, v_max], dtype=np.float32),
                dtype=np.float32,
            )

        oa.LidarStateObservationLineFollowingEnv.__init__ = forward_init
        # Reverse env is patched ONLY when we actually load a variable-
        # speed reverse checkpoint — see _patch_reverse_env_variable_speed
        # below. Per-checkpoint detection (via state_dict shape) lets the
        # bridge load either a legacy 1-D reverse model OR a v15+ 2-D
        # reverse model without manual config flipping.

    def _patch_reverse_env_variable_speed(self, *, v_min: float, v_max: float) -> None:
        """Patch ReverseLidarStateObservationLineFollowingEnv.__init__ so
        its action_space matches a variable-speed (2-D) reverse policy.
        Called ONLY when the loaded reverse .pth has actor.mu out-dim = 2.

        Velocity bounds are flipped negative for the reverse convention
        ([-v_max, -v_min] for the velocity dimension), matching how the
        reverse policy was trained (e2e_rl reverse envs use ẋ < 0)."""
        import numpy as np
        from gymnasium import spaces
        import Environments.LineFollowing as lf

        try:
            from e2erl_utils import config as c
            steering_deg = float(c.steering_action)
        except Exception:
            steering_deg = 25.0
        max_steer_rate = np.deg2rad(steering_deg)

        orig_reverse_init = lf.ReverseLidarStateObservationLineFollowingEnv.__init__

        def reverse_init(self, render_mode="human", max_episode_steps=1000,
                         lidar_beams=16, reward_mode: str = "dense",
                         fixed_speed: bool = True):  # noqa: ARG001 — forced
            orig_reverse_init(
                self,
                render_mode=render_mode,
                max_episode_steps=max_episode_steps,
                lidar_beams=lidar_beams,
                reward_mode=reward_mode,
                fixed_speed=False,
            )
            self.action_space = spaces.Box(
                low=np.array([-max_steer_rate, -v_max], dtype=np.float32),
                high=np.array([max_steer_rate, -v_min], dtype=np.float32),
                dtype=np.float32,
            )

        lf.ReverseLidarStateObservationLineFollowingEnv.__init__ = reverse_init

    def _stop_signal_action_space(self):
        """2-D action space [steer_rate, stop_signal] for constant-speed +
        learned-stop policies. stop_signal ∈ [-1, 1]; at deploy, action[1] >
        stop_threshold ⇒ STOP. Identical for forward and reverse (the stop dim
        is direction-agnostic) and for trailer vs tractor-only (action space is
        independent of the obs space). The [-1,1] bound is essential: TD3.predict
        clips the action to it, so a velocity-style bound would corrupt the stop
        signal."""
        import numpy as np
        from gymnasium import spaces
        try:
            from e2erl_utils import config as c
            steering_deg = float(c.steering_action)
        except Exception:
            steering_deg = 25.0
        max_steer_rate = np.deg2rad(steering_deg)
        return spaces.Box(
            low=np.array([-max_steer_rate, -1.0], dtype=np.float32),
            high=np.array([max_steer_rate, 1.0], dtype=np.float32),
            dtype=np.float32,
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
        # Seed target steering with the measured tire angle when we drift far
        # from the actual actuator state. Catches two cases:
        #   1. First tick after startup (target=0, measured may be nonzero).
        #   2. Sim resets (e.g. user re-publishes /initialpose mid-run for
        #      A/B testing). The sim resets the vehicle's tire angle to 0,
        #      so a large gap means the bridge is integrating stale state.
        # Threshold 0.40 rad (was 0.15): with the 2026-07-01 steering-feedback
        # calibration the measured angle is HONEST, so during fast maneuvers it
        # lags the target by (steer_rate x servo lag) ~ 0.2+ rad. At 0.15 this
        # guard fired mid-maneuver, snapping the integrator back to the lagging
        # measurement every few ticks -> limit-cycle oscillation. (It never
        # fired before the calibration only because measured read 1.32x target,
        # peaking at 0.139 gap — coincidentally just under 0.15.) 0.40 is only
        # reachable by a genuine reset/takeover, its actual purpose.
        if abs(self._target_steering - self._steering) > 0.40:
            self.get_logger().warn(
                f"steering re-seed: target {self._target_steering:+.3f} -> "
                f"measured {self._steering:+.3f} (reset/takeover?)")
            self._target_steering = self._steering
            # Big state jump (e.g. sim reset via /initialpose) — also
            # re-seed the Smith Predictor's shadows so they don't carry
            # stale state into the new episode.
            if self.smith_predictor is not None:
                # Pass the current ROS state so both shadows re-initialise
                # at the post-reset position; otherwise the first tick
                # after reset would have a huge residual.
                self.smith_predictor.reset(
                    x=x, y=y, yaw=yaw,
                    steer=self._steering, xd=xd,
                    hitch_angle=self._hitch_angle,
                )
                self._last_smith_action = np.zeros_like(self._last_smith_action)

        # Pick policy + adapter frame for this tick. Only honour drive_reverse
        # if we actually loaded a reverse checkpoint; otherwise stay forward.
        is_reverse = self._drive_reverse and self.reverse_model is not None
        self.adapter.set_reverse_mode(is_reverse)
        active_model = self.reverse_model if is_reverse else self.model

        # Smith Predictor (dual-shadow, with explicit residual feedback):
        # Both shadows step forward each tick with the previous command;
        # the residual = (real - delayed_shadow) absorbs model mismatch
        # between our Python kinematic model and ROS's C++ sim (different
        # RK4 vs Euler, different trailer kinematic equation, etc.). The
        # delay-free shadow's state + residual = the delay-free-equivalent
        # state to feed the (delay-free-trained) policy.
        if self.smith_predictor is not None and self._drive_enabled:
            # Use the last commanded action — the one currently propagating
            # through the actuator pipeline whose effect we just measured.
            x, y, yaw, predicted_steer, xd, predicted_hitch = (
                self.smith_predictor.step_and_predict(
                    self._last_smith_action,
                    x, y, yaw, self._steering, xd, self._hitch_angle,
                )
            )
            self.adapter.set_ego_state(x, y, yaw, predicted_steer, xd)
            self.adapter.set_trailer_state_from_hitch(predicted_hitch)
        else:
            self.adapter.set_ego_state(x, y, yaw, self._target_steering, xd)
            self.adapter.set_trailer_state_from_hitch(self._hitch_angle)

        try:
            obs = self.adapter.get_observation()
        except Exception as e:
            self.get_logger().warn(f"observation failed: {e}")
            return

        action, _ = active_model.predict(obs, deterministic=True)
        action = np.asarray(action).flatten()

        raw_msg = Float32MultiArray()
        raw_msg.data = [float(v) for v in action]
        self.pub_raw_action.publish(raw_msg)

        # Y-axis flip about the truck's forward axis is applied to the obs
        # in ros_env_adapter (centerline ys_local, v.s, β). The policy's
        # output is in that mirrored frame, so we negate the steering rate
        # to un-mirror it before commanding the real (un-flipped) vehicle.
        # With native_obs on, the policy sees the un-mirrored (native) obs and
        # emits the native steer_rate. REVERSE keeps obs[0] mirrored so it uses
        # reverse_steer_rate_sign=-1; FORWARD un-mirrors obs[0] too so it applies
        # the action directly via forward_steer_rate_sign=+1. (If native_obs is
        # off, set the sign to -1 to fall back to the old mirror convention.)
        # Both live-tunable.
        if is_reverse:
            steering_rate = self.reverse_steer_rate_sign * float(action[0])
        else:
            # Forward: the obs convention (and thus the action sign) depends on
            # whether the centerline was reversed (which lane-travel direction).
            # Reversed (anti-canonical) -> full-native un-mirror + sign +1;
            # not reversed (canonical) -> mirror convention + sign -1.
            if getattr(self.adapter, "_centerline_reversed", False) and self.forward_native_obs:
                steering_rate = self.forward_steer_rate_sign * float(action[0])
            else:
                steering_rate = -float(action[0])

        # Remember the just-chosen action so next tick's dual-shadow
        # `step_and_predict` can use it as the action that's currently
        # propagating through the actuator pipeline. The shadows propagate
        # in real-world frame (same as ROS), so they must receive the
        # un-flipped steering rate, NOT the policy's raw mirrored-frame
        # output — otherwise the predictor's state diverges from reality.
        if self.smith_predictor is not None:
            self._last_smith_action = np.asarray(action, dtype=np.float32).copy()
            self._last_smith_action[0] = steering_rate
        if self.action_space == "variable_speed" and action.size >= 2:
            velocity_cmd = float(action[1])
        else:
            velocity_cmd = -self.default_velocity if is_reverse else self.default_velocity
            # stop_signal mode: the policy's 2nd action dim is a LEARNED stop
            # decision. Above the threshold ⇒ command zero speed. The model owns
            # the decision (trained to stop only when it should); the bridge just
            # listens and zeroes velocity. Steering still tracks so the wheels
            # hold their angle while stopped.
            if self.action_space == "stop_signal" and action.size >= 2:
                stop_signal = float(action[1])
                stop_now = stop_signal > self.stop_threshold
                if stop_now != self._stop_active:
                    self.get_logger().info(
                        f"stop_signal {'ENGAGED' if stop_now else 'released'} "
                        f"(action[1]={stop_signal:.3f} vs thr {self.stop_threshold})"
                    )
                self._stop_active = stop_now
                if stop_now:
                    velocity_cmd = 0.0

        # When the Smith Predictor is on, the policy sees a delay-free
        # virtual env and its commanded rate is already calibrated for
        # that scenario. Skip the kinematic-scaling, EMA smoothing, and
        # rate-gain hacks — they were workarounds for an uncompensated
        # delayed env and are counterproductive once the predictor is
        # absorbing the latency.
        if self.smith_predictor is None:
            # Proportional kinematic scaling for reverse: the policy was trained
            # with the vehicle moving at v∈[-3, -0.5] m/s and outputs steer rates
            # calibrated for those speeds. The bridge clamps velocity to a much
            # lower value for the lab robot (~0.4 m/s), but the policy doesn't
            # see velocity in its observation, so it still commands rates as if
            # moving at -3 m/s. Result: tire angle accumulates over the same
            # number of seconds but the truck travels less distance, producing
            # a far tighter geometric turn than the policy intended. Scaling
            # steer_rate by |v_clamped|/|v_policy_intent| preserves the
            # steer-per-meter relationship.
            if (self.reverse_kinematic_scaling and is_reverse
                    and abs(velocity_cmd) > 1e-3):
                v_intent = abs(velocity_cmd)  # before clamp
                v_clamped = min(v_intent, self.bridge_v_max_reverse)
                v_clamped = max(v_clamped, self.bridge_v_min)
                scale = v_clamped / v_intent
                steering_rate *= scale
                self._last_kine_scale = scale  # for debug
            # EMA smoothing (kills 5 Hz bang-bang oscillation from the policy)
            if is_reverse:
                if self.reverse_steer_rate_ema_alpha < 1.0:
                    a = self.reverse_steer_rate_ema_alpha
                    self._reverse_steer_rate_ema = (
                        a * steering_rate + (1.0 - a) * self._reverse_steer_rate_ema
                    )
                    steering_rate = self._reverse_steer_rate_ema
                if self.reverse_steer_rate_gain != 1.0:
                    steering_rate *= self.reverse_steer_rate_gain
            else:
                self._reverse_steer_rate_ema = 0.0

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

        # Deployment-side magnitude clamp: the variable-speed policy was
        # trained against an action range of up to 3 m/s, but the AgileX
        # rig shouldn't run that fast in the MVSL space. Preserve the sign
        # (forward / reverse direction) while clamping the magnitude into
        # [bridge_v_min, bridge_v_max]. We never clamp through zero — if
        # the policy *would* command 0 / drive_disabled, that's set above
        # (velocity_cmd = 0.0 in the else branch) and skipped here.
        if self._drive_enabled and abs(velocity_cmd) > 0.0:
            sign = 1.0 if velocity_cmd >= 0.0 else -1.0
            v_max_effective = self.bridge_v_max_reverse if is_reverse else self.bridge_v_max
            mag = max(self.bridge_v_min, min(abs(velocity_cmd), v_max_effective))
            velocity_cmd = sign * mag

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

        # ---- debug publish (cheap part only) ----
        # Publish the full observation as a flat float vector on
        # /rl_bridge/state_vector every control tick (cheap). The 32x32 BEV
        # image is rendered + published on the separate _on_bev_tick timer so
        # its cost stays out of the control loop.
        if isinstance(obs, dict):
            vec = obs["vector"].astype(np.float32).flatten()
        else:
            vec = np.asarray(obs).astype(np.float32).flatten()
        vmsg = Float32MultiArray()
        vmsg.data = vec.tolist()
        self.pub_vec.publish(vmsg)

    def _on_bev_tick(self):
        """Render + publish the 32x32 debug BEV image on a slow timer,
        decoupled from the control loop. Always published (regardless of the
        policy's obs pipeline) so the env's view can be verified in RViz — a
        relay maps /rl_bridge/bev_image onto the RViz RecognitionResultOnImage
        panel topic. Uses the latest ego/trailer state the control tick stored
        via the adapter setters; skips quietly until both ego + path exist."""
        if self._ego is None or not self.adapter.has_path():
            return
        img = self.adapter.get_debug_bev_image()
        if img is None:
            return
        img_msg = Image()
        img_msg.header.stamp = self.get_clock().now().to_msg()
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
