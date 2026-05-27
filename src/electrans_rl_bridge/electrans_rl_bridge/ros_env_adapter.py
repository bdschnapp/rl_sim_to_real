"""
Headless adapter around the e2e_rl BEV training env.

The env's observation builders (_get_state_vector_obs / _get_bev_image_obs /
_render_frame / _build_occupancy_grid) are tightly coupled to self.vehicle and
self.xx/self.yy. Rather than copy ~200 LOC of those methods into this bridge
(which would drift out of sync with training), we subclass the env and push
ROS state into its members. The exact training-time observation pipeline then
runs unchanged.

e2e_rl is read-only — nothing in this file modifies anything under e2e_rl/.
"""

from __future__ import annotations

import math
import os
import sys
import numpy as np


def install_e2e_rl_on_path(e2e_rl_path: str) -> None:
    """Add the e2e_rl directory to sys.path so Environments / Models / e2erl_utils
    resolve. Also force pygame to render headlessly (no X11 needed inside ROS).
    Call this BEFORE importing from e2e_rl.
    """
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    if e2e_rl_path not in sys.path:
        sys.path.insert(0, e2e_rl_path)


class ROSLineFollowingAdapter:
    """Wraps BevObservationLineFollowingEnv. Composition over inheritance so we
    can fully control which methods are called and avoid running gym step logic.

    Usage:
        install_e2e_rl_on_path("/path/to/e2e_rl")
        adapter = ROSLineFollowingAdapter(fixed_speed=True)
        adapter.set_reference_path(xs, ys)
        adapter.set_ego_state(x, y, yaw, steering, xd)
        adapter.set_trailer_state_from_hitch(hitch_angle)
        obs = adapter.get_observation()
    """

    def __init__(
        self,
        env_class_module: str = "Environments.LineFollowing",
        env_class_name: str = "BevObservationLineFollowingEnv",
        env_kwargs: dict | None = None,
        world_scale: float = 1.0,
    ):
        """Instantiate the e2e_rl env that matches the trained policy. The
        env class + kwargs are picked at runtime so the same bridge binary can
        drive state-only / lidar-state / BEV checkpoints with no code change.

        env_kwargs is merged on top of {render_mode=None, reward_mode='dense'}.
        Imports are deferred so install_e2e_rl_on_path runs first.

        world_scale lets a sim or robot that's smaller than the training-time
        truck (e.g. 1/8 AgileX vs. real semi) feed observations to the policy
        at training scale. All positions handed to the env (ego, trailer,
        centerline) are multiplied by world_scale; conversely the bridge must
        divide the policy's velocity output by world_scale before commanding
        the real/sim vehicle. Steering angles are dimensionless and pass
        through unchanged.
        """
        import importlib
        import inspect

        kwargs = {"render_mode": None, "reward_mode": "dense"}
        if env_kwargs:
            kwargs.update(env_kwargs)

        mod = importlib.import_module(env_class_module)
        env_cls = getattr(mod, env_class_name)

        # Drop kwargs the target env doesn't accept (e.g.
        # LidarStateObservationLineFollowingEnv has no fixed_speed param).
        sig = inspect.signature(env_cls.__init__)
        kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

        self.env = env_cls(**kwargs)
        self.env_class_name = env_class_name
        self.world_scale = float(world_scale)

        # Raw ROS-frame state, transformed into truck-local on observation.
        self._raw_xs = None
        self._raw_ys = None
        self._ego_pose = None  # (x_world, y_world, yaw_world)
        self._steering = 0.0
        self._xd = 0.0
        self._hitch_angle = 0.0
        self._path_set = False

    # ---------------------------------------------------------------- setters
    # All setters store ROS-frame state; get_observation() does a single
    # transform into truck-local (+X-forward, vehicle at origin) before
    # invoking the env. This is necessary because the env's get_errors() uses
    # np.arctan(dy/dx) which collapses -X paths onto +X (they share the same
    # slope), making e_ψ = π whenever the centerline flows in -X relative to
    # the ROS map. Training only ever saw paths flowing +X.
    def set_reference_path(self, xs, ys):
        xs = np.asarray(xs, dtype=np.float32)
        ys = np.asarray(ys, dtype=np.float32)
        if xs.shape != ys.shape or xs.ndim != 1 or xs.size < 2:
            raise ValueError(
                f"set_reference_path: xs/ys must be 1-D arrays of matching size; "
                f"got xs={xs.shape}, ys={ys.shape}"
            )
        self._raw_xs = xs
        self._raw_ys = ys
        self._path_set = True

    def set_ego_state(self, x: float, y: float, yaw: float, steering: float, xd: float):
        self._ego_pose = (float(x), float(y), float(yaw))
        self._steering = float(steering)
        self._xd = float(xd)

    def set_trailer_state_from_hitch(self, hitch_angle: float):
        """Definition (matches training-time): γ = tractor.p - trailer.yaw."""
        self._hitch_angle = float(hitch_angle)

    # --------------------------------------------------------------- observ.
    def has_path(self) -> bool:
        return self._path_set

    def get_observation(self):
        """Run the env's training-time observation builder in truck-local
        frame. Return type depends on the configured env:
          - BevObservationLineFollowingEnv → dict {image, vector}
          - StateObservationLineFollowingEnv → np.ndarray (8,)
          - LidarStateObservationLineFollowingEnv → np.ndarray (8 + lidar_beams,)

        Caller must have called set_ego_state() this tick (so _ego_pose is
        not None) -- the bridge always does this in _on_control_tick before
        calling get_observation.
        """
        if not self._path_set or self._ego_pose is None:
            raise RuntimeError("get_observation: reference path or ego not set")
        self._refresh_env()
        return self.env._get_obs()

    def _refresh_env(self):
        x, y, yaw = self._ego_pose
        c, s = math.cos(-yaw), math.sin(-yaw)  # rotate world by -yaw -> truck-local

        # Centerline in truck-local frame, then scaled.
        dx = self._raw_xs - x
        dy = self._raw_ys - y
        xs_local = (dx * c - dy * s) * self.world_scale
        ys_local = (dx * s + dy * c) * self.world_scale
        self.env.xx = xs_local.astype(np.float32)
        self.env.yy = ys_local.astype(np.float32)
        self.env._build_occupancy_grid()

        v = self.env.vehicle
        v.x = 0.0           # vehicle is at env origin
        v.y = 0.0
        v.p = 0.0           # facing +X
        v.s = self._steering
        v.xd = self._xd * self.world_scale

        # Trailer in env-local. With v.p = 0, hitch is at (-v.lr, 0); trailer
        # axle sits behind hitch by trailer.L along trailer yaw.
        hitch_x = -v.lr
        hitch_y = 0.0
        trailer_yaw = -self._hitch_angle  # = v.p - hitch_angle, v.p == 0
        v.trailer.yaw = trailer_yaw
        v.trailer.x = hitch_x - v.trailer.L * math.cos(trailer_yaw)
        v.trailer.y = hitch_y - v.trailer.L * math.sin(trailer_yaw)
