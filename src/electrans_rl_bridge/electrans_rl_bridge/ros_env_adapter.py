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

        # The env hardcodes vehicle_params={'lf': 1.2, 'lr': 1.6, ...} inside
        # TractorTrailerEnv.__init__ (not config-driven). That puts the rear
        # axle 1.6 m behind the CG, which causes a ~1 m visible gap between
        # the truck rectangle (TRACTOR_LENGTH=1.0 m, centered at CG) and the
        # trailer rectangle (whose front is at the hitch = rear axle). Patch
        # lf/lr to the AgileX wheelbase split so the trailer connects flush.
        self._patch_vehicle_params(self.env)

        # Always-on BEV debug renderer. When the policy env is itself a BEV env
        # we reuse it; otherwise we spin up a separate BevObservationLineFollowingEnv
        # that mirrors the policy env's state on every tick so the bridge can
        # publish /rl_bridge/bev_image regardless of which obs pipeline the
        # policy uses. Roughly a few MB of extra pygame surfaces in memory.
        from Environments.LineFollowing import BevObservationLineFollowingEnv
        from Environments.TractorTrailer import (
            WINDOW_WIDTH, WINDOW_HEIGHT, METERS_PER_PIXEL,
        )
        if isinstance(self.env, BevObservationLineFollowingEnv):
            self.debug_bev_env = self.env
        else:
            debug_kwargs = {"render_mode": None, "reward_mode": "dense", "fixed_speed": True}
            # Filter to what BEV env actually accepts.
            sig = inspect.signature(BevObservationLineFollowingEnv.__init__)
            debug_kwargs = {k: v for k, v in debug_kwargs.items() if k in sig.parameters}
            self.debug_bev_env = BevObservationLineFollowingEnv(**debug_kwargs)
            self._patch_vehicle_params(self.debug_bev_env)

        # The env's pygame canvas spans [0, WORLD_W] × [0, WORLD_H] meters,
        # with (0, 0) at the bottom-left corner. We always centre the truck
        # in this canvas so the BEV crop captures the surrounding lane.
        # All obs values (e_y, e_ψ, e_y_t, lidar) are differences between
        # truck and centerline, so this constant offset is invisible to the
        # policy.
        self._world_offset_x = WINDOW_WIDTH * METERS_PER_PIXEL / 2.0
        self._world_offset_y = WINDOW_HEIGHT * METERS_PER_PIXEL / 2.0

        # Raw ROS-frame state, transformed into truck-local on observation.
        self._raw_xs = None
        self._raw_ys = None
        self._ego_pose = None  # (x_world, y_world, yaw_world)
        self._steering = 0.0
        self._xd = 0.0
        self._hitch_angle = 0.0
        self._path_set = False
        # When True, the policy in use was trained against a Reverse* env
        # (truck reset at yaw + π, vx < 0, motion along the path). In env
        # frame we mirror that: rotate world by -yaw + π so env +X is the
        # truck's BACKWARD direction (= the direction the truck physically
        # needs to move when backing up), set v.p = π so the truck's nose
        # in env-local points toward -X (matching training distribution).
        self._is_reverse = False

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

    def set_reverse_mode(self, is_reverse: bool):
        """Switch between forward and reverse env-frame layout. See
        _is_reverse in __init__ for the geometric meaning."""
        self._is_reverse = bool(is_reverse)

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
        # Forward mode: rotate world by -yaw so env +X is truck heading.
        # Reverse mode: rotate by -yaw + π so env +X is truck's BACKWARD
        # direction — that's the direction the truck physically moves when
        # backing up, and matches the reverse env's training-time setup
        # where motion was along the path's +X with the truck facing -X.
        rot_angle = -yaw + (math.pi if self._is_reverse else 0.0)
        c, s = math.cos(rot_angle), math.sin(rot_angle)

        # Centerline in truck-local frame, then scaled.
        dx = self._raw_xs - x
        dy = self._raw_ys - y
        xs_local = ((dx * c - dy * s) * self.world_scale).astype(np.float32)
        ys_local = ((dx * s + dy * c) * self.world_scale).astype(np.float32)

        # Direction sanity check: lane_reference_node always publishes the
        # centerline in the canonical lanelet direction. If the truck is
        # driving the OPPOSITE way along the lane, the local centerline flows
        # in -X (backward) through the truck in env-frame. The env's
        # curvature lookahead always reads forward in array order, so we
        # must reverse the order before storing it, otherwise k1/k2 read
        # from cells behind the truck instead of ahead.
        xs_local, ys_local = self._orient_centerline_forward(xs_local, ys_local)

        self._apply_state_to(self.env, xs_local, ys_local)
        if self.debug_bev_env is not self.env:
            self._apply_state_to(self.debug_bev_env, xs_local, ys_local)

    @staticmethod
    def _orient_centerline_forward(xs_local, ys_local):
        """Reverse the centerline order if its local tangent at the truck
        (env origin) points in -X. Ensures the lane always flows AHEAD of
        the truck in env-frame, regardless of which way the truck is
        driving along the canonical lanelet."""
        if xs_local.size < 2:
            return xs_local, ys_local
        # Nearest segment of centerline to the truck (at env origin).
        i = int(np.argmin(xs_local * xs_local + ys_local * ys_local))
        if 0 < i < xs_local.size - 1:
            dx_tan = xs_local[i + 1] - xs_local[i - 1]
        elif i == 0:
            dx_tan = xs_local[1] - xs_local[0]
        else:
            dx_tan = xs_local[-1] - xs_local[-2]
        if dx_tan < 0.0:
            return xs_local[::-1].copy(), ys_local[::-1].copy()
        return xs_local, ys_local

    @staticmethod
    def _patch_vehicle_params(env):
        """The training env hardcodes lf=1.2, lr=1.6 (Tesla Model S scale)
        in TractorTrailerEnv.__init__, which is wrong for the AgileX lab
        robot. Patch the vehicle model after construction so the rear-axle
        position used for trailer attachment matches the AgileX wheelbase."""
        v = env.vehicle
        # AgileX wheelbase = 0.65 m, split ~half-half around the CG so the
        # rear axle is 0.325 m behind the CG (i.e. inside the 1.0 m tractor
        # rectangle, not 1.6 m behind it where the trailer would float).
        v.lf = 0.325
        v.lr = 0.325

    def _apply_state_to(self, env, xs_local, ys_local):
        """Push centerline + vehicle + trailer state into an env in truck-local
        frame, then translate by the world-center offset so the pygame canvas
        actually contains the action. Called for both the policy env and the
        debug BEV env so they always agree on what's being rendered."""
        env.xx = (xs_local + self._world_offset_x).astype(np.float32)
        env.yy = (ys_local + self._world_offset_y).astype(np.float32)
        env._build_occupancy_grid()

        v = env.vehicle
        v.x = self._world_offset_x   # vehicle at world centre
        v.y = self._world_offset_y
        # Forward mode: truck nose at env +X. Reverse mode: nose at env -X
        # (matches Reverse* env training where p = path_tangent + π).
        v.p = math.pi if self._is_reverse else 0.0
        v.s = self._steering
        v.xd = self._xd * self.world_scale

        # Trailer reconstruction from hitch angle. Works for both modes
        # because we plug v.p into the standard formula; only the cos/sin
        # values flip sign when v.p = π.
        hitch_x = v.x - v.lr * math.cos(v.p)
        hitch_y = v.y - v.lr * math.sin(v.p)
        trailer_yaw = v.p - self._hitch_angle
        v.trailer.yaw = trailer_yaw
        v.trailer.x = hitch_x - v.trailer.L * math.cos(trailer_yaw)
        v.trailer.y = hitch_y - v.trailer.L * math.sin(trailer_yaw)

    def get_debug_bev_image(self):
        """Return the 32x32 BEV image of the current state. Always available
        (the adapter spins up a BEV env on construction even when the policy
        doesn't use a BEV obs). Caller must have set ego + reference path."""
        if not self._path_set or self._ego_pose is None:
            return None
        # If the policy env isn't BEV, _refresh_env may not have been called
        # yet via get_observation; ensure both envs see fresh state here.
        self._refresh_env()
        return self.debug_bev_env._get_bev_image_obs()
