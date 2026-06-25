#!/usr/bin/env python3
"""
Train a lab-scale TD3 policy by wrapping e2e_rl/train.py.

The upstream e2e_rl trees defaults at semi-truck scale (lf=1.2, lr=1.6,
trailer=10 m, 0.1 m/px map). This wrapper:

  1. Overrides e2erl_utils.config with AgileX lab values BEFORE the env
     modules import them at module load.
  2. Monkey-patches Environments.TractorTrailer.TractorTrailerEnv.__init__
     so the vehicle's lf/lr are read from config.tesla_model_s_vehicle_params
     instead of the upstream hardcoded semi-truck constants (the env's
     in-file dict bypasses the config dict for these two fields).
  3. Changes CWD to <repo>/lab_models so the resulting tree
     (lab_models/models/<scenario>/...) stays inside this repo and never
     touches e2e_rl/models/.

Outputs (default --reward multiplicative, --lidar-beams 24):

  <repo>/lab_models/models/<scenario>/lidar_24/multiplicative/best_model.zip
  <repo>/lab_models/models/<scenario>/lidar_24/multiplicative/final.zip
  <repo>/lab_models/models/<scenario>/lidar_24/multiplicative/logs/...

After training, run scripts/re_export_td3.py (with --reverse for the reverse
checkpoint) on best_model.zip to produce the portable .pth + .policy_kwargs.pkl
pair the bridge loads at runtime.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_E2E_RL = Path("/home/ben/Ben/Thesis/e2e_rl")
DEFAULT_OUT = REPO_ROOT / "lab_models_v16"


def _apply_lab_config_overrides(e2e_rl_path: Path) -> None:
    """Inject the AgileX lab values into e2erl_utils.config. Must be called
    BEFORE Environments.TractorTrailer (or anything that transitively imports
    it) is imported, since those modules read config at module-load time."""
    if str(e2e_rl_path) not in sys.path:
        sys.path.insert(0, str(e2e_rl_path))

    from e2erl_utils import config as c

    # We intentionally DO NOT shrink the world canvas here. The path generator
    # in LineFollowing.generate_path() hardcodes y≈45 m with x∈[5, world_w-5],
    # so a small canvas (e.g. the lab's 25×20 m) puts the spawn out-of-bounds
    # and every episode ends in 1 step. The canvas only affects rendering;
    # the policy's observation is dimensionless w.r.t. it (cross-track error,
    # angles, lidar distances, hitch angle) so keeping the training canvas
    # at the e2e_rl default 150×90 m is safe and matches the bridge's
    # truck-local-frame observation pipeline at runtime.

    # Vehicle dimensions. trailer_length_m doubles as the kinematic wheelbase
    # passed to StateSpaceTractorTrailer; 2.0 m matches the real AgileX rig.
    c.tractor_length_m = 1.0
    c.tractor_width_m = 0.65
    c.trailer_length_m = 2.0
    # Trailer width matches the tractor width (real AgileX trailer is the
    # same box as the cab, not a narrower stub).
    c.trailer_width_m = 0.65

    # Lane corridor sized to MVSL's LL7 (2.81 m wide → 1.41 m half-width).
    # These are read by the occupancy-grid + lidar pipelines so cross-track
    # error magnitudes and lidar distance distributions match the lab.
    c.lane_centerline_half_width_m = 1.41
    c.lane_shoulder_m = 0.20

    # Tractor wheelbase + CG split. Keep the rest of the dict (mass, inertia,
    # tire stiffness, dt) at training-default values — the AgileX is much
    # lighter but the env's TD3 trains a kinematic-dominant controller; the
    # dynamic-mode terms mostly shape transient response.
    c.tesla_model_s_vehicle_params = dict(
        c.tesla_model_s_vehicle_params,
        lf=0.33,
        lr=0.32,
    )


def _patch_path_generator(e2e_rl_path: Path) -> None:
    """Replace LineFollowingEnv.generate_path with a mixture distribution
    that exposes the policy to lab-corner-scale tangent changes during
    training.

    Upstream generate_path produces a cubic spline through 6 control points
    spread over 140 m with y ∈ ±5 m → min radius ≈ 40 m. For the lab-scale
    vehicle (wheelbase 0.65 m), that's ~60× wheelbase — gentle enough that
    the policy converges to near-zero steering outputs and can't handle the
    MVSL ~2 m corner (~3× wheelbase) at deployment.

    The mixture below per episode:
      -  5%  straight lane (prevents over-steering bias)
      - 80%  original gentle cubic spline (keeps the bulk of training in
             the existing distribution so the policy doesn't regress on
             the easy cases)
      - 15%  chained tanh bends — 4–8 alternating-sign ramps spread along
             the whole path so the policy is continuously tracking sharp
             curvature rather than 95% straight + one bump (the latter
             let the vehicle learn to ignore the corner). Per-bend Δy is
             3–8 m, width 1.5–3 m, so each local tangent jump is
             ~50–75°. Net path stays bounded in y because signs
             alternate.

    Per-episode velocity randomisation is applied alongside this in
    _patch_velocity_randomisation: gentle and straight episodes sample
    xd from [1.0, 5.0] m/s, sharp from [0.5, 2.0] m/s. The lower bound
    on sharp matches the bridge's deployment speed (0.6 m/s), where the
    MVSL corner is most demanding.

    Paths stay monotonic in x so the env's progress / success check
    (vehicle.x > 0.85 * world_width) is unchanged.
    """
    if str(e2e_rl_path) not in sys.path:
        sys.path.insert(0, str(e2e_rl_path))

    import numpy as np
    from scipy.interpolate import CubicSpline
    import Environments.LineFollowing as lf
    import Environments.TractorTrailer as tt

    def generate_path(self):
        world_width_m = tt.WINDOW_WIDTH * tt.METERS_PER_PIXEL
        x0 = 5.0
        x_end = world_width_m - 5.0
        vert_offset = 45.0

        self.xx = np.arange(int(x0), int(x_end), 1)
        n_pts = len(self.xx)

        # v18 mode mix: reclaims v15oos-level "straight" exposure (18%)
        # to fix v17's straight-driving regression, while keeping
        # "lab_corner" at 25% so the policy still gets heavy exposure
        # to the MVSL ~90° corner geometry. Other modes trimmed to
        # make the budget add up.
        mode = self.np_random.choice(
            ["straight", "gentle", "sharp", "winding", "lab_seam", "lab_corner"],
            p=[0.18, 0.12, 0.15, 0.18, 0.12, 0.25],
        )
        # Recorded so the velocity-randomisation reset patch (see
        # _patch_velocity_randomisation) can pick a speed appropriate for
        # the path type — lower for sharp paths so the MVSL corner is in
        # distribution at deployment speed (~0.6 m/s).
        self._path_kind = str(mode)

        if mode == "straight":
            y_local = np.zeros(n_pts, dtype=float)

        elif mode == "gentle":
            n = 6
            ctrl_x = np.linspace(x0, x_end, n)
            ctrl_y = -1.0 + 2.0 * self.np_random.random(n)
            # Match upstream: start near zero so spawn (10% along path)
            # sits roughly on the centerline.
            ctrl_y[0] = (-0.25 / 50) + self.np_random.random() * (0.5 / 50)
            cs = CubicSpline(ctrl_x, ctrl_y, bc_type=((1, 0.0), "not-a-knot"))
            y_local = cs(self.xx) * 5.0

        elif mode == "sharp":
            # Chained tanh bends, not "straight with one bump". Place
            # 4–8 bend centres evenly along the (post-spawn) path so
            # the vehicle is constantly tracking curvature. Centres
            # start at x0 + margin so the tanh prefix is flat at the
            # spawn point (x ≈ 18 m for path-fraction 0.1).
            n_bends = int(self.np_random.integers(4, 9))
            margin = 20.0
            base_centers = np.linspace(x0 + margin, x_end - margin, n_bends)
            gap = (x_end - x0 - 2 * margin) / max(1, n_bends - 1)
            jitter = (self.np_random.random(n_bends) - 0.5) * gap * 0.4
            bend_xs = sorted((base_centers + jitter).tolist())

            sign = 1.0 if self.np_random.random() < 0.5 else -1.0
            y_local = np.zeros(n_pts, dtype=float)
            for bend_x in bend_xs:
                # Tighter bend params than v12: width 1.0-1.5 m + dy 2.5-4 m
                # produce κ_max ≈ 0.30-0.45 m⁻¹, matching the MVSL 2 m
                # corner (κ = 0.5 m⁻¹).
                width = float(self.np_random.uniform(1.0, 1.5))
                dy = sign * float(self.np_random.uniform(2.5, 4.0))
                sign = -sign
                ramp = (np.tanh((self.xx - bend_x) / width) + 1.0) * 0.5
                y_local += dy * ramp

        elif mode == "lab_seam":
            # Single very-sharp tanh bend mid-path, modelling the MVSL map's
            # lanelet 7 → 402 → 381 seam: ~6 m straight, ~2 m of sharp
            # corner (κ ≈ 0.5 - 1.0 m⁻¹), then straight continues to the
            # post-bend tangent. Path stays monotonic in x (env requirement)
            # so we can't render the full 90° world rotation, but the
            # *curvature signal* the policy sees in the obs lookahead is
            # what matters — it'll be a localised step from κ=0 to
            # κ=max for a short window, then back to 0, which matches
            # what the deployment lookahead sees at the MVSL corner.
            #
            # κ_peak = (dy / 2) / width² ≈ 0.5–1.0 m⁻¹ with the params
            # below (width 0.4-0.7 m, dy 1.5-3 m). Clamped at the
            # obs_high curvature_observation=0.3 in the policy obs but
            # the actual path geometry preserves the sharpness.
            bend_x = float(self.np_random.uniform(x0 + 6.0, x_end - 4.0))
            width = float(self.np_random.uniform(0.4, 0.7))
            sign = 1.0 if self.np_random.random() < 0.5 else -1.0
            dy = sign * float(self.np_random.uniform(1.5, 3.0))
            ramp = (np.tanh((self.xx - bend_x) / width) + 1.0) * 0.5
            y_local = dy * ramp

        elif mode == "lab_corner":
            # MVSL ~90° corner replica, S-curve form: two sharp opposing
            # tanh bends with a LONG straight section between them. The
            # policy must drive into the first 90° turn, stabilise back
            # on straight, then handle a second 90° turn in the OPPOSITE
            # direction. Models the real MVSL trajectory.
            #
            # v18 tweaks vs v17:
            #   - Inter-bend straight is now 100-115 m (was 50-90 m), so
            #     the policy has substantial time to settle between turns
            #     before the next curvature event. Pre-bend straight is
            #     10-15 m (was 15-30) since the truck spawns at 10% along
            #     and we want the corner reasonably early.
            #   - Wider dy/width ranges (dy 2-6 m, width 0.3-0.8 m)
            #     diversify the corner sharpness so the policy interpolates
            #     between mild and very sharp 90°s instead of seeing only
            #     the hardest. Peak tangent angle atan(dy/(2w)) now spans
            #     ~50-85° per bend.
            sign = 1.0 if self.np_random.random() < 0.5 else -1.0
            bend1_x = float(self.np_random.uniform(x0 + 10.0, x0 + 15.0))
            width1 = float(self.np_random.uniform(0.3, 0.8))
            dy1 = sign * float(self.np_random.uniform(2.0, 6.0))

            bend2_x = float(self.np_random.uniform(bend1_x + 100.0, x_end - 10.0))
            width2 = float(self.np_random.uniform(0.3, 0.8))
            dy2 = -dy1  # opposite-direction equal-magnitude → net y ≈ 0

            ramp1 = (np.tanh((self.xx - bend1_x) / width1) + 1.0) * 0.5
            ramp2 = (np.tanh((self.xx - bend2_x) / width2) + 1.0) * 0.5
            y_local = dy1 * ramp1 + dy2 * ramp2

        else:  # winding — chained alternating-sign sustained turns
            # Replaces single-bend sustained_turn. Chains N=3 or 5 bends
            # so the slope alternates {+s, -s, +s, …} between segments
            # and returns to 0 after the last bend. The vehicle is
            # continuously either in a transition (high κ) or driving at
            # a sustained ±s slope — no long pre-bend straight that
            # dominated the single sustained_turn variant.
            #
            # n_bends odd → cumulative ∫slope·dx = 0 with evenly-spaced
            # bends, so y returns to baseline at path end (clean canvas
            # fit). The signed-slope contributions at each bend are:
            #     Δ_i = interval_slope[i+1] - interval_slope[i]
            # with interval_slope being [0, +s, -s, +s, -s, ..., 0].
            #
            # Interior bends contribute Δ=±2s (slope flips sign across
            # them) → peak κ ≈ 2s / (2w) = s/w. With s∈[0.5, 0.8],
            # w∈[0.8, 1.5], interior κ ∈ [0.33, 1.0], comfortably
            # spanning the MVSL corner's κ=0.5.
            n_bends = int(self.np_random.choice([3, 5]))
            s = float(self.np_random.uniform(0.5, 0.8))
            width = float(self.np_random.uniform(0.8, 1.5))
            initial_sign = 1.0 if self.np_random.random() < 0.5 else -1.0

            margin = 18.0
            base_xs = np.linspace(x0 + margin, x_end - margin, n_bends)
            gap = (x_end - x0 - 2.0 * margin) / max(1, n_bends - 1)
            jitter = (self.np_random.random(n_bends) - 0.5) * gap * 0.3
            bend_xs = sorted((base_xs + jitter).tolist())

            # Slope levels: first and last are 0 (flat ends), middle
            # alternate ±s.
            interval_slopes = [0.0]
            for i in range(n_bends - 1):
                sign = initial_sign if (i % 2 == 0) else -initial_sign
                interval_slopes.append(sign * s)
            interval_slopes.append(0.0)
            # len(interval_slopes) == n_bends + 1; one slope per interval
            # between adjacent bends, plus pre- and post-bend intervals.

            y_local = np.zeros(n_pts, dtype=float)
            for bend_x, slope_prev, slope_next in zip(
                bend_xs, interval_slopes[:-1], interval_slopes[1:]
            ):
                delta = slope_next - slope_prev
                if abs(delta) < 1e-9:
                    continue
                arg = (self.xx - bend_x) / width
                arg_0 = (x0 - bend_x) / width
                ln_cosh = np.log(np.cosh(arg))
                ln_cosh_0 = float(np.log(np.cosh(arg_0)))
                y_local += (delta / 2.0) * (
                    (self.xx - x0) + width * (ln_cosh - ln_cosh_0)
                )

        self.yy = y_local + vert_offset

        x_g = self.xx[-1]
        y_g = self.yy[-1]
        yaw_g = float(
            np.arctan2(self.yy[-1] - self.yy[-2], self.xx[-1] - self.xx[-2])
        )
        self.goal_pose = (x_g, y_g, yaw_g)

    lf.LineFollowingEnv.generate_path = generate_path


# Velocity-randomisation bounds keyed by path kind. Sharp paths get a
# lower range so the policy practices tight tracking at low speeds —
# exactly the regime the bridge uses at deployment (~0.6 m/s on the
# MVSL corner). Straight + gentle stay in the original training range.
_VELOCITY_BOUNDS_BY_PATH_KIND = {
    "straight":   (1.0, 5.0),
    "gentle":     (1.0, 5.0),
    "sharp":      (0.5, 2.0),
    "winding":    (0.5, 2.0),
    "lab_seam":   (0.5, 2.0),
    # lab_corner runs even slower — the ~90° bend demands tight tracking
    # at MVSL deployment speeds (~0.6 m/s) where the corner is widest in
    # the policy's distribution.
    "lab_corner": (0.4, 1.5),
}


def _patch_variable_speed_action(
    e2e_rl_path: Path,
    *,
    v_min: float = 0.1,
    v_max: float = 5.0,
) -> None:
    """Switch the forward and reverse Lidar envs from fixed-speed (1-D action,
    velocity hardcoded to config.initial_xd) to variable-speed (2-D action,
    policy outputs [steer_rate, velocity]). Override the env's variable-speed
    action_space so the velocity component is bounded to [v_min, v_max] for
    forward and [-v_max, -v_min] for reverse — preventing the policy from
    commanding zero/negative speed during forward driving (or vice versa).

    The lower bound > 0 means the policy can't stop the vehicle to avoid
    path-tracking penalty; it must always make progress. Combined with the
    env's max_episode_steps=1000 truncation and the +100 success bonus at
    0.85 * world_width, the policy is pressured to pick a higher speed
    where path tracking allows, and slow down only when curvature demands.
    """
    if str(e2e_rl_path) not in sys.path:
        sys.path.insert(0, str(e2e_rl_path))

    import numpy as np
    from gymnasium import spaces
    import Environments.LineFollowing as lf
    import Environments.ObstacleAvoidance as oa

    # Forward Lidar env: doesn't accept fixed_speed kwarg. We wrap its
    # __init__ so that after the original (fixed_speed=True) construction
    # we flip self.fixed_speed and overwrite the action_space.
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
        max_steer_rate = np.deg2rad(_config_steering_action(self))
        self.action_space = spaces.Box(
            low=np.array([-max_steer_rate, v_min], dtype=np.float32),
            high=np.array([max_steer_rate, v_max], dtype=np.float32),
            dtype=np.float32,
        )

    oa.LidarStateObservationLineFollowingEnv.__init__ = forward_init

    # Reverse Lidar env: accepts fixed_speed=True; we pass False and then
    # override the action_space bounds (the env's own variable-speed
    # action_space uses [-speed_action_high, -speed_action_low] = [-15, +5]
    # which is too wide and allows zero / sign flips).
    orig_reverse_init = lf.ReverseLidarStateObservationLineFollowingEnv.__init__

    def reverse_init(self, render_mode="human", max_episode_steps=1000,
                     lidar_beams=16, reward_mode: str = "dense",
                     fixed_speed: bool = True):  # noqa: ARG001 — ignored, forced False
        orig_reverse_init(
            self,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            lidar_beams=lidar_beams,
            reward_mode=reward_mode,
            fixed_speed=False,
        )
        max_steer_rate = np.deg2rad(_config_steering_action(self))
        self.action_space = spaces.Box(
            low=np.array([-max_steer_rate, -v_max], dtype=np.float32),
            high=np.array([max_steer_rate, -v_min], dtype=np.float32),
            dtype=np.float32,
        )

    lf.ReverseLidarStateObservationLineFollowingEnv.__init__ = reverse_init


def _config_steering_action(env_self) -> float:
    """Best-effort lookup of steering_action degrees; falls back if config
    isn't importable from this scope (it almost always is)."""
    try:
        from e2erl_utils import config as c
        return float(c.steering_action)
    except Exception:
        return 25.0


def _patch_velocity_randomisation(e2e_rl_path: Path) -> None:
    """Override LaneDrivingEnv.reset (forward) and
    ReverseStateObservationLineFollowingEnv.reset (reverse) so that each
    episode's initial speed is sampled from a path-kind-specific range
    instead of the e2e_rl default of config.initial_xd = 5 m/s.

    The forward env stores the per-step speed command in
    self.fixed_speed_command (positive); the reverse env stores it as
    -|speed| (line 957 of e2e_rl/Environments/LineFollowing.py). We
    write both that field and self.vehicle.xd directly after the parent
    reset, so the first step uses the new speed."""
    if str(e2e_rl_path) not in sys.path:
        sys.path.insert(0, str(e2e_rl_path))

    import Environments.LineFollowing as lf

    def _sample_speed(self) -> float:
        kind = getattr(self, "_path_kind", "gentle")
        low, high = _VELOCITY_BOUNDS_BY_PATH_KIND.get(kind, (1.0, 5.0))
        return float(self.np_random.uniform(low, high))

    original_lane_reset = lf.LaneDrivingEnv.reset

    def forward_reset(self, seed=None, options=None):
        obs, info = original_lane_reset(self, seed=seed, options=options)
        # Skip per-episode velocity randomisation in variable-speed mode —
        # the policy chooses velocity on every step from its 2-D action.
        if getattr(self, "fixed_speed", True):
            speed = _sample_speed(self)
            self.vehicle.xd = speed
            if hasattr(self, "fixed_speed_command"):
                self.fixed_speed_command = speed
        return obs, info

    lf.LaneDrivingEnv.reset = forward_reset

    original_reverse_reset = lf.ReverseStateObservationLineFollowingEnv.reset

    def reverse_reset(self, seed=None, options=None):
        obs, info = original_reverse_reset(self, seed=seed, options=options)
        if getattr(self, "fixed_speed", True):
            speed = _sample_speed(self)
            # Reverse env convention (see e2e_rl L957, L1069): vehicle.xd
            # stored positive in body frame; fixed_speed_command stored as
            # the negative target so step() commands [steer, -speed] and
            # the vehicle ramps from +speed body-forward to -speed body-rev.
            self.vehicle.xd = speed
            if hasattr(self, "fixed_speed_command"):
                self.fixed_speed_command = -speed
        return obs, info

    lf.ReverseStateObservationLineFollowingEnv.reset = reverse_reset


def _patch_curvature_speed_penalty(
    e2e_rl_path: Path,
    *,
    beta: float = 1.0,
    K: int = 10,
) -> None:
    """Add a curvature-aware speed penalty to the multiplicative reward.

    For every per-step reward in reward_mode == "multiplicative", subtract:

        β · |ẋ| · κ̂_K

    where

        κ̂_K = max_{j=0..K-1} |κ(s_i + j·Δs)|

    is the maximum absolute centerline curvature over the next K path
    samples ahead of the vehicle's nearest path-point projection.
    Δs ≈ 1 m is the path-sample spacing (env stores path as `self.xx` /
    `self.yy` at 1 m x-spacing).

    Physical reading: |ẋ| · κ̂ is the yaw rate the vehicle would need to
    track the upcoming curve at its current speed (rad/s). Penalising
    that quantity directly nudges the policy to lower |ẋ| when sharp
    curvature is upcoming — without changing behaviour on straight
    sections (κ̂ small) or near-stop motion (|ẋ| small).

    The policy already observes κ at lookahead 10 and 20 in the state
    vector (κ₁, κ₂), so input→output mapping for this reward is
    learnable.

    Terminal rewards (+100 success / -100 failure) and non-multiplicative
    reward modes are left untouched.
    """
    if str(e2e_rl_path) not in sys.path:
        sys.path.insert(0, str(e2e_rl_path))

    import numpy as np
    import Environments.LineFollowing as lf

    MAX_KAPPA = 0.3  # match compute_curvature's clip in e2e_rl

    def kappa_hat(env) -> float:
        # Nearest path sample to the trailer axle, matching compute_curvature.
        dx = env.xx - env.vehicle.trailer.x
        dy = env.yy - env.vehicle.trailer.y
        nearest = int(np.argmin(dx * dx + dy * dy))
        n = len(env.xx)
        # Finite-diff curvature needs i±1 valid, so clamp the window.
        start = max(1, nearest)
        end = min(nearest + K, n - 2)
        if end < start:
            return 0.0
        i = np.arange(start, end + 1)
        x_im1, x_i, x_ip1 = env.xx[i - 1], env.xx[i], env.xx[i + 1]
        y_im1, y_i, y_ip1 = env.yy[i - 1], env.yy[i], env.yy[i + 1]
        x_p = (x_ip1 - x_im1) * 0.5
        y_p = (y_ip1 - y_im1) * 0.5
        x_pp = x_ip1 - 2.0 * x_i + x_im1
        y_pp = y_ip1 - 2.0 * y_i + y_im1
        denom = (x_p**2 + y_p**2) ** 1.5 + 1e-6
        kappa = (x_p * y_pp - y_p * x_pp) / denom
        kappa = np.clip(kappa, -MAX_KAPPA, MAX_KAPPA)
        return float(np.max(np.abs(kappa)))

    # Forward env: LineFollowingEnv.get_reward
    original_forward_get_reward = lf.LineFollowingEnv.get_reward

    def forward_get_reward(self, error, error_theta, error_t, error_theta_t):
        base = original_forward_get_reward(
            self, error, error_theta, error_t, error_theta_t
        )
        if getattr(self, "reward_mode", None) != "multiplicative":
            return base
        # Skip on terminal: original returned ±100 for success/failure.
        # _get_term is cheap and cached on LaneDrivingEnv per step.
        if self._get_term():
            return base
        return base - beta * abs(self.vehicle.xd) * kappa_hat(self)

    lf.LineFollowingEnv.get_reward = forward_get_reward

    # Reverse env: ReverseStateObservationLineFollowingEnv.get_reward
    original_reverse_get_reward = (
        lf.ReverseStateObservationLineFollowingEnv.get_reward
    )

    def reverse_get_reward(self, error, error_theta, error_t, error_theta_t):
        base = original_reverse_get_reward(
            self, error, error_theta, error_t, error_theta_t
        )
        if getattr(self, "reward_mode", None) != "multiplicative":
            return base
        if self._get_term():
            return base
        return base - beta * abs(self.vehicle.xd) * kappa_hat(self)

    lf.ReverseStateObservationLineFollowingEnv.get_reward = reverse_get_reward


def _patch_target_speed_reward(
    e2e_rl_path: Path,
    *,
    gamma: float,
    sigma_low: float,
    sigma_high: float,
    v_target_min: float,
    v_target_max: float,
    K: int = 10,
    K_back: int = 5,
    kappa_max: float = 0.3,
    path_tightness: float = 2.0,
) -> None:
    """Replace the forward multiplicative reward (now truly multiplicative
    after the source edit) with a curvature-aware asymmetric baseline-
    subtracted Gaussian speed shape inside the product:

        R = α · path_term · hitch_term · speed_shape(ẋ, κ̂_K) - P_prox

    For v_target > epsilon (default 0.05 m/s, NO-OBSTACLE branch):

      σ_eff = σ_low  if |ẋ| < v_target  else  σ_high

                       max(0, g(|ẋ|, v_target; σ_eff) - g(0, v_target; σ_low))
        speed_shape = ─────────────────────────────────────────────────────────
                                  1 - g(0, v_target; σ_low)

        where g(x, μ; σ) = exp(-(x - μ)² / (2σ²))

    Asymmetric σ lets us penalise the slow side more sharply (σ_low
    narrow) while keeping tolerance to overshooting target (σ_high
    wider). At v=0 we use σ_low for the baseline so the normalisation
    is consistent and speed_shape(0) = 0 exactly.

    Properties at v_target > epsilon:
      - at |ẋ| = 0:        speed_shape = 0          (park gives zero reward)
      - at |ẋ| = v_target: speed_shape = 1          (peak preserved)
      - left of peak:      narrow falloff (σ_low)   → slow heavily penalised
      - right of peak:     wider falloff (σ_high)   → moderate fast tolerance

    For v_target ≤ epsilon (obstacle case — placeholder, not active
    until obstacle support lands):

        speed_shape = exp(-|ẋ|² / (2σ_low²))   ← peak at v=0, stop is optimal

    The obstacle branch deliberately keeps the (symmetric) peak-at-zero
    Gaussian shape so that future v_safety composition (v_target =
    min(v_curve, v_safety) → 0 when obstacle close) gracefully shifts
    the peak to v=0 without any code change here.

    Only the FORWARD multiplicative reward is replaced. Reverse already
    has speed inside the product (line ~1011) and is untouched.
    """
    if str(e2e_rl_path) not in sys.path:
        sys.path.insert(0, str(e2e_rl_path))

    import numpy as np
    import Environments.LineFollowing as lf

    MAX_KAPPA = kappa_max
    two_sigma_low_sq = 2.0 * sigma_low * sigma_low
    two_sigma_high_sq = 2.0 * sigma_high * sigma_high

    def kappa_hat(env) -> float:
        # Window covers K_back samples behind nearest + K samples ahead.
        # Look-behind keeps κ̂ high until the trailer is *fully past* a
        # bend, preventing the policy from speeding up mid-curve when
        # the look-ahead-only window's high portion has scrolled past.
        dx = env.xx - env.vehicle.trailer.x
        dy = env.yy - env.vehicle.trailer.y
        nearest = int(np.argmin(dx * dx + dy * dy))
        n = len(env.xx)
        start = max(1, nearest - K_back)
        end = min(nearest + K, n - 2)
        if end < start:
            return 0.0
        i = np.arange(start, end + 1)
        x_im1, x_i, x_ip1 = env.xx[i - 1], env.xx[i], env.xx[i + 1]
        y_im1, y_i, y_ip1 = env.yy[i - 1], env.yy[i], env.yy[i + 1]
        x_p = (x_ip1 - x_im1) * 0.5
        y_p = (y_ip1 - y_im1) * 0.5
        x_pp = x_ip1 - 2.0 * x_i + x_im1
        y_pp = y_ip1 - 2.0 * y_i + y_im1
        denom = (x_p**2 + y_p**2) ** 1.5 + 1e-6
        kappa = (x_p * y_pp - y_p * x_pp) / denom
        kappa = np.clip(kappa, -MAX_KAPPA, MAX_KAPPA)
        return float(np.max(np.abs(kappa)))

    original_forward_get_reward = lf.LineFollowingEnv.get_reward

    def forward_get_reward(self, error, error_theta, error_t, error_theta_t):
        if getattr(self, "reward_mode", None) != "multiplicative":
            return original_forward_get_reward(
                self, error, error_theta, error_t, error_theta_t
            )
        # Terminal: keep upstream ±100 success/failure bonuses unchanged.
        if self._get_term():
            return 100.0 if getattr(self, "success", False) else -100.0

        # Path-tracking term, with all four coefficients scaled by
        # path_tightness. The original e2e_rl ratios are preserved
        # (1.0 : 0.5 : 0.75 : 0.5) so the *relative* weighting of
        # vehicle vs trailer cross-track and heading errors is the
        # same; only the overall decay rate is sharpened. At
        # path_tightness=2.0, a 0.5 m e_y drops path_term from 0.61
        # (orig) to 0.37 — strong gradient toward centerline,
        # especially important for reverse where small drifts amplify.
        path_term = float(np.exp(
            -path_tightness * (
                abs(error)
                + 0.5 * abs(error_theta)
                + 0.75 * abs(error_t)
                + 0.5 * abs(error_theta_t)
            )
        ))
        hitch_angle = abs(self.vehicle.p - self.vehicle.trailer.yaw)
        hitch_term = float(np.exp(-1.5 * hitch_angle))

        kappa_val = kappa_hat(self)
        kappa_frac = min(kappa_val / MAX_KAPPA, 1.0)
        v_target = v_target_max + (v_target_min - v_target_max) * kappa_frac
        # Guard against div-by-zero if v_target_min is set to 0.
        v_target = max(v_target, 1e-3)

        # Asymmetric baseline-subtracted Gaussian. σ_low (narrow) on the
        # slow side of v_target, σ_high (wider) on the fast side. v=0
        # baseline uses σ_low so normalisation is consistent and
        # speed_shape(0) = 0 exactly. For v_target ≤ ε we fall back to
        # a peak-at-0 Gaussian (placeholder for obstacle mode — not in
        # use today since v_target_min ≥ 1.0).
        xd_abs = abs(self.vehicle.xd)
        EPSILON_V_TARGET = 0.05
        if v_target > EPSILON_V_TARGET:
            sigma_sq = two_sigma_low_sq if xd_abs < v_target else two_sigma_high_sq
            g_v = float(np.exp(-((xd_abs - v_target) ** 2) / sigma_sq))
            g_0 = float(np.exp(-(v_target ** 2) / two_sigma_low_sq))
            speed_shape = max(0.0, (g_v - g_0) / (1.0 - g_0))
        else:
            speed_shape = float(np.exp(-(xd_abs ** 2) / two_sigma_low_sq))

        return gamma * path_term * hitch_term * speed_shape - self._proximity_penalty()

    lf.LineFollowingEnv.get_reward = forward_get_reward

    # ── Reverse counterpart ──────────────────────────────────────────
    # Reverse env's source-level multiplicative reward (LineFollowing.py
    # ~L1011) is already truly multiplicative:
    #     R = 5 · clip(-ẋ, 0, 1) · path_term · hitch_term - P_prox
    # so no source-edit was needed (unlike forward). We swap the
    # monotonic `clip(-ẋ, 0, 1)` factor for the same baseline-subtracted
    # Gaussian we use forward, evaluated on |ẋ| (reverse has ẋ < 0).
    #
    # Differences from forward:
    #   * hitch coefficient stays at -2.0 (vs forward's -1.5) — e2e_rl's
    #     original stricter penalty for reverse, preserves anti-jackknife
    #     bias for the much-less-stable reverse dynamics.
    #   * terminal rewards stay at +200 success / -500 failure (vs
    #     forward's ±100) — reverse's original convention. The much
    #     larger failure penalty discourages jackknifing strongly.
    original_reverse_get_reward = (
        lf.ReverseStateObservationLineFollowingEnv.get_reward
    )

    def reverse_get_reward(self, error, error_theta, error_t, error_theta_t):
        if getattr(self, "reward_mode", None) != "multiplicative":
            return original_reverse_get_reward(
                self, error, error_theta, error_t, error_theta_t
            )
        if self._get_term():
            return 200.0 if getattr(self, "success", False) else -500.0

        path_term = float(np.exp(
            -path_tightness * (
                abs(error)
                + 0.5 * abs(error_theta)
                + 0.75 * abs(error_t)
                + 0.5 * abs(error_theta_t)
            )
        ))
        hitch_angle = abs(self.vehicle.p - self.vehicle.trailer.yaw)
        hitch_term = float(np.exp(-2.0 * hitch_angle))

        kappa_val = kappa_hat(self)
        kappa_frac = min(kappa_val / MAX_KAPPA, 1.0)
        v_target = v_target_max + (v_target_min - v_target_max) * kappa_frac
        v_target = max(v_target, 1e-3)

        # Compare |ẋ| against v_target. Reverse env produces ẋ < 0; the
        # speed-shape is defined on speed *magnitude*, identical to forward.
        xd_abs = abs(self.vehicle.xd)
        EPSILON_V_TARGET = 0.05
        if v_target > EPSILON_V_TARGET:
            sigma_sq = two_sigma_low_sq if xd_abs < v_target else two_sigma_high_sq
            g_v = float(np.exp(-((xd_abs - v_target) ** 2) / sigma_sq))
            g_0 = float(np.exp(-(v_target ** 2) / two_sigma_low_sq))
            speed_shape = max(0.0, (g_v - g_0) / (1.0 - g_0))
        else:
            speed_shape = float(np.exp(-(xd_abs ** 2) / two_sigma_low_sq))

        return gamma * path_term * hitch_term * speed_shape - self._proximity_penalty()

    lf.ReverseStateObservationLineFollowingEnv.get_reward = reverse_get_reward


def _patch_env_vehicle_params(e2e_rl_path: Path) -> None:
    """Override TractorTrailerEnv.__init__'s hardcoded vehicle_params with the
    config dict, so lf/lr actually reach the StateSpaceTractorTrailer the
    env constructs. (TractorTrailer.py:74-77 hardcodes lf=1.2, lr=1.6.)"""
    if str(e2e_rl_path) not in sys.path:
        sys.path.insert(0, str(e2e_rl_path))

    from e2erl_utils import config as c
    import Environments.TractorTrailer as tt
    from VehicleModels.tractor_trailer import StateSpaceTractorTrailer

    original_init = tt.TractorTrailerEnv.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # Replace the vehicle constructed inside original_init with one whose
        # params come from config.tesla_model_s_vehicle_params. The vehicle
        # has no episode state yet (reset() is called from outside __init__),
        # so substitution is safe.
        self.vehicle = StateSpaceTractorTrailer(
            args=dict(c.tesla_model_s_vehicle_params),
            trailer_length=c.trailer_length_m,
        )

    tt.TractorTrailerEnv.__init__ = patched_init


def _patch_actuator_lag(
    e2e_rl_path: Path,
    *,
    steer_tau: float = 0.05,
    velocity_tau: float = 0.10,
) -> None:
    """Monkey-patch StateSpaceVehicleModel.loop to simulate ROS-side
    actuator lag during training.

    The ROS simple_planning_simulator applies a first-order lag to the
    commanded steering tire angle (steer_time_constant=0.05 s) and
    acceleration (acc_time_constant=0.1 s) before the vehicle model
    consumes them. The pygame training env has zero lag — the policy
    commands `steering_rate` and the env applies it instantly via
    `self.s = clip(self.s + rate * dt, ...)`.

    This is the dominant sim-to-real gap for reverse driving (non-
    minimum-phase trailer dynamics amplify any phase delay into
    instability). We close the gap here by adding the same first-order
    lag in training:

        s_target = clip(s_target + action[0] * dt, ±π/4)   # the integrated commanded angle
        s        = s + (s_target - s) * dt / steer_tau     # actual tire angle, lagging
        # then use `s` (not s_target) in the dynamics

    This way the policy learns to anticipate actuator lag during training,
    so deployment behavior matches what it saw during training.
    """
    if str(e2e_rl_path) not in sys.path:
        sys.path.insert(0, str(e2e_rl_path))

    import numpy as np
    from VehicleModels.vehicle_model import StateSpaceVehicleModel

    orig_loop = StateSpaceVehicleModel.loop
    orig_reset = StateSpaceVehicleModel.reset

    def patched_reset(self, xd, x=0, y=0, p=0):
        orig_reset(self, xd, x, y, p)
        # Initialise lag state — both target and actual at 0 at reset
        self._s_target = 0.0
        self._xd_target = float(xd)

    def patched_loop(self, action):
        # Initialise lag state on first call (in case reset wasn't called
        # since the patch — e.g. older training resumes).
        if not hasattr(self, "_s_target"):
            self._s_target = float(self.s)
        if not hasattr(self, "_xd_target"):
            self._xd_target = float(self.xd)

        # 1. Update steering target by policy's commanded rate, clamp to ±π/4
        self._s_target = float(np.clip(
            self._s_target + action[0] * self.dt, -np.pi / 4, np.pi / 4
        ))
        # 2. Actual tire angle lags the target by first-order dynamics
        if steer_tau > 1e-6:
            self.s = self.s + (self._s_target - self.s) * (self.dt / steer_tau)
        else:
            self.s = self._s_target

        # 3. Velocity target lag — same first-order dynamics applied
        #    BEFORE passing to the longitudinal PID controller. The PID
        #    already has its own time constant from the vehicle mass etc.,
        #    so this lag is on top — but matches the sim's "acc_time_delay
        #    + acc_time_constant" which both lag the commanded acceleration.
        if velocity_tau > 1e-6:
            self._xd_target = self._xd_target + (
                action[1] - self._xd_target
            ) * (self.dt / velocity_tau)
        else:
            self._xd_target = float(action[1])

        # 4. Call original loop with the LAGGED velocity target. The
        #    original loop also re-applies action[0] to self.s — we
        #    pre-empted it by setting self.s already, so we pass a
        #    zero-rate action[0] to prevent double-integration.
        modified_action = np.array([0.0, self._xd_target], dtype=float)
        return orig_loop(self, modified_action)

    StateSpaceVehicleModel.loop = patched_loop
    StateSpaceVehicleModel.reset = patched_reset


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Lab-scale TD3 trainer. Wraps e2e_rl/train.py with config "
            "overrides for the AgileX 1/8-scale rig and routes outputs into "
            "this repo's lab_models/ directory."
        )
    )
    parser.add_argument(
        "--scenario",
        choices=["forward", "reverse"],
        required=True,
        help="forward or reverse lane-following.",
    )
    parser.add_argument(
        "--timesteps", type=int, default=200_000,
        help="Total environment steps (default: 200_000).",
    )
    parser.add_argument(
        "--n-envs", dest="n_envs", type=int, default=1,
        help="Parallel SubprocVecEnv workers (default: 1).",
    )
    parser.add_argument(
        "--lidar-beams", dest="lidar_beams", type=int, default=24,
        help="Number of lidar beams in the observation (default: 24).",
    )
    parser.add_argument(
        "--reward", default="multiplicative",
        help="Reward variant passed to the env (default: multiplicative).",
    )
    parser.add_argument(
        "--device", default="auto",
        help="PyTorch device: auto, cuda, cpu (default: auto).",
    )
    parser.add_argument(
        "--eval-freq", dest="eval_freq", type=int, default=10_000,
        help="Env steps between eval passes (default: 10_000).",
    )
    parser.add_argument(
        "--normalized-eval-freq", dest="normalized_eval_freq", type=int, default=30_000,
        help="Env steps between normalized-eval passes (default: 30_000).",
    )
    parser.add_argument(
        "--out-dir", dest="out_dir", default=str(DEFAULT_OUT),
        help=(
            "Output root. train.py writes to <out_dir>/models/<scenario>/"
            "lidar_<beams>/<reward>/. Defaults to <repo>/lab_models."
        ),
    )
    parser.add_argument(
        "--e2e-rl-path", dest="e2e_rl_path", default=str(DEFAULT_E2E_RL),
        help="Filesystem path to the e2e_rl checkout (default: %(default)s).",
    )
    parser.add_argument(
        "--variable-speed", dest="variable_speed", action="store_true",
        help=(
            "Train a variable-speed policy: action becomes 2-D "
            "[steer_rate, velocity]. Velocity action bounded by "
            "--v-min/--v-max. The bridge must be launched with "
            "action_space:=variable_speed to consume the 2-D action."
        ),
    )
    parser.add_argument(
        "--v-min", dest="v_min", type=float, default=0.5,
        help="Min |velocity| for variable-speed policy (default: 0.5 m/s).",
    )
    parser.add_argument(
        "--v-max", dest="v_max", type=float, default=3.0,
        help="Max |velocity| for variable-speed policy action (default: 3.0 m/s).",
    )
    parser.add_argument(
        "--target-speed-bonus", dest="target_speed_bonus", type=float, default=0.0,
        help=(
            "Overall scale α of the multiplicative reward: "
            "α · path · hitch · speed_shape(ẋ, κ̂) - P_prox, where "
            "speed_shape = (|ẋ|/v_target) · exp(-(|ẋ|-v_target)²/(2σ²)). "
            "0 disables (default — falls through to the source-level "
            "multiplicative reward which already uses clip(ẋ,0,1) for "
            "the speed factor). Try α = 4.0 to match the source's "
            "4 · path · hitch ceiling. v_target is a linear interp "
            "between --target-speed-v-max (at κ̂=0) and "
            "--target-speed-v-min (at κ̂ ≥ κ_max=0.3)."
        ),
    )
    parser.add_argument(
        "--target-speed-sigma-low", dest="target_speed_sigma_low", type=float, default=0.2,
        help=(
            "σ used on the slow side of v_target (default: 0.2 m/s). "
            "Smaller = sharper penalty for crawling below v_target."
        ),
    )
    parser.add_argument(
        "--target-speed-sigma-high", dest="target_speed_sigma_high", type=float, default=0.5,
        help=(
            "σ used on the fast side of v_target (default: 0.5 m/s). "
            "Wider = more tolerance for overshooting target."
        ),
    )
    parser.add_argument(
        "--target-speed-v-min", dest="target_speed_v_min", type=float, default=0.6,
        help=(
            "Target speed at max curvature (default: 0.6 m/s). Matches "
            "an equivalent-radius semi-truck taking a 90° turn at ~3 mph "
            "scaled down to the lab 1/8 vehicle: v_lab = v_semi · √(R_lab/R_semi) "
            "≈ 0.55 m/s at MVSL's 2 m corner radius."
        ),
    )
    parser.add_argument(
        "--target-speed-v-max", dest="target_speed_v_max", type=float, default=2.0,
        help="Target speed on straights (default: 2.0 m/s).",
    )
    parser.add_argument(
        "--curvature-speed-weight", dest="curvature_speed_weight", type=float,
        default=0.0,
        help=(
            "Strength of the curvature-aware speed penalty added to "
            "multiplicative reward. Penalty per step = β · |ẋ| · κ̂_K "
            "where κ̂_K = max upcoming centerline curvature over K "
            "samples. 0 disables (default). Try 1.0 first, 0.5 if "
            "training is unstable, 2.0 if behaviour doesn't change."
        ),
    )
    parser.add_argument(
        "--curvature-lookahead-samples", dest="curvature_lookahead",
        type=int, default=10,
        help=(
            "Path-sample lookahead window K for the curvature penalty. "
            "Default 10 ≈ 10 m of x-distance (path is 1 m-spaced). "
            "Matches the κ₁ lookahead already in obs."
        ),
    )
    parser.add_argument(
        "--curvature-lookbehind-samples", dest="curvature_lookbehind",
        type=int, default=5,
        help=(
            "Path-sample lookbehind window K_back for the curvature "
            "penalty. Default 5 ≈ 5 m. Keeps κ̂ high until the trailer "
            "is fully past a bend, preventing mid-curve speed-up."
        ),
    )
    parser.add_argument(
        "--path-tightness", dest="path_tightness", type=float, default=2.0,
        help=(
            "Multiplier on the path_term exponent coefficients in the "
            "multiplicative reward. 1.0 = original e2e_rl decay rate; "
            "2.0 (default) doubles it for tighter centerline tracking "
            "(important for reverse stability)."
        ),
    )
    args = parser.parse_args()

    e2e_rl_path = Path(args.e2e_rl_path).resolve()
    out_dir = Path(args.out_dir).resolve()

    _apply_lab_config_overrides(e2e_rl_path)
    _patch_env_vehicle_params(e2e_rl_path)
    _patch_path_generator(e2e_rl_path)
    # v16: simulate ROS-side actuator lag during training so the policy
    # learns to handle the phase delay that destabilised v15 in deployment.
    _patch_actuator_lag(e2e_rl_path, steer_tau=0.05, velocity_tau=0.10)
    if args.variable_speed:
        _patch_variable_speed_action(
            e2e_rl_path, v_min=args.v_min, v_max=args.v_max
        )
    # Per-episode velocity randomisation guards on self.fixed_speed
    # internally, so it's a no-op when variable_speed is on.
    _patch_velocity_randomisation(e2e_rl_path)
    if args.curvature_speed_weight > 0.0:
        _patch_curvature_speed_penalty(
            e2e_rl_path,
            beta=args.curvature_speed_weight,
            K=args.curvature_lookahead,
        )
    if args.target_speed_bonus > 0.0:
        _patch_target_speed_reward(
            e2e_rl_path,
            gamma=args.target_speed_bonus,
            sigma_low=args.target_speed_sigma_low,
            sigma_high=args.target_speed_sigma_high,
            v_target_min=args.target_speed_v_min,
            v_target_max=args.target_speed_v_max,
            K=args.curvature_lookahead,
            K_back=args.curvature_lookbehind,
            path_tightness=args.path_tightness,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(out_dir)
    print(f"[train_lab_model] cwd={out_dir}")
    print(f"[train_lab_model] e2e_rl={e2e_rl_path}")

    # Import only AFTER config overrides + patch are in place.
    import train as e2e_train

    e2e_train.main(
        scenario=args.scenario,
        obs="lidar",
        reward=args.reward,
        encoder="scratch",
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        lidar_beams=args.lidar_beams,
        device=args.device,
        eval_freq_timesteps=args.eval_freq,
        normalized_eval_freq_timesteps=args.normalized_eval_freq,
    )


if __name__ == "__main__":
    main()
