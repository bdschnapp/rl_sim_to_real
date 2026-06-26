#!/usr/bin/env python3
"""
Evaluate a lab-scale TD3 checkpoint inside the e2e_rl env with rendering.

Applies the same e2erl_utils.config + TractorTrailerEnv monkey-patches as
scripts/train_lab_model.py so the env matches the lab-scale geometry the
policy was trained against. Defaults to the lab-trained best_model.zip
under <repo>/lab_models. Pass --model <path> to evaluate a different
checkpoint (e.g. compare the semi-truck-trained checkpoint against the
lab-trained one).

Usage
-----
    # Watch the lab-trained forward policy
    python eval_lab_model.py --scenario forward

    # Watch the lab-trained reverse policy
    python eval_lab_model.py --scenario reverse

    # Compare: same env, but the OLD semi-truck checkpoint
    python eval_lab_model.py --scenario reverse \\
        --model /home/ben/Ben/Thesis/e2e_rl/models/reverse/lidar_24/multiplicative/best_model.zip

    # No window, just numbers (useful in a tmux pane or for batch metrics)
    python eval_lab_model.py --scenario forward --no-render --episodes 20
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_E2E_RL = Path("/home/ben/Ben/Thesis/e2e_rl")
DEFAULT_LAB_MODELS = REPO_ROOT / "lab_models_v18" / "models"


def _apply_variable_speed(
    e2e_rl_path: Path,
    *,
    v_min: float = 0.1,
    v_max: float = 5.0,
) -> None:
    """Mirror train_lab_model._patch_variable_speed_action so that the
    eval env actually consumes the 2-D action a variable-speed policy
    produces. Without this, format_action drops action[1] and uses
    self.fixed_speed_command (the training-default 5 m/s) regardless
    of what the policy chose — making the velocity head invisible
    during eval."""
    if str(e2e_rl_path) not in sys.path:
        sys.path.insert(0, str(e2e_rl_path))

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

    orig_reverse_init = lf.ReverseLidarStateObservationLineFollowingEnv.__init__

    def reverse_init(self, render_mode="human", max_episode_steps=1000,
                     lidar_beams=16, reward_mode: str = "dense",
                     fixed_speed: bool = True):  # noqa: ARG001
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


def _apply_path_override(e2e_rl_path: Path, path_type: str) -> None:
    """Monkey-patch LineFollowingEnv.generate_path so every reset spawns a
    deterministic path of a chosen type. Used to evaluate a policy against
    specific failure modes (e.g. the MVSL 90° corner) without waiting for
    the training-time mixture distribution to roll one.

    Path types
    ----------
    "mix"             — 5/20/25/50 (straight/gentle/sharp/winding), the
                        v15 training distribution.
    "straight"        — flat y, every episode.
    "gentle"          — cubic spline through 6 random control points
                        (∓5 m vertical range).
    "sharp"           — chained tanh bends: 4–8 alternating-sign ramps,
                        width 1.0–1.5 m, dy 2.5–4 m. Peak κ ≈ 0.30–0.45 m⁻¹.
    "winding"         — chained alternating-sign sustained turns: 5
                        bends with slope ±0.6, width 1.0. Active training
                        path for v15+ — vehicle is densely turning
                        throughout, no long pre-bend straight runway.
                        Peak κ ≈ 0.6.
    "90"              — alias for "winding" (the corner test). The
                        legacy single-bend "90" had a 95 m pre-bend
                        straight from the canvas-fit constraint that
                        made it a poor diagnostic.
    "sustained_turn"  — alias for "winding". Same reasoning. If you
                        specifically want the legacy single-bend
                        long-straight form, ask for it explicitly.
    """
    if str(e2e_rl_path) not in sys.path:
        sys.path.insert(0, str(e2e_rl_path))

    import numpy as np
    from scipy.interpolate import CubicSpline
    import Environments.LineFollowing as lf
    import Environments.TractorTrailer as tt

    def make_path_type(self, kind: str):
        world_width_m = tt.WINDOW_WIDTH * tt.METERS_PER_PIXEL
        x0 = 5.0
        x_end = world_width_m - 5.0
        self.xx = np.arange(int(x0), int(x_end), 1)
        n_pts = len(self.xx)

        if kind == "straight":
            return np.zeros(n_pts, dtype=float)
        if kind == "gentle":
            ctrl_x = np.linspace(x0, x_end, 6)
            ctrl_y = -1.0 + 2.0 * self.np_random.random(6)
            ctrl_y[0] = (-0.25 / 50) + self.np_random.random() * (0.5 / 50)
            cs = CubicSpline(ctrl_x, ctrl_y, bc_type=((1, 0.0), "not-a-knot"))
            return cs(self.xx) * 5.0
        if kind == "sharp":
            # Mirror train_lab_model._patch_path_generator's "sharp"
            # branch EXACTLY — v15 params: width 1.0–1.5 m, dy 2.5–4 m,
            # κ_max ≈ 0.30–0.45. Older eval versions had v13-era
            # params (width 1.5–3, dy 3–8) which produced gentler κ ≈
            # 0.05–0.24 — wrong for testing the current policy.
            n_bends = int(self.np_random.integers(4, 9))
            margin = 20.0
            base_centers = np.linspace(x0 + margin, x_end - margin, n_bends)
            gap = (x_end - x0 - 2 * margin) / max(1, n_bends - 1)
            jitter = (self.np_random.random(n_bends) - 0.5) * gap * 0.4
            bend_xs = sorted((base_centers + jitter).tolist())
            sign = 1.0 if self.np_random.random() < 0.5 else -1.0
            y = np.zeros(n_pts, dtype=float)
            for bend_x in bend_xs:
                width = float(self.np_random.uniform(1.0, 1.5))
                dy = sign * float(self.np_random.uniform(2.5, 4.0))
                sign = -sign
                ramp = (np.tanh((self.xx - bend_x) / width) + 1.0) * 0.5
                y += dy * ramp
            return y
        if kind == "lab_corner":
            # S-curve: two opposing 90° bends with a LONG straight
            # between them (≥100 m). Mirrors train_lab_model v18 setup.
            sign = 1.0 if self.np_random.random() < 0.5 else -1.0
            bend1_x = float(self.np_random.uniform(x0 + 10.0, x0 + 15.0))
            width1 = float(self.np_random.uniform(0.3, 0.8))
            dy1 = sign * float(self.np_random.uniform(2.0, 6.0))
            bend2_x = float(self.np_random.uniform(bend1_x + 100.0, x_end - 10.0))
            width2 = float(self.np_random.uniform(0.3, 0.8))
            dy2 = -dy1
            ramp1 = (np.tanh((self.xx - bend1_x) / width1) + 1.0) * 0.5
            ramp2 = (np.tanh((self.xx - bend2_x) / width2) + 1.0) * 0.5
            return dy1 * ramp1 + dy2 * ramp2
        if kind == "lab_seam":
            bend_x = float(self.np_random.uniform(x0 + 6.0, x_end - 4.0))
            width = float(self.np_random.uniform(0.4, 0.7))
            sign = 1.0 if self.np_random.random() < 0.5 else -1.0
            dy = sign * float(self.np_random.uniform(1.5, 3.0))
            ramp = (np.tanh((self.xx - bend_x) / width) + 1.0) * 0.5
            return dy * ramp
        if kind in ("winding", "90", "sustained_turn"):
            # Chained alternating-sign sustained turns — matches the v15+
            # training "winding" mode. Deterministic params for eval:
            # 5 bends, s=0.6, w=1.0, initial_sign=+1.
            n_bends = 5
            s = 0.6
            width = 1.0
            initial_sign = 1.0
            margin = 18.0
            base_xs = np.linspace(x0 + margin, x_end - margin, n_bends)
            bend_xs = base_xs.tolist()
            interval_slopes = [0.0]
            for i in range(n_bends - 1):
                sign = initial_sign if (i % 2 == 0) else -initial_sign
                interval_slopes.append(sign * s)
            interval_slopes.append(0.0)
            y = np.zeros(n_pts, dtype=float)
            for bend_x, sp, sn in zip(bend_xs, interval_slopes[:-1], interval_slopes[1:]):
                delta = sn - sp
                if abs(delta) < 1e-9:
                    continue
                arg = (self.xx - bend_x) / width
                arg_0 = (x0 - bend_x) / width
                ln_cosh = np.log(np.cosh(arg))
                ln_cosh_0 = float(np.log(np.cosh(arg_0)))
                y += (delta / 2.0) * (
                    (self.xx - x0) + width * (ln_cosh - ln_cosh_0)
                )
            return y
        raise ValueError(f"unknown path kind: {kind!r}")

    def generate_path(self):
        if path_type == "mix":
            kind = self.np_random.choice(
                ["straight", "gentle", "sharp", "winding"],
                p=[0.05, 0.20, 0.25, 0.50],
            )
        else:
            kind = path_type
        y_local = make_path_type(self, kind)

        self.yy = y_local + 45.0
        x_g = self.xx[-1]
        y_g = self.yy[-1]
        yaw_g = float(
            np.arctan2(self.yy[-1] - self.yy[-2], self.xx[-1] - self.xx[-2])
        )
        self.goal_pose = (x_g, y_g, yaw_g)

    lf.LineFollowingEnv.generate_path = generate_path


def _apply_lab_overrides(
    e2e_rl_path: Path,
    *,
    trailer_length: float,
    trailer_width: float,
    lane_half_width: float,
    initial_xd: float | None,
) -> None:
    """Mirror train_lab_model._apply_lab_config_overrides, but with values
    that can be overridden from the CLI so you can probe sensitivity. e.g.
    set initial_xd=0.6 to evaluate the policy at the bridge's deployment
    speed instead of the training-default 5 m/s."""
    if str(e2e_rl_path) not in sys.path:
        sys.path.insert(0, str(e2e_rl_path))

    from e2erl_utils import config as c

    c.tractor_length_m = 1.0
    c.tractor_width_m = 0.65
    c.trailer_length_m = trailer_length
    c.trailer_width_m = trailer_width
    c.lane_centerline_half_width_m = lane_half_width
    c.lane_shoulder_m = 0.20
    if initial_xd is not None:
        c.initial_xd = initial_xd
    c.tesla_model_s_vehicle_params = dict(
        c.tesla_model_s_vehicle_params,
        lf=0.33,
        lr=0.32,
    )

    import Environments.TractorTrailer as tt
    from VehicleModels.tractor_trailer import StateSpaceTractorTrailer

    original_init = tt.TractorTrailerEnv.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.vehicle = StateSpaceTractorTrailer(
            args=dict(c.tesla_model_s_vehicle_params),
            trailer_length=c.trailer_length_m,
        )

    tt.TractorTrailerEnv.__init__ = patched_init


def _default_model_path(scenario: str, lidar_beams: int, reward: str) -> Path:
    return (
        DEFAULT_LAB_MODELS
        / scenario
        / f"lidar_{lidar_beams}"
        / reward
        / "best_model.zip"
    )


def _run_tractor_only_eval(
    *,
    model_path: Path,
    scenario: str,
    reward: str,
    lidar_beams: int,
    n_episodes: int,
    render: bool,
) -> None:
    """Evaluate a tractor-only (no-trailer) policy. run_model.run_rl_model builds
    the trailer env (32-dim obs), incompatible with the 29-dim tractor-only
    policy, so we build Environments.TractorOnly directly and reuse e2e_rl's
    generic _run_episodes loop. The --variable-speed patch (applied in main) hits
    the base LidarState classes TractorOnly subclasses, so the 2-D action space
    carries over transparently."""
    import importlib
    from stable_baselines3 import TD3
    from run_model import _run_episodes

    cls_name = (
        "TractorOnlyLidarStateLineFollowingEnv" if scenario == "forward"
        else "ReverseTractorOnlyLidarStateLineFollowingEnv"
    )
    env_cls = getattr(importlib.import_module("Environments.TractorOnly"), cls_name)
    env = env_cls(
        render_mode="human" if render else None,
        max_episode_steps=1000,
        lidar_beams=lidar_beams,
        reward_mode=reward,
    )
    model = TD3.load(str(model_path), device="auto")
    _run_episodes(
        env,
        lambda obs: model.predict(obs, deterministic=True)[0],
        n_episodes,
        render,
        f"{scenario}/tractor-only",
    )
    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scenario", choices=["forward", "reverse"], required=True)
    parser.add_argument(
        "--tractor-only", dest="tractor_only", action="store_true",
        help=(
            "Evaluate a TRACTOR-ONLY (no-trailer) policy on the "
            "Environments.TractorOnly env (29-dim obs = 5 state + lidar) "
            "instead of the trailer env. Default model dir when --model is "
            "omitted: <repo>/lab_models_tractor_only/."
        ),
    )
    parser.add_argument(
        "--model", default=None,
        help=(
            "Path to a .zip checkpoint. Default: "
            "<repo>/lab_models/models/<scenario>/lidar_<beams>/<reward>/best_model.zip"
        ),
    )
    parser.add_argument("--reward", default="multiplicative")
    parser.add_argument("--lidar-beams", dest="lidar_beams", type=int, default=24)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--no-render", dest="render", action="store_false",
        help="Skip the pygame window (default: render).",
    )
    parser.add_argument(
        "--trailer-length", dest="trailer_length", type=float, default=2.8,
        help="Trailer length (m) used by pygame + obs pipeline (default: 2.8, the real AgileX trailer).",
    )
    parser.add_argument(
        "--trailer-width", dest="trailer_width", type=float, default=0.65,
        help="Trailer width (m) (default: 0.65, matches AgileX truck width).",
    )
    parser.add_argument(
        "--lane-half-width", dest="lane_half_width", type=float, default=1.41,
        help="Half lane width (m) used for obs + occupancy grid (default: 1.41).",
    )
    parser.add_argument(
        "--initial-xd", dest="initial_xd", type=float, default=None,
        help=(
            "Override training-default initial speed (5 m/s). Set 0.6 to "
            "match the bridge's deployment speed and check whether a "
            "policy that works in pygame transfers."
        ),
    )
    parser.add_argument(
        "--path-type", dest="path_type", default="mix",
        choices=["mix", "straight", "gentle", "sharp", "90", "sustained_turn",
                 "winding", "lab_seam", "lab_corner"],
        help=(
            "Force every episode to spawn a specific path geometry. "
            "'mix' (default) matches training. 'lab_corner' is the sharp "
            "single-bend ramp matching the v16 training-time MVSL corner "
            "replica (width 0.3-0.6 m, dy 3-6 m, tangent peak 70-85°). "
            "'lab_seam' is the milder v15 variant. See _apply_path_override "
            "docstring for details."
        ),
    )
    parser.add_argument(
        "--target-speed-bonus", dest="target_speed_bonus", type=float, default=0.0,
        help=(
            "Mirror the training-time Gaussian target-speed reward in "
            "the eval env so on-screen rewards reflect what the v8 "
            "policy was optimising (default 0 = use source-level "
            "multiplicative reward with monotonic clip(ẋ,0,1) factor). "
            "Try 4.0 to match v8 training. Doesn't change policy "
            "behaviour — only the reward telemetry."
        ),
    )
    parser.add_argument(
        "--target-speed-sigma-low", dest="target_speed_sigma_low", type=float, default=0.2,
    )
    parser.add_argument(
        "--target-speed-sigma-high", dest="target_speed_sigma_high", type=float, default=0.5,
    )
    parser.add_argument(
        "--target-speed-v-min", dest="target_speed_v_min", type=float, default=0.6,
    )
    parser.add_argument(
        "--target-speed-v-max", dest="target_speed_v_max", type=float, default=2.0,
    )
    parser.add_argument(
        "--curvature-lookahead-samples", dest="curvature_lookahead",
        type=int, default=10,
    )
    parser.add_argument(
        "--curvature-lookbehind-samples", dest="curvature_lookbehind",
        type=int, default=5,
    )
    parser.add_argument(
        "--path-tightness", dest="path_tightness", type=float, default=2.0,
    )
    parser.add_argument(
        "--log-velocity", dest="log_velocity", action="store_true",
        help=(
            "Run a single instrumented episode after the standard eval, "
            "recording per-step (vehicle.xd, action_velocity, κ̂, v_target) "
            "and printing a summary table. Diagnoses 'is the policy actually "
            "crawling, or does it just look slow on the long path?'"
        ),
    )
    parser.add_argument(
        "--variable-speed", dest="variable_speed", action="store_true",
        help=(
            "Make the eval env consume a 2-D [steer_rate, velocity] "
            "action. Required for v6+ checkpoints that were trained "
            "variable-speed — otherwise format_action drops the velocity "
            "and uses the training-default 5 m/s, making the eval "
            "useless. Default off for backwards compatibility with "
            "older fixed-speed checkpoints."
        ),
    )
    parser.add_argument(
        "--v-min", dest="v_min", type=float, default=0.5,
        help="Lower velocity bound for --variable-speed eval (default 0.5).",
    )
    parser.add_argument(
        "--v-max", dest="v_max", type=float, default=3.0,
        help="Upper velocity bound for --variable-speed eval (default 3.0).",
    )
    parser.add_argument(
        "--stop-signal", dest="stop_signal", action="store_true",
        help=(
            "Evaluate a CONSTANT-SPEED + STOP-SIGNAL policy: 2-D action "
            "[steer_rate, stop_signal], speed held constant; stop_signal > "
            "--stop-threshold ends the episode (penalised in no-obstacle envs). "
            "Mutually exclusive with --variable-speed."
        ),
    )
    parser.add_argument("--stop-threshold", dest="stop_threshold", type=float, default=0.0)
    parser.add_argument("--stop-penalty", dest="stop_penalty", type=float, default=200.0)
    parser.add_argument(
        "--e2e-rl-path", dest="e2e_rl_path", default=str(DEFAULT_E2E_RL),
    )
    parser.set_defaults(render=True)
    args = parser.parse_args()

    e2e_rl_path = Path(args.e2e_rl_path).resolve()
    _apply_lab_overrides(
        e2e_rl_path,
        trailer_length=args.trailer_length,
        trailer_width=args.trailer_width,
        lane_half_width=args.lane_half_width,
        initial_xd=args.initial_xd,
    )
    if args.variable_speed and args.stop_signal:
        parser.error("--variable-speed and --stop-signal are mutually exclusive.")
    if args.variable_speed:
        _apply_variable_speed(e2e_rl_path, v_min=args.v_min, v_max=args.v_max)
    elif args.stop_signal:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from stop_signal_patch import patch_stop_signal_action
        patch_stop_signal_action(
            e2e_rl_path, threshold=args.stop_threshold, stop_penalty=args.stop_penalty,
        )
    if args.target_speed_bonus > 0.0:
        # Reuse the training-time patch so the eval env's reward exactly
        # matches what v8+ was optimising. Doesn't affect policy actions.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from train_lab_model import _patch_target_speed_reward
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
    _apply_path_override(e2e_rl_path, args.path_type)

    if args.model:
        model_path = Path(args.model)
    elif args.tractor_only:
        obs_tag = f"lidar_{args.lidar_beams}" if args.lidar_beams != 16 else "lidar"
        model_path = (REPO_ROOT / "lab_models_tractor_only" / "models" /
                      args.scenario / obs_tag / args.reward / "best_model.zip")
    else:
        model_path = _default_model_path(
            args.scenario, args.lidar_beams, args.reward,
        )
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # run_model writes its print() output relative to CWD; nothing important
    # lands on disk for a render-only run, but chdir to lab_models so any
    # ./models/... lookup inside e2e_rl resolves cleanly.
    os.chdir(REPO_ROOT / "lab_models_v18")

    print(f"[eval_lab_model] model = {model_path}")
    print(f"[eval_lab_model] e2e_rl = {e2e_rl_path}")
    print(f"[eval_lab_model] scenario={args.scenario}  reward={args.reward}  "
          f"lidar_beams={args.lidar_beams}  episodes={args.episodes}  "
          f"render={args.render}")

    if args.tractor_only:
        _run_tractor_only_eval(
            model_path=model_path,
            scenario=args.scenario,
            reward=args.reward,
            lidar_beams=args.lidar_beams,
            n_episodes=args.episodes,
            render=args.render,
        )
    else:
        from run_model import run_rl_model

        run_rl_model(
            model_path=model_path,
            scenario=args.scenario,
            obs="lidar",
            reward=args.reward,
            encoder="scratch",
            encoder_path=None,
            lidar_beams=args.lidar_beams,
            n_episodes=args.episodes,
            render=args.render,
        )

    if args.log_velocity:
        _run_velocity_diagnostic(
            model_path=model_path,
            scenario=args.scenario,
            reward=args.reward,
            lidar_beams=args.lidar_beams,
            v_target_min=args.target_speed_v_min,
            v_target_max=args.target_speed_v_max,
        )


def _run_velocity_diagnostic(
    *,
    model_path: Path,
    scenario: str,
    reward: str,
    lidar_beams: int,
    v_target_min: float,
    v_target_max: float,
    kappa_max: float = 0.3,
    K: int = 10,
) -> None:
    """Run one headless episode of the loaded policy and report what
    velocities it actually chose, alongside the v_target it 'should' have
    been targeting at each step. Diagnoses 'is the policy actually
    crawling or just feels slow because the path is long'."""
    import numpy as np
    from stable_baselines3 import TD3
    from stable_baselines3.common.monitor import Monitor
    from run_model import make_env

    env = make_env(scenario, "lidar", render_mode=None, reward=reward,
                   lidar_beams=lidar_beams)
    env = Monitor(env)

    model = TD3.load(str(model_path), env=env, device="auto")

    obs, _ = env.reset()
    raw_env = env.env  # unwrap Monitor

    xds, kappas, v_targets, action_vs = [], [], [], []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc
        # vehicle.xd is body-frame; sign depends on env (forward positive,
        # reverse negative). Use magnitude for diagnostics.
        xds.append(abs(float(raw_env.vehicle.xd)))
        # Re-compute κ̂ same way the training reward did.
        try:
            dx = raw_env.xx - raw_env.vehicle.trailer.x
            dy = raw_env.yy - raw_env.vehicle.trailer.y
            nearest = int(np.argmin(dx * dx + dy * dy))
            n = len(raw_env.xx)
            start = max(1, nearest)
            end = min(nearest + K, n - 2)
            if end < start:
                k_hat = 0.0
            else:
                i = np.arange(start, end + 1)
                x_im1, x_i, x_ip1 = raw_env.xx[i-1], raw_env.xx[i], raw_env.xx[i+1]
                y_im1, y_i, y_ip1 = raw_env.yy[i-1], raw_env.yy[i], raw_env.yy[i+1]
                x_p = (x_ip1 - x_im1) * 0.5
                y_p = (y_ip1 - y_im1) * 0.5
                x_pp = x_ip1 - 2.0 * x_i + x_im1
                y_pp = y_ip1 - 2.0 * y_i + y_im1
                denom = (x_p ** 2 + y_p ** 2) ** 1.5 + 1e-6
                kappa = (x_p * y_pp - y_p * x_pp) / denom
                kappa = np.clip(kappa, -kappa_max, kappa_max)
                k_hat = float(np.max(np.abs(kappa)))
        except Exception:
            k_hat = float("nan")
        kappas.append(k_hat)
        kf = min(k_hat / kappa_max, 1.0) if not np.isnan(k_hat) else 0.0
        v_targets.append(v_target_max + (v_target_min - v_target_max) * kf)
        # Action[1] is the policy's commanded velocity (variable-speed mode).
        action_arr = np.asarray(action).flatten()
        action_vs.append(float(action_arr[1]) if action_arr.size >= 2 else float("nan"))

    xds = np.asarray(xds)
    kappas = np.asarray(kappas)
    v_targets = np.asarray(v_targets)
    action_vs = np.asarray(action_vs)

    print()
    print("=== velocity diagnostic (1 episode, deterministic) ===")
    print(f"  steps:       {len(xds)}")
    print(f"  vehicle.xd:  mean={xds.mean():.3f}  median={np.median(xds):.3f}  "
          f"min={xds.min():.3f}  max={xds.max():.3f}")
    print(f"  action_v:    mean={np.nanmean(action_vs):.3f}  "
          f"min={np.nanmin(action_vs):.3f}  max={np.nanmax(action_vs):.3f}")
    print(f"  κ̂ (lookahead {K}):  mean={kappas.mean():.3f}  max={kappas.max():.3f}")
    print(f"  v_target:    mean={v_targets.mean():.3f}  "
          f"range=[{v_targets.min():.3f}, {v_targets.max():.3f}]")
    print(f"  |xd - v_target|: mean={np.abs(xds - v_targets).mean():.3f}")

    # Sample some interesting timesteps
    sharp_idx = np.where(kappas > 0.15)[0]
    if len(sharp_idx) > 0:
        print(f"  on sharp (κ̂ > 0.15, {len(sharp_idx)} steps): "
              f"mean xd={xds[sharp_idx].mean():.3f}  "
              f"vs v_target={v_targets[sharp_idx].mean():.3f}")
    straight_idx = np.where(kappas < 0.05)[0]
    if len(straight_idx) > 0:
        print(f"  on straight (κ̂ < 0.05, {len(straight_idx)} steps): "
              f"mean xd={xds[straight_idx].mean():.3f}  "
              f"vs v_target={v_targets[straight_idx].mean():.3f}")

    env.close()


if __name__ == "__main__":
    main()
