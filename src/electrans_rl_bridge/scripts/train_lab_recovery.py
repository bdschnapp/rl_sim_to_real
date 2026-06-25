#!/usr/bin/env python3
"""Fine-tune v15 reverse policy on near-jackknife recovery scenarios.

Architectural intent:
- v15's training distribution rarely visited |hitch| > 0.3 rad, so the
  policy has no learned response in the high-hitch region. When ROS
  deployment pushes the truck into that region (via accumulated error
  from delay / model mismatch / etc.), v15 outputs kinematically wrong
  steering and can't recover.
- This trainer adds a "recovery scenario" reset distribution: 25% of
  episodes (configurable) spawn the truck with steer + hitch perturbed
  into the same-sign large-magnitude region (≈±20° steer, ≈±40° hitch),
  on a straight path. The policy gets dense gradient signal for the
  recovery task, learning to steer toward the trailer (the kinematically
  correct counter-jackknife action).
- Other 75% of episodes use v15's existing path/velocity distribution
  unchanged, so the policy doesn't forget normal driving.

Reuses `train_lab_model`'s patches for everything except the new
recovery reset hook.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_E2E_RL = Path("/home/ben/Ben/Thesis/e2e_rl")
DEFAULT_INIT_ZIP = (
    REPO_ROOT / "lab_models_v15/models/reverse/lidar_24/multiplicative/best_model.zip"
)
DEFAULT_OUT = REPO_ROOT / "lab_models_v15oos"
SCRIPTS_DIR = REPO_ROOT / "src/electrans_rl_bridge/scripts"

sys.path.insert(0, str(SCRIPTS_DIR))


def _patch_recovery_scenarios(
    e2e_rl_path: Path,
    *,
    recovery_probability: float = 0.25,
    # Mode A ("stabilize tractor compared to trailer"): trailer axle is
    # placed on the centerline aligned with the path; the TRACTOR is
    # twisted by bad_hitch. The trailer (the larger body) is in-corridor;
    # the smaller tractor can swing without leaving the lane.
    mode_a_steer_low: float = 0.30, mode_a_steer_high: float = 0.40,
    mode_a_hitch_low: float = 0.60, mode_a_hitch_high: float = 0.80,
    # Mode B ("stabilize trailer compared to path"): tractor is on the
    # centerline aligned with the path; the TRAILER swings off-axis.
    # Hitch magnitude must be smaller here so the trailer axle stays in
    # the 2.82 m corridor — L*sin(β) ≈ 0.96 m at β=0.5 rad keeps the
    # trailer just inside (half-width 1.41 m, less trailer half-width
    # 0.325 m → 1.08 m room).
    mode_b_steer_low: float = 0.30, mode_b_steer_high: float = 0.40,
    mode_b_hitch_low: float = 0.35, mode_b_hitch_high: float = 0.50,
    # Per-recovery-episode split between Modes A and B.
    mode_a_probability: float = 0.5,
    # Mode B lateral offset (meters): shift the entire setup laterally
    # AWAY from the trailer's natural drift direction, giving the trailer
    # clearance from the corridor edge. Required at large hitch magnitudes
    # to prevent immediate kinematic collision before the policy can react.
    mode_b_lateral_offset: float = 0.0,
    init_speed_low: float = 0.5, init_speed_high: float = 1.0,  # |m/s| in reverse
    force_recovery: bool = False,
    # Used by the eval visualiser to lock to a single mode (None = mix).
    force_mode: str = None,
):
    """Monkey-patch the reverse env's `reset()` to sometimes spawn the
    truck in a near-jackknife state on a forced-straight path.

    Sign sampling: the kinematic-failure mode in reverse driving is
    `sign(δ) != sign(β)` — i.e., steering pushing the truck INTO the
    jackknife while the trailer is already swinging out. (Recall
    dβ/dt = (v/L_truck)·tan(δ) - (v/L_trailer)·sin(β); with v<0, the
    truck-driven term needs sign(δ)=sign(β) to shrink |β|. Opposite
    signs grow |β|.) We sample steer-sign and hitch-sign INDEPENDENTLY
    so the four combinations (++, +-, -+, --) each get ≈25% of
    recovery episodes — the two opposite-sign combos are the failure
    state we most need the policy to learn to recover from, and the
    two same-sign combos are useful generalization (the steering is
    already correct, the policy must hold direction while β decays).

    Two recovery spawn modes per recovery episode:
        - Mode A: trailer axle on centerline aligned with path, tractor
          twisted (bigger hitch ~40°). Teaches the policy to bring the
          tractor in line with the trailer.
        - Mode B: tractor on centerline aligned with path, trailer
          swung off-axis (smaller hitch ~20-29° so trailer stays in
          corridor). Teaches the policy to bring the trailer back onto
          the path.

    Args:
        recovery_probability: fraction of episodes that get the recovery
            perturbation. Other episodes use the unchanged v15 distribution.
        mode_a_* / mode_b_*: per-mode magnitude bounds (sign sampled per ep).
        mode_a_probability: per-recovery-episode split between A and B.
        init_speed_low/high: |xd| range for recovery-scenario start.
        force_recovery: if True, every episode is a recovery scenario.
            Used by the eval script to visualise the setup.
        force_mode: "A" or "B" to lock to a single mode (used by eval to
            visualise each mode separately). None for the normal mix.
    """
    if str(e2e_rl_path) not in sys.path:
        sys.path.insert(0, str(e2e_rl_path))

    import Environments.LineFollowing as lf
    from e2erl_utils import config as c

    # Idempotent re-patch support: stash the TRUE original reset on first
    # patch; subsequent calls use that, so curriculum phases can re-patch
    # with new bounds without nesting closures.
    if not hasattr(lf.ReverseStateObservationLineFollowingEnv,
                   "_recovery_orig_reset"):
        lf.ReverseStateObservationLineFollowingEnv._recovery_orig_reset = (
            lf.ReverseStateObservationLineFollowingEnv.reset
        )
    orig_reset = lf.ReverseStateObservationLineFollowingEnv._recovery_orig_reset

    # Constant from the lab path generator (`train_lab_model._patch_path_generator`)
    VERT_OFFSET = 45.0

    def patched_reset(self, seed=None, options=None):
        # Run standard reset first — gets a random path + vehicle placement
        obs, info = orig_reset(self, seed=seed, options=options)

        is_recovery = force_recovery or (self.np_random.random() < recovery_probability)
        if not is_recovery:
            if isinstance(info, dict):
                info["recovery_scenario"] = False
            return obs, info

        # ---- Override the path to a clean straight line ----
        # Match the same x-range and offset as the lab path generator so
        # the env's projections, lidar, etc. stay consistent. Use float64
        # arrays (not float32) so the env's path-error projections stay in
        # the same precision as the un-patched generator (the float32 cast
        # in earlier versions interacted badly with the env's downstream
        # arctan2 / atan2 calls when the path was effectively constant in y
        # — producing the occasional NaN that broke pygame rendering).
        x0 = float(self.xx[0])
        x_end = float(self.xx[-1])
        self.xx = np.arange(int(x0), int(x_end), 1, dtype=np.float64)
        self.yy = np.full_like(self.xx, VERT_OFFSET, dtype=np.float64)
        self._path_kind = "straight"

        # Pick path point at 10% along (new) straight path. Force plain
        # Python floats so nothing downstream sees a numpy scalar (pygame
        # builds occasionally choke on np.float64 inside polygon points).
        x_p, y_p, yaw_p = self.get_point_on_path(0.10)
        x_p = float(x_p); y_p = float(y_p); yaw_p = float(yaw_p)

        # Choose recovery mode + per-mode magnitudes.
        if force_mode is not None:
            mode = str(force_mode).upper()
        else:
            mode = "A" if self.np_random.random() < mode_a_probability else "B"
        if mode == "A":
            steer_lo, steer_hi = mode_a_steer_low, mode_a_steer_high
            hitch_lo, hitch_hi = mode_a_hitch_low, mode_a_hitch_high
        else:
            steer_lo, steer_hi = mode_b_steer_low, mode_b_steer_high
            hitch_lo, hitch_hi = mode_b_hitch_low, mode_b_hitch_high

        # Independent signs: opposite-sign(δ, β) is the *failure* state
        # the policy can't escape; same-sign is the already-recovering
        # state. Independent sampling visits all 4 combinations evenly.
        steer_sign = 1.0 if self.np_random.random() < 0.5 else -1.0
        hitch_sign = 1.0 if self.np_random.random() < 0.5 else -1.0
        bad_steer = steer_sign * float(self.np_random.uniform(steer_lo, steer_hi))
        bad_hitch = hitch_sign * float(self.np_random.uniform(hitch_lo, hitch_hi))

        lr = float(self.vehicle.lr)
        L = float(self.vehicle.trailer.L)
        path_yaw_rev = yaw_p + float(np.pi)  # reverse-driving convention

        if mode == "A":
            # --- Mode A: trailer aligned, tractor twisted ---
            # Place TRAILER axle on the centerline with trailer.yaw = path
            # reverse direction. The (small) tractor ends up off-centerline
            # but well inside the corridor since the larger body is the
            # reference.
            trailer_yaw = path_yaw_rev
            trailer_x = x_p
            trailer_y = y_p
            vehicle_p = trailer_yaw + bad_hitch       # β = vehicle.p - trailer.yaw
            hitch_x = trailer_x + L * float(np.cos(trailer_yaw))
            hitch_y = trailer_y + L * float(np.sin(trailer_yaw))
            vehicle_x = hitch_x + lr * float(np.cos(vehicle_p))
            vehicle_y = hitch_y + lr * float(np.sin(vehicle_p))
        else:
            # --- Mode B: tractor aligned, trailer twisted ---
            # Place TRACTOR on the centerline aligned with the path. The
            # trailer swings off-axis by L*sin(β); the mode_b_lateral_offset
            # shifts the entire setup laterally AWAY from the trailer's
            # natural drift direction so the trailer has clearance from
            # the corridor edge (otherwise the trailer hits the edge
            # within ~10 steps regardless of policy action).
            vehicle_p = path_yaw_rev
            # Perpendicular to path direction (left-of-path = +90° from path_yaw)
            perp_x = -float(np.sin(yaw_p))
            perp_y =  float(np.cos(yaw_p))
            # Shift sign: trailer drifts in -sign(hitch) direction, so we
            # shift the setup in +sign(hitch) direction along the perp axis.
            offset_sign = 1.0 if bad_hitch > 0 else -1.0
            shift_x = perp_x * offset_sign * float(mode_b_lateral_offset)
            shift_y = perp_y * offset_sign * float(mode_b_lateral_offset)
            vehicle_x = x_p + shift_x
            vehicle_y = y_p + shift_y
            trailer_yaw = vehicle_p - bad_hitch
            hitch_x = vehicle_x - lr * float(np.cos(vehicle_p))
            hitch_y = vehicle_y - lr * float(np.sin(vehicle_p))
            trailer_x = hitch_x - L * float(np.cos(trailer_yaw))
            trailer_y = hitch_y - L * float(np.sin(trailer_yaw))

        # Reset vehicle at the chosen tractor pose
        self.vehicle.reset(float(c.initial_xd), x=vehicle_x, y=vehicle_y,
                           p=vehicle_p)

        # Apply steering immediately AND set the lag-target accumulator
        # (if patch_vehicle_with_delay is active) so the first lag tick
        # doesn't immediately drag self.s back to zero.
        self.vehicle.s = bad_steer
        if hasattr(self.vehicle, "_s_target"):
            self.vehicle._s_target = bad_steer

        # Override trailer pose (vehicle.reset re-aligned it; we want our
        # mode-specific pose).
        self.vehicle.trailer.x = trailer_x
        self.vehicle.trailer.y = trailer_y
        self.vehicle.trailer.yaw = trailer_yaw
        if hasattr(self.vehicle.trailer, "yaw_rate"):
            self.vehicle.trailer.yaw_rate = 0.0

        # Slow initial velocity (in reverse) so the trailer doesn't fully
        # jackknife before the policy has a chance to react.
        init_speed = float(self.np_random.uniform(init_speed_low, init_speed_high))
        self.vehicle.xd = -init_speed  # reverse → negative
        if hasattr(self.vehicle, "_xd_target"):
            self.vehicle._xd_target = -init_speed

        # Rebuild occupancy grid so lidar reflects the new straight path
        if hasattr(self, "_build_occupancy_grid"):
            try:
                self._build_occupancy_grid()
            except Exception:
                pass

        # Re-fetch observation with all the perturbed state in place.
        # Coerce every vehicle attribute to a plain Python float — pygame
        # renderers can choke on stray numpy scalars in some versions.
        self.vehicle.x = float(self.vehicle.x)
        self.vehicle.y = float(self.vehicle.y)
        self.vehicle.p = float(self.vehicle.p)
        self.vehicle.s = float(self.vehicle.s)
        self.vehicle.xd = float(self.vehicle.xd)
        self.vehicle.trailer.x = float(self.vehicle.trailer.x)
        self.vehicle.trailer.y = float(self.vehicle.trailer.y)
        self.vehicle.trailer.yaw = float(self.vehicle.trailer.yaw)
        obs = self._get_obs()
        if isinstance(info, dict):
            info["recovery_scenario"] = True
            info["recovery_mode"] = mode
            info["recovery_steer"] = bad_steer
            info["recovery_hitch"] = bad_hitch
        return obs, info

    lf.ReverseStateObservationLineFollowingEnv.reset = patched_reset


def _build_env(e2e_rl_path: Path, *, lidar_beams: int, reward_mode: str,
               max_episode_steps: int = 1000):
    """Construct the v15-style reverse env (flat 32-dim Box obs, 2-D
    variable-speed action). Identical to v15's training env."""
    import Environments.LineFollowing as lf
    return lf.ReverseLidarStateObservationLineFollowingEnv(
        render_mode=None,
        max_episode_steps=max_episode_steps,
        lidar_beams=lidar_beams,
        reward_mode=reward_mode,
        fixed_speed=False,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-from-zip", default=str(DEFAULT_INIT_ZIP),
                    help="v15 checkpoint to fine-tune from")
    ap.add_argument("--timesteps", type=int, default=75_000)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--lidar-beams", type=int, default=24)
    ap.add_argument("--reward", default="multiplicative")
    ap.add_argument("--recovery-probability", type=float, default=0.25,
                    help="Fraction of episodes spawned in near-jackknife state")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--eval-freq", type=int, default=5_000)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--e2e-rl-path", default=str(DEFAULT_E2E_RL))
    args = ap.parse_args()

    e2e_rl_path = Path(args.e2e_rl_path).resolve()
    out_dir = Path(args.out_dir).resolve()
    save_dir = out_dir / "models/reverse/lidar_24/multiplicative"
    logs_dir = save_dir / "logs"
    save_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # ---- Apply patches (same as v15) + the new recovery patch ----
    sys.path.insert(0, str(e2e_rl_path))
    import train_lab_model as tlm
    tlm._apply_lab_config_overrides(e2e_rl_path)
    tlm._patch_env_vehicle_params(e2e_rl_path)
    tlm._patch_path_generator(e2e_rl_path)
    tlm._patch_variable_speed_action(e2e_rl_path, v_min=0.5, v_max=3.0)
    tlm._patch_velocity_randomisation(e2e_rl_path)
    _patch_recovery_scenarios(
        e2e_rl_path,
        recovery_probability=args.recovery_probability,
        force_recovery=False,
    )
    print(f"[v15oos] patches applied. recovery_probability={args.recovery_probability}")

    # ---- Build train + eval envs ----
    from stable_baselines3 import TD3
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    from stable_baselines3.common.monitor import Monitor

    def make_env_fn():
        def _f():
            return Monitor(_build_env(e2e_rl_path,
                                      lidar_beams=args.lidar_beams,
                                      reward_mode=args.reward))
        return _f

    if args.n_envs > 1:
        import multiprocessing as mp
        try: mp.set_start_method("fork", force=True)
        except RuntimeError: pass
        train_env = SubprocVecEnv([make_env_fn() for _ in range(args.n_envs)])
    else:
        train_env = DummyVecEnv([make_env_fn()])
    eval_env = DummyVecEnv([make_env_fn()])

    # ---- Build TD3 model + load v15 weights ----
    model = TD3(
        policy="MlpPolicy",
        env=train_env,
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=3e-4,
        buffer_size=200_000,
        batch_size=256,
        tau=0.005, gamma=0.99,
        train_freq=1, gradient_steps=1,
        policy_delay=2,
        target_noise_clip=0.5, target_policy_noise=0.2,
        device=args.device,
        verbose=1,
    )

    init_zip = Path(args.init_from_zip).resolve()
    if init_zip.exists():
        print(f"[v15oos] warm-starting policy weights from {init_zip}")
        import zipfile, io
        import torch as th
        with zipfile.ZipFile(str(init_zip)) as z:
            with z.open("policy.pth") as f:
                sd = th.load(io.BytesIO(f.read()),
                             map_location=model.device, weights_only=False)
        model.policy.load_state_dict(sd)
    else:
        print(f"[v15oos] WARN: --init-from-zip not found ({init_zip}); training from scratch")

    # ---- Use the RewardPerStepEvalCallback so model selection is by ----
    # ---- reward/step (not total reward — see v17_delay_aware comment) ----
    import v17_delay_aware as v17  # for RewardPerStepEvalCallback
    eval_callback = v17.RewardPerStepEvalCallback(
        eval_env,
        best_model_save_path=str(save_dir),
        log_path=str(logs_dir),
        eval_freq=max(args.eval_freq // max(1, args.n_envs), 1),
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )

    print(f"[v15oos] training {args.timesteps} steps → {save_dir}")
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=eval_callback)
    model.save(str(save_dir / "final.zip"))
    print(f"[v15oos] done in {(time.time()-t0)/60:.1f} min")
    print(f"[v15oos] best_model.zip + final.zip saved under {save_dir}")


if __name__ == "__main__":
    main()
