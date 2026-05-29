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
DEFAULT_OUT = REPO_ROOT / "lab_models"


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
    c.trailer_width_m = 0.5

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
    args = parser.parse_args()

    e2e_rl_path = Path(args.e2e_rl_path).resolve()
    out_dir = Path(args.out_dir).resolve()

    _apply_lab_config_overrides(e2e_rl_path)
    _patch_env_vehicle_params(e2e_rl_path)

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
