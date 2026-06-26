#!/usr/bin/env python3
"""
Train a lab-scale TD3 policy for the TRACTOR-ONLY (no-trailer) env.

This mirrors train_lab_model.py but targets the new no-trailer environments in
``e2e_rl/Environments/TractorOnly.py``:

    TractorOnlyLidarStateLineFollowingEnv          (--scenario forward)
    ReverseTractorOnlyLidarStateLineFollowingEnv   (--scenario reverse)

Observation = 5 tractor-only state dims [s, e_y, e_ψ, κ₁, κ₂] + N lidar beams
(vs. the trailer envs' 8 state dims + N lidar). No hitch angle, no trailer
cross-track / heading error.

Why this is a standalone script rather than a thin wrapper over e2e_rl/train.py:
train.py's make_env() is a closed dispatcher keyed on (scenario, obs) that only
knows the trailer env classes, and we are NOT allowed to edit it. So this script
reproduces the small slice of train.py.main() that builds the TD3 model and the
eval callbacks, but instantiates the tractor-only env directly. Everything else
(AgileX lab config overrides, the path-generator patch, the variable-speed
action patch, the actuator-lag patch, the velocity-randomisation patch) is
imported and reused verbatim from train_lab_model.py — including the trailer
config values, which are harmless here (the env ignores the trailer) and keep
the rendering canvas / lane geometry identical to the trailer training runs.

Outputs (default --reward multiplicative, --lidar-beams 24):

  <out_dir>/models/<scenario>/lidar_24/multiplicative/best_model.zip
  <out_dir>/models/<scenario>/lidar_24/multiplicative/final.zip
  <out_dir>/models/<scenario>/lidar_24/multiplicative/logs/...

After training, run scripts/re_export_td3.py (with --reverse for the reverse
checkpoint) on best_model.zip — BUT note the bridge/adapter must be pointed at
the tractor-only env classes for the reduced obs dim; see the report.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Reuse all the lab patches from the sibling trailer trainer rather than
# re-implementing them. They live in the same scripts/ dir.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import train_lab_model as tlm  # noqa: E402  (after sys.path tweak)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_E2E_RL = Path("/home/ben/Ben/Thesis/e2e_rl")
DEFAULT_OUT = REPO_ROOT / "lab_models_tractor_only"


_SCENARIO_TO_ENV = {
    "forward": ("Environments.TractorOnly", "TractorOnlyLidarStateLineFollowingEnv"),
    "reverse": ("Environments.TractorOnly", "ReverseTractorOnlyLidarStateLineFollowingEnv"),
}


def _make_tractor_only_env(scenario: str, reward: str, lidar_beams: int,
                           render_mode=None):
    """Construct the tractor-only env for the given scenario. The variable-speed
    / actuator-lag / velocity-randomisation monkey-patches applied by
    train_lab_model patch the BASE classes
    (LidarStateObservationLineFollowingEnv /
    ReverseLidarStateObservationLineFollowingEnv) which our env subclasses, so
    they apply transparently here too."""
    import importlib

    mod_name, cls_name = _SCENARIO_TO_ENV[scenario]
    mod = importlib.import_module(mod_name)
    env_cls = getattr(mod, cls_name)
    return env_cls(
        render_mode=render_mode,
        max_episode_steps=1000,
        lidar_beams=lidar_beams,
        reward_mode=reward,
    )


def _train(scenario: str, reward: str, lidar_beams: int, timesteps: int,
           n_envs: int, device: str, eval_freq: int, normalized_eval_freq: int):
    """A trimmed copy of e2e_rl/train.py:main() for obs='lidar', MlpPolicy,
    using the tractor-only env. Writes into ./models/<scenario>/lidar_<beams>/
    <reward>/ relative to the (already chdir'd) out_dir."""
    import numpy as np
    from stable_baselines3 import TD3
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.noise import NormalActionNoise
    from stable_baselines3.common.vec_env import SubprocVecEnv

    import train as e2e_train  # for NormalizedEvalCallback + make_action_noise_sigma

    obs_tag = f"lidar_{lidar_beams}" if lidar_beams != 16 else "lidar"
    save_root = Path(f"./models/{scenario}/{obs_tag}/{reward}")
    save_root.mkdir(parents=True, exist_ok=True)
    log_dir = save_root / "logs"
    log_dir.mkdir(exist_ok=True)
    norm_dir = save_root / "normalized"
    norm_dir.mkdir(exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  TRACTOR-ONLY  scenario={scenario}  obs=lidar_{lidar_beams}  reward={reward}")
    print(f"  timesteps={timesteps:,}  n_envs={n_envs}  device={device}")
    print(f"  save → {save_root.resolve()}")
    print(f"{'=' * 60}")

    def _env_fn(rank: int):
        def _init():
            env = _make_tractor_only_env(scenario, reward, lidar_beams)
            env = Monitor(env)
            env.reset(seed=rank)
            return env
        return _init

    if n_envs > 1:
        train_env = SubprocVecEnv([_env_fn(i) for i in range(n_envs)],
                                  start_method="fork")
    else:
        train_env = Monitor(_make_tractor_only_env(scenario, reward, lidar_beams))

    eval_env = Monitor(_make_tractor_only_env(scenario, reward, lidar_beams))

    n_actions = train_env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=e2e_train.make_action_noise_sigma(n_actions),
    )
    model = TD3(
        "MlpPolicy",
        train_env,
        action_noise=action_noise,
        verbose=0,
        device=device,
        policy_kwargs=dict(net_arch=[256, 256]),
        buffer_size=300_000,
        learning_starts=5_000,
        batch_size=256,
        train_freq=(1, "step"),
        gradient_steps=-1,
        target_policy_noise=0.05,
        target_noise_clip=0.15,
    )

    cbs = [
        EvalCallback(
            eval_env,
            best_model_save_path=str(save_root),
            log_path=str(log_dir),
            eval_freq=max(eval_freq // n_envs, 1),
            n_eval_episodes=10,
            deterministic=True,
            render=False,
        ),
        e2e_train.NormalizedEvalCallback(
            eval_env,
            best_model_save_path=str(norm_dir),
            log_path=str(norm_dir / "logs"),
            eval_freq=max(normalized_eval_freq // n_envs, 1),
            n_eval_episodes=10,
            deterministic=True,
        ),
    ]

    model.learn(total_timesteps=timesteps, callback=cbs, progress_bar=True)
    model.save(str(save_root / "final"))
    print(f"  Saved final model → {save_root / 'final.zip'}")

    train_env.close()
    eval_env.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Lab-scale TD3 trainer for the TRACTOR-ONLY (no-trailer) env. "
            "Same AgileX lab config + patches as train_lab_model.py, but the "
            "observation/reward carry no trailer or hitch terms."
        )
    )
    parser.add_argument("--scenario", choices=["forward", "reverse"], required=True)
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--n-envs", dest="n_envs", type=int, default=1)
    parser.add_argument("--lidar-beams", dest="lidar_beams", type=int, default=24)
    parser.add_argument("--reward", default="multiplicative")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-freq", dest="eval_freq", type=int, default=10_000)
    parser.add_argument("--normalized-eval-freq", dest="normalized_eval_freq",
                        type=int, default=30_000)
    parser.add_argument("--out-dir", dest="out_dir", default=str(DEFAULT_OUT))
    parser.add_argument("--e2e-rl-path", dest="e2e_rl_path", default=str(DEFAULT_E2E_RL))
    parser.add_argument("--variable-speed", dest="variable_speed", action="store_true",
                        help="2-D action [steer_rate, velocity]; bounded by --v-min/--v-max.")
    parser.add_argument("--v-min", dest="v_min", type=float, default=0.5)
    parser.add_argument("--v-max", dest="v_max", type=float, default=3.0)
    # Optional reward-shaping passthroughs (same semantics as train_lab_model).
    parser.add_argument("--curvature-speed-weight", dest="curvature_speed_weight",
                        type=float, default=0.0)
    parser.add_argument("--curvature-lookahead-samples", dest="curvature_lookahead",
                        type=int, default=10)
    args = parser.parse_args()

    e2e_rl_path = Path(args.e2e_rl_path).resolve()
    out_dir = Path(args.out_dir).resolve()

    # Apply the exact same lab overrides + monkey-patches as the trailer
    # trainer. They patch the BASE env classes our env subclasses, so they
    # apply to the tractor-only env transparently.
    tlm._apply_lab_config_overrides(e2e_rl_path)
    tlm._patch_env_vehicle_params(e2e_rl_path)
    tlm._patch_path_generator(e2e_rl_path)
    tlm._patch_actuator_lag(e2e_rl_path, steer_tau=0.05, velocity_tau=0.10)
    if args.variable_speed:
        tlm._patch_variable_speed_action(e2e_rl_path, v_min=args.v_min, v_max=args.v_max)
    tlm._patch_velocity_randomisation(e2e_rl_path)
    if args.curvature_speed_weight > 0.0:
        # NOTE: the curvature/target-speed reward patches in train_lab_model
        # override LineFollowingEnv.get_reward at the SOURCE level, which our
        # tractor-only get_reward override shadows. So these passthroughs only
        # take effect if you also remove our get_reward override — left here
        # for parity but expected unused for the no-trailer reward.
        tlm._patch_curvature_speed_penalty(
            e2e_rl_path, beta=args.curvature_speed_weight, K=args.curvature_lookahead
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(out_dir)
    print(f"[train_lab_tractor_only] cwd={out_dir}")
    print(f"[train_lab_tractor_only] e2e_rl={e2e_rl_path}")

    _train(
        scenario=args.scenario,
        reward=args.reward,
        lidar_beams=args.lidar_beams,
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        device=args.device,
        eval_freq=args.eval_freq,
        normalized_eval_freq=args.normalized_eval_freq,
    )


if __name__ == "__main__":
    main()
