#!/usr/bin/env python3
"""Eval the v15 (delay-free) reverse policy on a delayed env, wrapped
with the SmithPredictor predictor-feedback shim.

Three default scenarios should be run back-to-back to validate:
  1. Baseline (no Smith Predictor) at τ = 0.05/0.10  →  policy struggles
  2. Smith Predictor on at τ = 0.05/0.10              →  policy recovers
  3. Smith Predictor on at τ = 0                      →  sanity (no-op)

Usage:
    /home/ben/Ben/Thesis/e2e_rl/venv/bin/python3 \\
      src/electrans_rl_bridge/scripts/eval_smith_predictor.py \\
      --episodes 5

    # Disable Smith Predictor (baseline comparison)
    /home/ben/Ben/Thesis/e2e_rl/venv/bin/python3 \\
      src/electrans_rl_bridge/scripts/eval_smith_predictor.py \\
      --episodes 5 --no-smith-predictor
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "src/electrans_rl_bridge/scripts"
DEFAULT_E2E_RL = Path("/home/ben/Ben/Thesis/e2e_rl")
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "lab_models_v15/models/reverse/lidar_24/multiplicative/best_model.zip"
)

sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(DEFAULT_E2E_RL))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-zip", default=str(DEFAULT_CHECKPOINT),
                    help=f"Path to delay-free policy .zip (default: v15 reverse)")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--lidar-beams", type=int, default=24)
    ap.add_argument("--reward", default="multiplicative")
    ap.add_argument("--v-min", type=float, default=0.5)
    ap.add_argument("--v-max", type=float, default=3.0)
    ap.add_argument("--steer-tau", type=float, default=0.05,
                    help="Eval-time steer actuator lag (0 = no delay).")
    ap.add_argument("--velocity-tau", type=float, default=0.10,
                    help="Eval-time velocity actuator lag (0 = no delay).")
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--path-type", default="mix",
                    choices=["mix", "straight", "gentle", "sharp", "winding", "lab_seam"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--no-smith-predictor", action="store_true",
                    help="Disable Smith Predictor (baseline comparison run).")
    ap.add_argument("--delay-steps", type=int, default=None,
                    help="Override predictor rollout length (default: auto-compute from τ/dt).")
    args = ap.parse_args()

    # ---- Apply config patches identical to training ----
    import train_lab_model as tlm
    import v17_delay_aware as v17

    e2e_path = DEFAULT_E2E_RL
    tlm._apply_lab_config_overrides(e2e_path)
    tlm._patch_env_vehicle_params(e2e_path)
    tlm._patch_path_generator(e2e_path)
    tlm._patch_variable_speed_action(e2e_path, v_min=args.v_min, v_max=args.v_max)
    tlm._patch_velocity_randomisation(e2e_path)

    # Eval-time delay. Even at τ=0 we apply the patch so the env has the
    # lag-capable loop available (zero τ short-circuits to instant
    # actuation), keeping the code path identical across the three
    # verification scenarios.
    v17.patch_vehicle_with_delay(
        e2e_path,
        steer_tau_fn=lambda: float(args.steer_tau),
        velocity_tau_fn=lambda: float(args.velocity_tau),
    )

    # ---- Path-type override (if requested) ----
    if args.path_type != "mix":
        import Environments.LineFollowing as lf
        orig_generate = lf.LineFollowingEnv.generate_path
        forced_mode = args.path_type

        def forced_generate(self):
            for _ in range(50):
                orig_generate(self)
                if getattr(self, "_path_kind", None) == forced_mode:
                    return
            print(f"[eval_sp] WARN: couldn't roll mode='{forced_mode}' in 50 tries")

        lf.LineFollowingEnv.generate_path = forced_generate
        print(f"[eval_sp] path-type forced to '{forced_mode}'")

    # ---- Build env (and optionally wrap with SmithPredictor) ----
    import Environments.LineFollowing as lf
    inner = lf.ReverseLidarStateObservationLineFollowingEnv(
        render_mode=None if args.no_render else "human",
        max_episode_steps=1000,
        lidar_beams=args.lidar_beams,
        reward_mode=args.reward,
        fixed_speed=False,
    )
    if args.no_smith_predictor:
        wrapped = inner
        print(f"[eval_sp] Smith Predictor: DISABLED (baseline)")
    else:
        from smith_predictor import SmithPredictor
        wrapped = SmithPredictor(
            inner,
            steer_tau=args.steer_tau,
            velocity_tau=args.velocity_tau,
            dt=args.dt,
            lidar_beams=args.lidar_beams,
            delay_steps=args.delay_steps,
        )
        print(f"[eval_sp] Smith Predictor: ENABLED  delay_steps={wrapped.delay_steps}")
    print(f"[eval_sp] env obs={wrapped.observation_space}, action={wrapped.action_space}")
    print(f"[eval_sp] eval τ_steer={args.steer_tau}s  τ_vel={args.velocity_tau}s")

    # ---- Load v15 delay-free policy ----
    from stable_baselines3 import TD3
    print(f"[eval_sp] loading policy from: {args.checkpoint_zip}")
    # v15 used plain MlpPolicy (default arch), 32-dim Box obs, 2-D action.
    # Build a fresh TD3 against the INNER env (same obs/action space as
    # what v15 was trained on) and load only the policy state_dict.
    model = TD3(
        policy="MlpPolicy",
        env=inner,
        policy_kwargs=dict(net_arch=[256, 256]),
        buffer_size=1,
        device=args.device,
    )
    import zipfile, io
    import torch as th
    with zipfile.ZipFile(args.checkpoint_zip) as z:
        with z.open("policy.pth") as f:
            policy_blob = f.read()
    state_dict = th.load(
        io.BytesIO(policy_blob), map_location=model.device, weights_only=False
    )
    model.policy.load_state_dict(state_dict)
    print(f"[eval_sp] policy loaded ({len(state_dict)} keys).")

    # ---- Run episodes ----
    rewards = []
    lengths = []
    do_render = not args.no_render
    for ep in range(args.episodes):
        obs, _ = wrapped.reset()
        if do_render:
            inner.render()
        ep_reward = 0.0
        steps = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = wrapped.step(action)
            if do_render:
                inner.render()
            ep_reward += float(r)
            steps += 1
            done = term or trunc
        rewards.append(ep_reward)
        lengths.append(steps)
        print(f"[eval_sp] episode {ep+1}/{args.episodes}: "
              f"reward={ep_reward:.1f}  steps={steps}")

    print(f"\n[eval_sp] mean reward: {np.mean(rewards):.1f}  "
          f"(min={min(rewards):.1f}, max={max(rewards):.1f})")
    print(f"[eval_sp] mean ep_len: {np.mean(lengths):.1f}  "
          f"(min={min(lengths)}, max={max(lengths)})")


if __name__ == "__main__":
    main()
