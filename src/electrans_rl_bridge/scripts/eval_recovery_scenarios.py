#!/usr/bin/env python3
"""Visualise the recovery-scenario reset distribution before training.

Forces 100% of episodes to be recovery scenarios so you can see the
starting state (steer ≈ ±20°, hitch ≈ ±40°, sign-matched) and watch how
v15 (or any other checkpoint you point at) responds. Useful as a sanity
check that the perturbation is set up correctly before kicking off the
fine-tune training.

By default loads v15; pass --checkpoint-zip to point at another policy.

Usage:
    /home/ben/Ben/Thesis/e2e_rl/venv/bin/python3 \\
      src/electrans_rl_bridge/scripts/eval_recovery_scenarios.py \\
      --episodes 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "src/electrans_rl_bridge/scripts"
DEFAULT_E2E_RL = Path("/home/ben/Ben/Thesis/e2e_rl")
DEFAULT_V15_ZIP = (
    REPO_ROOT / "lab_models_v15/models/reverse/lidar_24/multiplicative/best_model.zip"
)

sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(DEFAULT_E2E_RL))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-zip", default=str(DEFAULT_V15_ZIP),
                    help="Policy .zip to drive the recovery scenarios.")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--lidar-beams", type=int, default=24)
    ap.add_argument("--reward", default="multiplicative")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-render", action="store_true")
    # Allow the user to tweak the perturbation bounds from the CLI if
    # they want to test edge cases.
    ap.add_argument("--mode-a-steer-low", type=float, default=0.30)
    ap.add_argument("--mode-a-steer-high", type=float, default=0.40)
    ap.add_argument("--mode-a-hitch-low", type=float, default=0.60)
    ap.add_argument("--mode-a-hitch-high", type=float, default=0.80)
    ap.add_argument("--mode-b-steer-low", type=float, default=0.30)
    ap.add_argument("--mode-b-steer-high", type=float, default=0.40)
    ap.add_argument("--mode-b-hitch-low", type=float, default=0.35)
    ap.add_argument("--mode-b-hitch-high", type=float, default=0.50)
    ap.add_argument("--mode-a-probability", type=float, default=0.5)
    ap.add_argument("--mode-b-lateral-offset", type=float, default=0.0,
                    help="Shift Mode B spawn laterally to give trailer clearance")
    ap.add_argument("--init-speed-low", type=float, default=0.5)
    ap.add_argument("--init-speed-high", type=float, default=1.0)
    ap.add_argument("--force-mode", choices=["A", "B"], default=None,
                    help="Lock every episode to this recovery mode "
                         "(default: alternate per --mode-a-probability)")
    args = ap.parse_args()

    # Apply v15-style patches
    import train_lab_model as tlm
    import train_lab_recovery as tlr
    e2e = DEFAULT_E2E_RL
    tlm._apply_lab_config_overrides(e2e)
    tlm._patch_env_vehicle_params(e2e)
    tlm._patch_path_generator(e2e)
    tlm._patch_variable_speed_action(e2e, v_min=0.5, v_max=3.0)
    tlm._patch_velocity_randomisation(e2e)

    # Recovery patch with force_recovery=True → every episode is a
    # recovery scenario.
    tlr._patch_recovery_scenarios(
        e2e,
        mode_a_steer_low=args.mode_a_steer_low,
        mode_a_steer_high=args.mode_a_steer_high,
        mode_a_hitch_low=args.mode_a_hitch_low,
        mode_a_hitch_high=args.mode_a_hitch_high,
        mode_b_steer_low=args.mode_b_steer_low,
        mode_b_steer_high=args.mode_b_steer_high,
        mode_b_hitch_low=args.mode_b_hitch_low,
        mode_b_hitch_high=args.mode_b_hitch_high,
        mode_a_probability=args.mode_a_probability,
        mode_b_lateral_offset=args.mode_b_lateral_offset,
        init_speed_low=args.init_speed_low,
        init_speed_high=args.init_speed_high,
        force_recovery=True,
        force_mode=args.force_mode,
    )

    # Build env (with rendering unless --no-render)
    import Environments.LineFollowing as lf
    env = lf.ReverseLidarStateObservationLineFollowingEnv(
        render_mode=None if args.no_render else "human",
        max_episode_steps=1000,
        lidar_beams=args.lidar_beams,
        reward_mode=args.reward,
        fixed_speed=False,
    )

    # Load policy
    from stable_baselines3 import TD3
    model = TD3(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=dict(net_arch=[256, 256]),
        buffer_size=1,
        device=args.device,
    )
    import zipfile, io
    import torch as th
    print(f"[eval_recovery] loading policy from {args.checkpoint_zip}")
    with zipfile.ZipFile(args.checkpoint_zip) as z:
        with z.open("policy.pth") as f:
            sd = th.load(io.BytesIO(f.read()),
                         map_location=model.device, weights_only=False)
    model.policy.load_state_dict(sd)
    print(f"[eval_recovery] policy loaded ({len(sd)} keys)")

    # Run episodes; report the spawn state from info so you can sanity-check
    do_render = not args.no_render
    rewards, lengths = [], []
    for ep in range(args.episodes):
        obs, info = env.reset()
        if do_render:
            env.render()
        spawn_mode = info.get("recovery_mode", "?")
        spawn_steer = info.get("recovery_steer", float("nan"))
        spawn_hitch = info.get("recovery_hitch", float("nan"))
        print(f"[eval_recovery] episode {ep+1}/{args.episodes} [mode {spawn_mode}]: "
              f"spawn  steer={spawn_steer:+.3f} rad ({np.degrees(spawn_steer):+.1f}°)  "
              f"hitch={spawn_hitch:+.3f} rad ({np.degrees(spawn_hitch):+.1f}°)")
        ep_reward = 0.0
        steps = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            if do_render:
                env.render()
            ep_reward += float(r)
            steps += 1
            done = term or trunc
        rewards.append(ep_reward)
        lengths.append(steps)
        print(f"   → reward={ep_reward:.1f}  steps={steps}")

    print(f"\n[eval_recovery] mean reward: {np.mean(rewards):.1f}  "
          f"(min={min(rewards):.1f}, max={max(rewards):.1f})")
    print(f"[eval_recovery] mean ep_len: {np.mean(lengths):.1f}  "
          f"(min={min(lengths)}, max={max(lengths)})")


if __name__ == "__main__":
    main()
