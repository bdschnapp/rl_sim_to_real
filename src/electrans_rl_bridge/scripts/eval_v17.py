#!/usr/bin/env python3
"""Eval the v17 delay-aware reverse policy in pygame.

Mirrors eval_lab_model.py's behavior but constructs the env with the
DA-MDP ActionHistoryWrapper and loads the policy via its state_dict
(.pth) so we don't depend on the .zip cloudpickling the closure-defined
classes from v17_delay_aware.py.

Usage:
    /home/ben/Ben/Thesis/e2e_rl/venv/bin/python3 \\
      src/electrans_rl_bridge/scripts/eval_v17.py \\
      --phase 3 \\
      --episodes 5 \\
      --path-type winding
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "src/electrans_rl_bridge/scripts"
DEFAULT_E2E_RL = Path("/home/ben/Ben/Thesis/e2e_rl")
DEFAULT_OUT = REPO_ROOT.parent / "previous_models" / "lab_models_v20"

sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(DEFAULT_E2E_RL))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["1", "2", "3", "final"], default="final",
                    help="Which trained checkpoint to load. 'final' = re-exported best.")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--lidar-beams", type=int, default=24)
    ap.add_argument("--reward", default="multiplicative")
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--gru-hidden", type=int, default=64)
    ap.add_argument("--v-min", type=float, default=0.5)
    ap.add_argument("--v-max", type=float, default=3.0)
    ap.add_argument("--steer-tau", type=float, default=0.05,
                    help="Eval-time steer actuator lag (default 50ms = Phase 2 setting)")
    ap.add_argument("--velocity-tau", type=float, default=0.10,
                    help="Eval-time velocity actuator lag (default 100ms = Phase 2 setting)")
    ap.add_argument("--path-type", default="mix",
                    choices=["mix", "straight", "gentle", "sharp", "winding", "lab_seam"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--checkpoint-zip", default=None,
                    help="Override the auto-detected checkpoint path with a custom .zip")
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

    # Eval-time delay (matches deployment scenario)
    v17.patch_vehicle_with_delay(
        e2e_path,
        steer_tau_fn=lambda: float(args.steer_tau),
        velocity_tau_fn=lambda: float(args.velocity_tau),
    )

    # ---- Resolve checkpoint ----
    if args.checkpoint_zip:
        ckpt_zip = Path(args.checkpoint_zip)
    elif args.phase == "final":
        ckpt_zip = DEFAULT_OUT / "models/reverse/lidar_24/multiplicative/best_model.zip"
    else:
        ckpt_zip = (
            DEFAULT_OUT / f"phase_{args.phase}"
            / "models/reverse/lidar_24/multiplicative/best_model.zip"
        )
    print(f"[eval_v17] checkpoint: {ckpt_zip}")
    if not ckpt_zip.exists():
        print(f"[eval_v17] FATAL: no checkpoint at {ckpt_zip}")
        sys.exit(1)

    # ---- Force path-type override (if requested) ----
    if args.path_type != "mix":
        # The path generator picks a mode via self.np_random.choice(...).
        # np.random.Generator's `choice` is read-only on the bound method,
        # so instead we wrap generate_path itself: re-roll until the chosen
        # mode matches. The mixture has 5 modes so expected re-rolls < 10.
        import Environments.LineFollowing as lf
        orig_generate = lf.LineFollowingEnv.generate_path
        forced_mode = args.path_type

        def forced_generate(self):
            for _ in range(50):
                orig_generate(self)
                if getattr(self, "_path_kind", None) == forced_mode:
                    return
            print(f"[eval_v17] WARN: couldn't roll mode='{forced_mode}' in 50 tries; using last one")

        lf.LineFollowingEnv.generate_path = forced_generate
        print(f"[eval_v17] path-type forced to '{forced_mode}'")

    # ---- Build env (with ActionHistoryWrapper) ----
    import Environments.LineFollowing as lf
    inner = lf.ReverseLidarStateObservationLineFollowingEnv(
        render_mode=None if args.no_render else "human",
        max_episode_steps=1000,
        lidar_beams=args.lidar_beams,
        reward_mode=args.reward,
        fixed_speed=False,
    )
    env = v17.ActionHistoryWrapper(inner, K=args.K)
    print(f"[eval_v17] env: obs={env.observation_space}, action={env.action_space}")
    print(f"[eval_v17] steer_tau={args.steer_tau}s, velocity_tau={args.velocity_tau}s")

    # ---- Build TD3 and load policy ----
    # v20+: env returns flat Box (state ++ flattened action_history). Use
    # MlpPolicy + pass slicing dims to the GRU extractor.
    from stable_baselines3 import TD3
    state_dim = int(env.observation_space.shape[0]) - args.K * 2
    policy_kwargs = dict(
        features_extractor_class=v17.GRUDelayAwareFeatureExtractor,
        features_extractor_kwargs=dict(
            gru_hidden=args.gru_hidden,
            state_dim=state_dim,
            K=args.K,
            action_dim=2,
        ),
        net_arch=[256, 256],
        share_features_extractor=False,
    )
    model = TD3(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        buffer_size=1,  # we're not training
        device=args.device,
    )
    # Load the policy weights from the .zip's policy.pth (TD3.load via the
    # zip would re-pickle the env etc., which is fragile across versions —
    # so we extract just policy.pth and load that as state_dict).
    import zipfile, io
    import torch as th
    with zipfile.ZipFile(str(ckpt_zip)) as z:
        with z.open("policy.pth") as f:
            policy_blob = f.read()
    state_dict = th.load(io.BytesIO(policy_blob), map_location=model.device, weights_only=False)
    model.policy.load_state_dict(state_dict)
    print(f"[eval_v17] policy loaded.")

    # ---- Run episodes ----
    rewards = []
    do_render = not args.no_render
    for ep in range(args.episodes):
        obs, _ = env.reset()
        if do_render:
            env.render()
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
        print(f"[eval_v17] episode {ep+1}/{args.episodes}: reward={ep_reward:.1f}  steps={steps}")

    print(f"\n[eval_v17] mean reward over {args.episodes} eps: "
          f"{np.mean(rewards):.1f}  (min={min(rewards):.1f}, max={max(rewards):.1f})")


if __name__ == "__main__":
    main()
