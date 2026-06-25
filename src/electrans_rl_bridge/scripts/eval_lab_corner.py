#!/usr/bin/env python3
"""Quick pygame eval of a lab-trained reverse policy on forced lab_corner
paths. Bypasses SB3's `TD3.load` strict action-space check by building a
fresh TD3 against the (patched) env and loading only the policy weights
— same load pattern the ROS bridge uses, so what you see here is what
the bridge runs in deployment.

Usage
-----
    /home/ben/Ben/Thesis/e2e_rl/venv/bin/python3 \\
      src/electrans_rl_bridge/scripts/eval_lab_corner.py \\
      --checkpoint-zip <path-to-best_model.zip> \\
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
DEFAULT_CHECKPOINT_ZIP = (
    REPO_ROOT / "lab_models_v18/models/reverse/lidar_24/multiplicative/best_model.zip"
)

sys.path.insert(0, str(SCRIPTS_DIR))


def _force_path_kind(e2e_rl_path: Path, kind: str):
    """Monkey-patch the path generator so every episode spawns the same
    kind. Only `lab_corner` and `lab_seam` supported here — for the rest
    use the standard mix-mode generator via train_lab_model."""
    if str(e2e_rl_path) not in sys.path:
        sys.path.insert(0, str(e2e_rl_path))
    import Environments.LineFollowing as lf
    import Environments.TractorTrailer as tt

    def generate_path(self):
        world_width_m = tt.WINDOW_WIDTH * tt.METERS_PER_PIXEL
        x0 = 5.0
        x_end = world_width_m - 5.0
        self.xx = np.arange(int(x0), int(x_end), 1)
        self._path_kind = kind

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
            y_local = dy1 * ramp1 + dy2 * ramp2
        elif kind == "lab_seam":
            bend_x = float(self.np_random.uniform(x0 + 6.0, x_end - 4.0))
            width = float(self.np_random.uniform(0.4, 0.7))
            sign = 1.0 if self.np_random.random() < 0.5 else -1.0
            dy = sign * float(self.np_random.uniform(1.5, 3.0))
            ramp = (np.tanh((self.xx - bend_x) / width) + 1.0) * 0.5
            y_local = dy * ramp
        else:
            raise ValueError(f"unsupported forced kind: {kind!r}")
        self.yy = y_local + 45.0
        x_g = self.xx[-1]
        y_g = self.yy[-1]
        yaw_g = float(
            np.arctan2(self.yy[-1] - self.yy[-2], self.xx[-1] - self.xx[-2])
        )
        self.goal_pose = (x_g, y_g, yaw_g)

    lf.LineFollowingEnv.generate_path = generate_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-zip", default=str(DEFAULT_CHECKPOINT_ZIP),
                    help="Policy .zip to drive the eval.")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--path-kind", choices=["lab_corner", "lab_seam"],
                    default="lab_corner")
    ap.add_argument("--lidar-beams", type=int, default=24)
    ap.add_argument("--reward", default="multiplicative")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-render", action="store_true")
    # Match ROS deployment: pygame env gets actuator lag applied, and
    # the policy is wrapped by the Smith Predictor that the bridge runs.
    # Default ON — these are the conditions the deployed model actually
    # experiences. Turn off with --no-smith-predictor to see what the
    # policy does in its native training-time delay-free env.
    ap.add_argument("--smith-predictor", dest="smith_predictor",
                    action="store_true", default=True,
                    help="(default) Apply actuator-lag patch + Smith "
                         "Predictor wrapper, matching ROS deployment.")
    ap.add_argument("--no-smith-predictor", dest="smith_predictor",
                    action="store_false",
                    help="Run delay-free, matching training conditions.")
    ap.add_argument("--smith-steer-tau", type=float, default=0.05,
                    help="Steering actuator time constant (matches sim "
                         "yaml's steer_time_constant)")
    ap.add_argument("--smith-velocity-tau", type=float, default=0.10,
                    help="Velocity actuator time constant (matches sim "
                         "yaml's acc_time_constant)")
    args = ap.parse_args()

    e2e = DEFAULT_E2E_RL

    # Apply v15/v16-style env patches in the same order as training
    import train_lab_model as tlm
    tlm._apply_lab_config_overrides(e2e)
    tlm._patch_env_vehicle_params(e2e)
    tlm._patch_path_generator(e2e)  # will be re-overridden below
    tlm._patch_variable_speed_action(e2e, v_min=0.5, v_max=3.0)
    tlm._patch_velocity_randomisation(e2e)

    # Force every episode to be the requested path kind.
    _force_path_kind(e2e, args.path_kind)

    # Optionally apply the actuator-lag patch so the pygame env mirrors
    # the ROS sim's first-order steer/velocity dynamics. The Smith
    # Predictor below compensates for this lag — same stack as the
    # bridge runs in deployment.
    if args.smith_predictor:
        import v17_delay_aware as v17
        v17.patch_vehicle_with_delay(
            e2e,
            steer_tau_fn=lambda: float(args.smith_steer_tau),
            velocity_tau_fn=lambda: float(args.smith_velocity_tau),
        )

    # Build env (with rendering unless --no-render)
    import Environments.LineFollowing as lf
    inner = lf.ReverseLidarStateObservationLineFollowingEnv(
        render_mode=None if args.no_render else "human",
        max_episode_steps=1000,
        lidar_beams=args.lidar_beams,
        reward_mode=args.reward,
        fixed_speed=False,
    )
    if args.smith_predictor:
        from smith_predictor import SmithPredictor
        env = SmithPredictor(
            inner,
            steer_tau=args.smith_steer_tau,
            velocity_tau=args.smith_velocity_tau,
            dt=0.1,
            lidar_beams=args.lidar_beams,
        )
        print(f"[eval_lab_corner] Smith Predictor: ENABLED  "
              f"τ_steer={args.smith_steer_tau}s  τ_vel={args.smith_velocity_tau}s  "
              f"delay_steps={env.delay_steps}")
    else:
        env = inner
        print(f"[eval_lab_corner] Smith Predictor: DISABLED (delay-free)")

    # Build a fresh TD3 against the inner env's (patched) action_space,
    # then load only the policy weights — bypasses SB3's strict action-
    # space check that fails when the saved model's action_space differs
    # from the eval env's. Use inner regardless of SP wrapping; the SP
    # exposes the same spaces but TD3 introspects more cleanly off the
    # unwrapped env.
    from stable_baselines3 import TD3
    model = TD3(
        policy="MlpPolicy",
        env=inner,
        policy_kwargs=dict(net_arch=[256, 256]),
        buffer_size=1,
        device=args.device,
    )
    import zipfile, io
    import torch as th
    print(f"[eval_lab_corner] loading policy from {args.checkpoint_zip}")
    with zipfile.ZipFile(args.checkpoint_zip) as z:
        with z.open("policy.pth") as f:
            sd = th.load(io.BytesIO(f.read()),
                         map_location=model.device, weights_only=False)
    model.policy.load_state_dict(sd)
    print(f"[eval_lab_corner] policy loaded ({len(sd)} keys)")

    do_render = not args.no_render
    rewards, lengths = [], []
    for ep in range(args.episodes):
        obs, info = env.reset()
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
        lengths.append(steps)
        print(f"[eval_lab_corner] ep {ep+1}/{args.episodes}: "
              f"steps={steps}  reward={ep_reward:.1f}")

    print(f"\n[eval_lab_corner] mean reward: {np.mean(rewards):.1f}  "
          f"(min={min(rewards):.1f}, max={max(rewards):.1f})")
    print(f"[eval_lab_corner] mean ep_len: {np.mean(lengths):.1f}  "
          f"(min={min(lengths)}, max={max(lengths)})")


if __name__ == "__main__":
    main()
