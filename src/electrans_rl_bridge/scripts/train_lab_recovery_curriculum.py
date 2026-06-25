#!/usr/bin/env python3
"""3-phase curriculum fine-tune of v15 → v15oos on recovery scenarios.

Phase schedule (recovery_probability = 0.40 throughout):

  P1 (easy, 25k):  hitch ≈ 0.15-0.30, steer ≈ 0.15-0.25, no lateral offset
  P2 (medium, 50k): hitch ≈ 0.25-0.55, steer ≈ 0.20-0.30, +0.4 m Mode B offset
  P3 (full, 75k):  hitch ≈ 0.35-0.80, steer ≈ 0.30-0.40, +0.8 m Mode B offset

Each phase warm-starts from the previous phase's best checkpoint
(Phase 1 warm-starts from v15). The final phase's best_model.zip is
copied to lab_models_v15oos/.../best_model.zip so the bridge launch
defaults pick it up.

The Mode B lateral offset is critical at higher hitch magnitudes: at
β=0.5 rad with v=−1 m/s the trailer's natural lateral drift is ~0.43
m/s, hitting the 1.41 m corridor edge in ~10 steps regardless of
policy action. The lateral offset shifts the entire spawn laterally
away from the trailer's natural drift direction, giving the trailer
0.5-1.0 m of clearance to react.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_E2E_RL = Path("/home/ben/Ben/Thesis/e2e_rl")
DEFAULT_INIT_ZIP = (
    REPO_ROOT / "lab_models_v15/models/reverse/lidar_24/multiplicative/best_model.zip"
)
DEFAULT_OUT = REPO_ROOT / "lab_models_v15oos"
SCRIPTS_DIR = REPO_ROOT / "src/electrans_rl_bridge/scripts"

sys.path.insert(0, str(SCRIPTS_DIR))


PHASES = [
    dict(
        name="phase1_easy",
        timesteps=25_000,
        recovery_probability=0.40,
        mode_a_steer_low=0.15, mode_a_steer_high=0.25,
        mode_a_hitch_low=0.20, mode_a_hitch_high=0.30,
        mode_b_steer_low=0.15, mode_b_steer_high=0.25,
        mode_b_hitch_low=0.15, mode_b_hitch_high=0.25,
        mode_b_lateral_offset=0.0,
    ),
    dict(
        name="phase2_medium",
        timesteps=50_000,
        recovery_probability=0.40,
        mode_a_steer_low=0.20, mode_a_steer_high=0.30,
        mode_a_hitch_low=0.40, mode_a_hitch_high=0.55,
        mode_b_steer_low=0.20, mode_b_steer_high=0.30,
        mode_b_hitch_low=0.25, mode_b_hitch_high=0.35,
        mode_b_lateral_offset=0.4,
    ),
    dict(
        name="phase3_full",
        timesteps=75_000,
        recovery_probability=0.40,
        mode_a_steer_low=0.30, mode_a_steer_high=0.40,
        mode_a_hitch_low=0.60, mode_a_hitch_high=0.80,
        mode_b_steer_low=0.30, mode_b_steer_high=0.40,
        mode_b_hitch_low=0.35, mode_b_hitch_high=0.50,
        mode_b_lateral_offset=0.8,
    ),
]


def _train_phase(
    e2e_rl_path: Path,
    phase: dict,
    init_zip: Path,
    out_dir: Path,
    n_envs: int,
    eval_freq: int,
    lidar_beams: int,
    reward_mode: str,
    device: str,
) -> Path:
    """Run one curriculum phase. Returns path to this phase's best_model.zip."""
    import train_lab_model as tlm
    import train_lab_recovery as tlr
    import v17_delay_aware as v17

    # Re-apply recovery patch with this phase's bounds. _patch_recovery_scenarios
    # is idempotent thanks to the _recovery_orig_reset stash.
    tlr._patch_recovery_scenarios(
        e2e_rl_path,
        recovery_probability=phase["recovery_probability"],
        mode_a_steer_low=phase["mode_a_steer_low"],
        mode_a_steer_high=phase["mode_a_steer_high"],
        mode_a_hitch_low=phase["mode_a_hitch_low"],
        mode_a_hitch_high=phase["mode_a_hitch_high"],
        mode_b_steer_low=phase["mode_b_steer_low"],
        mode_b_steer_high=phase["mode_b_steer_high"],
        mode_b_hitch_low=phase["mode_b_hitch_low"],
        mode_b_hitch_high=phase["mode_b_hitch_high"],
        mode_b_lateral_offset=phase["mode_b_lateral_offset"],
        force_recovery=False,
    )

    save_dir = out_dir / phase["name"] / "models/reverse/lidar_24/multiplicative"
    logs_dir = save_dir / "logs"
    save_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    from stable_baselines3 import TD3
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    from stable_baselines3.common.monitor import Monitor

    def make_env_fn():
        def _f():
            return Monitor(tlr._build_env(
                e2e_rl_path, lidar_beams=lidar_beams,
                reward_mode=reward_mode,
            ))
        return _f

    if n_envs > 1:
        import multiprocessing as mp
        try: mp.set_start_method("fork", force=True)
        except RuntimeError: pass
        train_env = SubprocVecEnv([make_env_fn() for _ in range(n_envs)])
    else:
        train_env = DummyVecEnv([make_env_fn()])
    eval_env = DummyVecEnv([make_env_fn()])

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
        device=device,
        verbose=1,
    )

    print(f"[curriculum:{phase['name']}] warm-starting policy from {init_zip}")
    import zipfile, io
    import torch as th
    with zipfile.ZipFile(str(init_zip)) as z:
        with z.open("policy.pth") as f:
            sd = th.load(io.BytesIO(f.read()),
                         map_location=model.device, weights_only=False)
    model.policy.load_state_dict(sd)

    eval_callback = v17.RewardPerStepEvalCallback(
        eval_env,
        best_model_save_path=str(save_dir),
        log_path=str(logs_dir),
        eval_freq=max(eval_freq // max(1, n_envs), 1),
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )

    print(f"[curriculum:{phase['name']}] training {phase['timesteps']} steps → {save_dir}")
    print(f"[curriculum:{phase['name']}] bounds: "
          f"A(steer={phase['mode_a_steer_low']:.2f}-{phase['mode_a_steer_high']:.2f}, "
          f"hitch={phase['mode_a_hitch_low']:.2f}-{phase['mode_a_hitch_high']:.2f}) "
          f"B(steer={phase['mode_b_steer_low']:.2f}-{phase['mode_b_steer_high']:.2f}, "
          f"hitch={phase['mode_b_hitch_low']:.2f}-{phase['mode_b_hitch_high']:.2f}, "
          f"offset={phase['mode_b_lateral_offset']:.2f} m)")

    t0 = time.time()
    model.learn(total_timesteps=phase["timesteps"], callback=eval_callback)
    model.save(str(save_dir / "final.zip"))
    print(f"[curriculum:{phase['name']}] done in {(time.time()-t0)/60:.1f} min")

    train_env.close()
    eval_env.close()

    return save_dir / "best_model.zip"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-from-zip", default=str(DEFAULT_INIT_ZIP),
                    help="v15 checkpoint to start phase 1 from")
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--lidar-beams", type=int, default=24)
    ap.add_argument("--reward", default="multiplicative")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--eval-freq", type=int, default=5_000)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--e2e-rl-path", default=str(DEFAULT_E2E_RL))
    args = ap.parse_args()

    e2e_rl_path = Path(args.e2e_rl_path).resolve()
    out_dir = Path(args.out_dir).resolve()

    # Apply v15-style env patches once (idempotent / persistent across phases)
    sys.path.insert(0, str(e2e_rl_path))
    import train_lab_model as tlm
    tlm._apply_lab_config_overrides(e2e_rl_path)
    tlm._patch_env_vehicle_params(e2e_rl_path)
    tlm._patch_path_generator(e2e_rl_path)
    tlm._patch_variable_speed_action(e2e_rl_path, v_min=0.5, v_max=3.0)
    tlm._patch_velocity_randomisation(e2e_rl_path)

    print(f"[curriculum] {len(PHASES)} phases, "
          f"total {sum(p['timesteps'] for p in PHASES)} steps")

    init_zip = Path(args.init_from_zip).resolve()
    for i, phase in enumerate(PHASES):
        print(f"\n{'='*60}\n[curriculum] PHASE {i+1}/{len(PHASES)}: {phase['name']}\n{'='*60}")
        if not init_zip.exists():
            raise FileNotFoundError(f"init zip missing: {init_zip}")
        init_zip = _train_phase(
            e2e_rl_path=e2e_rl_path,
            phase=phase,
            init_zip=init_zip,
            out_dir=out_dir,
            n_envs=args.n_envs,
            eval_freq=args.eval_freq,
            lidar_beams=args.lidar_beams,
            reward_mode=args.reward,
            device=args.device,
        )

    # Promote phase 3's best to the standard v15oos location the bridge
    # launch defaults already point at.
    final_dst_dir = out_dir / "models/reverse/lidar_24/multiplicative"
    final_dst_dir.mkdir(parents=True, exist_ok=True)
    final_dst = final_dst_dir / "best_model.zip"
    shutil.copy(str(init_zip), str(final_dst))
    print(f"\n[curriculum] DONE. Final best_model.zip → {final_dst}")
    print(f"[curriculum] Re-export to .policy.pth with:")
    print(f"  /home/ben/Ben/Thesis/e2e_rl/venv/bin/python3 "
          f"{SCRIPTS_DIR}/re_export_td3.py {final_dst} "
          f"{final_dst_dir}/best_model.policy.pth --reverse")


if __name__ == "__main__":
    main()
