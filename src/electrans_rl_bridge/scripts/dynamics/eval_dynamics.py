#!/usr/bin/env python3
"""Evaluate a trained `DynamicsMLP` checkpoint.

Two modes:

  1. One-step prediction MSE on a held-out `.npz` dataset, broken down
     per state component. Compares against a "zero-delta" baseline
     (predicting that the next state is identical to the current one).

  2. Closed-loop rollout: from the same initial state, step the learned
     model and the e2e_rl kinematic `StateSpaceTractorTrailer` forward
     with the same action sequence. Reports trajectory divergence —
     useful both as a sanity check (NN trained on synthetic data should
     ~match the kinematic model) and as a delta-finder (NN trained on
     real bags should DIVERGE from the kinematic model in the same
     direction the bag observations did).

Usage
-----
    python eval_dynamics.py \\
      --checkpoint lab_models_dynamics/v1/best.pt \\
      --dataset bags/sample_existing/dataset.npz

    python eval_dynamics.py \\
      --checkpoint lab_models_dynamics/v1/best.pt \\
      --rollout-steps 50
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(
    0, str(REPO_ROOT / "src" / "electrans_rl_bridge")
)
from electrans_rl_bridge.dynamics_model import (
    DynamicsMLP, NeuralDynamicsTractorTrailer, STATE_DIM, ACTION_DIM
)

STATE_LABELS = ["xd", "yd", "pd", "s", "dx_t", "dy_t", "beta", "t_yaw_rate"]


# ----------------------------------------------------------------------------
# Mode 1: one-step MSE on held-out data
# ----------------------------------------------------------------------------


def one_step_eval(checkpoint: Path, dataset: Path, device: str) -> None:
    print(f"[eval] loading checkpoint {checkpoint}")
    model = DynamicsMLP.load_checkpoint(str(checkpoint), device=device)
    model.eval()

    d = np.load(dataset, allow_pickle=True)
    state = torch.from_numpy(d["state"][:-1]).float()
    hist = torch.from_numpy(d["action_history"][:-1]).float()
    delta = torch.from_numpy(d["state_delta"][:-1]).float()

    print(f"[eval] dataset: {state.shape[0]} samples")

    # Model predictions
    with torch.no_grad():
        pred = []
        bs = 1024
        for i in range(0, state.size(0), bs):
            pred.append(model(state[i:i+bs].to(device), hist[i:i+bs].to(device)).cpu())
        pred = torch.cat(pred, 0)

    # Per-component MSE: model vs zero-delta baseline
    model_mse = ((pred - delta) ** 2).mean(0).numpy()
    zero_mse = (delta ** 2).mean(0).numpy()  # baseline: predict delta=0

    print("\n[eval] One-step MSE per component (smaller is better):")
    print(f"  {'dim':<12} {'model':>10} {'zero-Δ':>10} {'ratio':>10}")
    for i, lab in enumerate(STATE_LABELS):
        ratio = model_mse[i] / zero_mse[i] if zero_mse[i] > 1e-12 else float("nan")
        print(f"  {lab:<12} {model_mse[i]:>10.2e} {zero_mse[i]:>10.2e} {ratio:>10.3f}")

    overall_model = model_mse.mean()
    overall_zero = zero_mse.mean()
    print(f"\n  {'overall':<12} {overall_model:>10.2e} {overall_zero:>10.2e} "
          f"{overall_model/overall_zero:>10.3f}")
    print()
    if overall_model < overall_zero:
        print("[eval] model beats zero-delta baseline ✓")
    else:
        print("[eval] WARN: model worse than zero-delta baseline — under-trained "
              "or input data dominated by missing topics.")


# ----------------------------------------------------------------------------
# Mode 2: closed-loop rollout vs StateSpaceTractorTrailer
# ----------------------------------------------------------------------------


def rollout_eval(
    checkpoint: Path,
    n_steps: int,
    device: str,
    seed: int = 0,
) -> None:
    print(f"[eval] loading checkpoint {checkpoint}")
    model = DynamicsMLP.load_checkpoint(str(checkpoint), device=device)

    # NN dynamics
    nn_vehicle = NeuralDynamicsTractorTrailer(
        dynamics_net=model,
        args={"dt": 0.1, "lf": 0.33, "lr": 0.32, "trailer_length_m": 2.0},
        device=device,
    )

    # e2e_rl kinematic reference
    sys.path.insert(0, "/home/ben/Ben/Thesis/e2e_rl")
    from VehicleModels.tractor_trailer import StateSpaceTractorTrailer
    from e2erl_utils import config as c
    ref_args = dict(c.tesla_model_s_vehicle_params)
    ref_args["lf"] = 0.33
    ref_args["lr"] = 0.32
    ref_vehicle = StateSpaceTractorTrailer(args=dict(ref_args), trailer_length=2.0)

    # Same initial state for both.
    rng = np.random.default_rng(seed)
    init_xd = float(rng.uniform(-1.5, 1.5))
    init_p = float(rng.uniform(-math.pi, math.pi))
    nn_vehicle.reset(xd=init_xd, x=0.0, y=0.0, p=init_p)
    ref_vehicle.reset(xd=init_xd, x=0.0, y=0.0, p=init_p)

    # Smooth-ish action sequence (same kind we used for synthetic training).
    steer = np.cumsum(rng.normal(0, 0.05, n_steps)).astype(np.float32)
    steer = np.clip(steer, -math.pi / 4, math.pi / 4)
    vel = rng.uniform(-2.0, 2.0, n_steps).astype(np.float32)
    actions = np.stack([steer, vel], axis=1)

    # Roll out and accumulate position trajectories.
    nn_traj = np.zeros((n_steps + 1, 4), dtype=np.float32)   # x, y, p, s
    ref_traj = np.zeros((n_steps + 1, 4), dtype=np.float32)
    nn_traj[0] = [nn_vehicle.x, nn_vehicle.y, nn_vehicle.p, nn_vehicle.s]
    ref_traj[0] = [ref_vehicle.x, ref_vehicle.y, ref_vehicle.p, ref_vehicle.s]

    for k in range(n_steps):
        nn_vehicle.loop(actions[k])
        ref_vehicle.loop(actions[k])
        nn_traj[k+1] = [nn_vehicle.x, nn_vehicle.y, nn_vehicle.p, nn_vehicle.s]
        ref_traj[k+1] = [ref_vehicle.x, ref_vehicle.y, ref_vehicle.p, ref_vehicle.s]

    pos_err = np.linalg.norm(nn_traj[:, :2] - ref_traj[:, :2], axis=1)
    yaw_err = np.abs(((nn_traj[:, 2] - ref_traj[:, 2] + np.pi) % (2 * np.pi)) - np.pi)
    s_err = np.abs(nn_traj[:, 3] - ref_traj[:, 3])

    print(f"\n[eval] Closed-loop rollout ({n_steps} steps, dt=0.1s, seed={seed})")
    print(f"  initial state:  xd={init_xd:.3f}  p={init_p:.3f}")
    print(f"  final NN pose:  x={nn_traj[-1,0]:.3f}  y={nn_traj[-1,1]:.3f}  p={nn_traj[-1,2]:.3f}")
    print(f"  final ref pose: x={ref_traj[-1,0]:.3f}  y={ref_traj[-1,1]:.3f}  p={ref_traj[-1,2]:.3f}")
    print(f"  position error: mean={pos_err.mean():.3f}  max={pos_err.max():.3f}  m")
    print(f"  yaw error:      mean={yaw_err.mean():.3f}  max={yaw_err.max():.3f}  rad")
    print(f"  steering error: mean={s_err.mean():.3f}  max={s_err.max():.3f}  rad")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Path to best.pt or final.pt")
    ap.add_argument("--dataset", default=None,
                    help="Path to .npz dataset for one-step MSE eval.")
    ap.add_argument("--rollout-steps", type=int, default=50,
                    help="Run closed-loop rollout for N steps. 0 to disable.")
    ap.add_argument("--rollout-seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    if args.dataset:
        one_step_eval(Path(args.checkpoint), Path(args.dataset), args.device)

    if args.rollout_steps > 0:
        rollout_eval(
            Path(args.checkpoint),
            n_steps=args.rollout_steps,
            device=args.device,
            seed=args.rollout_seed,
        )


if __name__ == "__main__":
    main()
