#!/usr/bin/env python3
"""Train the `DynamicsMLP` on one or more `.npz` datasets produced by
`bag_to_dataset.py`, OR on synthetic rollouts from the e2e_rl
`StateSpaceTractorTrailer` (for sanity-checking that the architecture
can represent the kinematics).

Usage
-----
    # Real bag(s):
    python train_dynamics.py \\
      --dataset bags/sample_existing/dataset.npz \\
      --out lab_models_dynamics/v1 \\
      --epochs 50

    # Synthetic data (e2e_rl rollouts, no real bag needed):
    python train_dynamics.py \\
      --synthetic 5000 \\
      --out lab_models_dynamics/synthetic_sanity \\
      --epochs 50
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

REPO_ROOT = Path(__file__).resolve().parents[4]  # up from scripts/dynamics/ → repo
sys.path.insert(
    0, str(REPO_ROOT / "src" / "electrans_rl_bridge")
)
from electrans_rl_bridge.dynamics_model import DynamicsMLP, STATE_DIM, ACTION_DIM

# Column labels for human-readable per-dimension logging.
STATE_LABELS = ["xd", "yd", "pd", "s", "dx_t", "dy_t", "beta", "t_yaw_rate"]


# ----------------------------------------------------------------------------
# Dataset loading
# ----------------------------------------------------------------------------


def load_npz_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Returns (state, action_history, state_delta, topic_status)."""
    d = np.load(path, allow_pickle=True)
    state = d["state"].astype(np.float32)
    action_history = d["action_history"].astype(np.float32)
    state_delta = d["state_delta"].astype(np.float32)
    # Drop the last row — its target was just `state[T-1]` repeated.
    state = state[:-1]
    action_history = action_history[:-1]
    state_delta = state_delta[:-1]
    topic_status = dict(d["topic_status"]) if "topic_status" in d else {}
    return state, action_history, state_delta, topic_status


def make_synthetic_dataset(
    n_samples: int,
    action_window: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate `n_samples` (state, action_history, delta) tuples by
    rolling out the e2e_rl `StateSpaceTractorTrailer` from random
    initial states under random action sequences.

    Useful as a sanity check: a perfectly-trained NN should match the
    kinematic model exactly on this data.
    """
    sys.path.insert(0, "/home/ben/Ben/Thesis/e2e_rl")
    from VehicleModels.tractor_trailer import StateSpaceTractorTrailer
    from e2erl_utils import config as c

    # Override lab vehicle params (matches train_lab_model._apply_lab_config_overrides).
    args = dict(c.tesla_model_s_vehicle_params)
    args["lf"] = 0.33
    args["lr"] = 0.32

    rng = np.random.default_rng(seed)
    states: List[np.ndarray] = []
    histories: List[np.ndarray] = []
    deltas: List[np.ndarray] = []

    # Generate as many short rollouts as needed to hit `n_samples`.
    # 50-step rollouts → 50 - action_window samples per rollout.
    rollout_len = 50
    samples_per_rollout = rollout_len - action_window
    n_rollouts = max(1, int(math.ceil(n_samples / samples_per_rollout)))

    for _ in range(n_rollouts):
        vehicle = StateSpaceTractorTrailer(args=dict(args), trailer_length=2.0)
        vehicle.reset(
            xd=float(rng.uniform(-2.0, 2.0)),
            x=0.0, y=0.0,
            p=float(rng.uniform(-math.pi, math.pi)),
        )

        # Random action sequence: small smooth-ish drifts.
        steer_cmds = np.cumsum(rng.normal(0, 0.05, rollout_len)).astype(np.float32)
        steer_cmds = np.clip(steer_cmds, -math.pi / 4, math.pi / 4)
        vel_cmds = rng.uniform(-2.0, 2.0, rollout_len).astype(np.float32)
        actions = np.stack([steer_cmds, vel_cmds], axis=1)

        # Step through, collect body-frame states.
        bf_states: List[np.ndarray] = []
        for t in range(rollout_len):
            # Body-frame state BEFORE this step.
            cos_p = math.cos(vehicle.p)
            sin_p = math.sin(vehicle.p)
            dx_w = vehicle.trailer.x - vehicle.x
            dy_w = vehicle.trailer.y - vehicle.y
            dx_t = cos_p * dx_w + sin_p * dy_w
            dy_t = -sin_p * dx_w + cos_p * dy_w
            beta = (vehicle.p - vehicle.trailer.yaw + math.pi) % (2 * math.pi) - math.pi
            bf_states.append(
                np.array([vehicle.xd, vehicle.yd, vehicle.pd, vehicle.s,
                          dx_t, dy_t, beta, vehicle.trailer.yaw_rate],
                         dtype=np.float32)
            )
            vehicle.loop(actions[t])

        # Final body-frame state (after rollout end).
        cos_p = math.cos(vehicle.p)
        sin_p = math.sin(vehicle.p)
        dx_w = vehicle.trailer.x - vehicle.x
        dy_w = vehicle.trailer.y - vehicle.y
        bf_states.append(
            np.array([vehicle.xd, vehicle.yd, vehicle.pd, vehicle.s,
                      cos_p * dx_w + sin_p * dy_w,
                      -sin_p * dx_w + cos_p * dy_w,
                      (vehicle.p - vehicle.trailer.yaw + math.pi) % (2 * math.pi) - math.pi,
                      vehicle.trailer.yaw_rate],
                     dtype=np.float32)
        )
        bf_states_arr = np.stack(bf_states, axis=0)  # (rollout_len+1, 8)

        # Build (state, action_history, delta) tuples for valid timesteps.
        for t in range(action_window - 1, rollout_len):
            hist = actions[t - action_window + 1 : t + 1]  # (action_window, 2)
            states.append(bf_states_arr[t])
            histories.append(hist)
            delta = bf_states_arr[t + 1] - bf_states_arr[t]
            # Wrap β delta.
            delta[6] = (delta[6] + math.pi) % (2 * math.pi) - math.pi
            deltas.append(delta)
            if len(states) >= n_samples:
                break
        if len(states) >= n_samples:
            break

    return (
        np.stack(states, axis=0)[:n_samples],
        np.stack(histories, axis=0)[:n_samples],
        np.stack(deltas, axis=0)[:n_samples],
    )


# ----------------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------------


def per_component_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Returns shape (state_dim,) MSE per component."""
    return ((pred - target) ** 2).mean(dim=0)


def train(
    model: DynamicsMLP,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    *,
    epochs: int,
    lr: float,
    device: str,
    save_dir: Path,
    weight_components: Optional[torch.Tensor] = None,
) -> dict:
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_per_comp": []}

    for epoch in range(epochs):
        model.train()
        train_losses: List[float] = []
        for state, hist, delta in train_loader:
            state = state.to(device)
            hist = hist.to(device)
            delta = delta.to(device)
            pred = model(state, hist)
            if weight_components is None:
                loss = ((pred - delta) ** 2).mean()
            else:
                w = weight_components.to(device)
                loss = (((pred - delta) ** 2) * w).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(float(loss.detach().cpu()))

        train_loss = float(np.mean(train_losses))
        history["train_loss"].append(train_loss)

        val_loss = float("nan")
        val_per_comp = None
        if val_loader is not None:
            model.eval()
            losses: List[float] = []
            comps: List[torch.Tensor] = []
            with torch.no_grad():
                for state, hist, delta in val_loader:
                    state = state.to(device)
                    hist = hist.to(device)
                    delta = delta.to(device)
                    pred = model(state, hist)
                    losses.append(float(((pred - delta) ** 2).mean().cpu()))
                    comps.append(per_component_mse(pred, delta).cpu())
            val_loss = float(np.mean(losses))
            val_per_comp = torch.stack(comps, 0).mean(0).numpy()
            history["val_loss"].append(val_loss)
            history["val_per_comp"].append(val_per_comp.tolist())

            if val_loss < best_val:
                best_val = val_loss
                save_dir.mkdir(parents=True, exist_ok=True)
                model.save_checkpoint(
                    str(save_dir / "best.pt"),
                    meta={"epoch": epoch, "val_loss": val_loss},
                )

        if val_per_comp is not None:
            comp_str = "  ".join(
                f"{lab}={v:.2e}" for lab, v in zip(STATE_LABELS, val_per_comp)
            )
            print(f"epoch {epoch:3d}  train={train_loss:.5f}  val={val_loss:.5f}  {comp_str}")
        else:
            print(f"epoch {epoch:3d}  train={train_loss:.5f}")

    # Final save (in addition to best)
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_checkpoint(
        str(save_dir / "final.pt"),
        meta={"epoch": epochs, "val_loss": val_loss},
    )
    return history


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", default=None,
                    help="Path to .npz produced by bag_to_dataset.py. Repeatable.")
    ap.add_argument("--synthetic", type=int, default=0,
                    help="If >0, generate this many synthetic samples instead of "
                         "reading a bag-derived dataset.")
    ap.add_argument("--out", required=True, help="Output checkpoint directory")
    ap.add_argument("--action-window", type=int, default=11)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.dataset and args.synthetic <= 0:
        ap.error("Must provide --dataset PATH or --synthetic N")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Build dataset.
    states_list: List[np.ndarray] = []
    hist_list: List[np.ndarray] = []
    delta_list: List[np.ndarray] = []
    if args.synthetic > 0:
        print(f"[train] generating {args.synthetic} synthetic samples")
        s, h, d = make_synthetic_dataset(args.synthetic, args.action_window, seed=args.seed)
        states_list.append(s)
        hist_list.append(h)
        delta_list.append(d)
    if args.dataset:
        for ds_path in args.dataset:
            print(f"[train] loading {ds_path}")
            s, h, d, status = load_npz_dataset(Path(ds_path))
            if h.shape[1] != args.action_window:
                print(f"  WARN: dataset has action_window={h.shape[1]} "
                      f"but --action-window={args.action_window} — regenerate "
                      f"the dataset to match.")
            print(f"  status={status}")
            states_list.append(s)
            hist_list.append(h)
            delta_list.append(d)

    state_all = np.concatenate(states_list, axis=0)
    hist_all = np.concatenate(hist_list, axis=0)
    delta_all = np.concatenate(delta_list, axis=0)
    n = state_all.shape[0]
    print(f"[train] total samples: {n}")

    # Train/val split.
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_val = int(args.val_fraction * n)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    print(f"[train] split: {train_idx.size} train, {val_idx.size} val")

    state_t = torch.from_numpy(state_all).float()
    hist_t = torch.from_numpy(hist_all).float()
    delta_t = torch.from_numpy(delta_all).float()

    train_ds = TensorDataset(state_t[train_idx], hist_t[train_idx], delta_t[train_idx])
    val_ds = TensorDataset(state_t[val_idx], hist_t[val_idx], delta_t[val_idx])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = DynamicsMLP(action_window=args.action_window, hidden=args.hidden)
    print(f"[train] model: action_window={args.action_window} hidden={args.hidden}  "
          f"params={sum(p.numel() for p in model.parameters())}")

    save_dir = Path(args.out)
    train(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        save_dir=save_dir,
    )

    print(f"[train] done. Best checkpoint: {save_dir/'best.pt'}")


if __name__ == "__main__":
    main()
