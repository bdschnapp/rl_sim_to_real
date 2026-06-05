#!/usr/bin/env python3
"""
Plot the multiplicative reward's speed_shape surface as a function of
(|ẋ|, κ̂) for chosen reward parameters. Used to tune α/σ/v_target without
running a full training cycle.

The surface plotted is exactly what scripts/train_lab_model.py applies
when --target-speed-bonus > 0, namely:

    R(ẋ, κ̂) = α · speed_shape(ẋ, v_target(κ̂))

where speed_shape uses the baseline-subtracted Gaussian for v_target > ε
and a plain peak-at-zero Gaussian for v_target ≤ ε:

    speed_shape(v, μ)
      = max(0, g(v, μ) - g(0, μ)) / (1 - g(0, μ))     if μ > ε
      = exp(-v² / (2σ²))                              otherwise
    g(x, μ) = exp(-(x - μ)² / (2σ²))

v_target is a linear interpolation between v_target_max (κ̂=0) and
v_target_min (κ̂ ≥ κ_max), clamped:

    v_target(κ̂) = v_target_max + (v_target_min - v_target_max) · clip(κ̂/κ_max, 0, 1)

Usage
-----
    # Plot the v10 shape
    python plot_reward_surface.py

    # Sweep σ to compare narrow vs wide peaks
    python plot_reward_surface.py --target-speed-sigma 0.3 --out /tmp/sigma_0p3.png
    python plot_reward_surface.py --target-speed-sigma 0.7 --out /tmp/sigma_0p7.png

    # Try a lower v_target floor (e.g. for tighter cornering)
    python plot_reward_surface.py --target-speed-v-min 0.5

    # Show the plot interactively instead of saving
    python plot_reward_surface.py --show

The figure has two panels:
  Left  — 2D heatmap (v on x, κ̂ on y, colour = reward), v_target ridge
          overlaid as a white curve.
  Right — Reward vs. v at three fixed κ̂ samples: 0 (straight),
          κ_max/2 (medium curve), κ_max (sharp curve).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def speed_shape(
    v: np.ndarray,
    v_target: np.ndarray,
    sigma_low: float,
    sigma_high: float,
    epsilon: float = 0.05,
) -> np.ndarray:
    """Vectorised asymmetric baseline-subtracted Gaussian. v and v_target
    broadcast against each other. Returns array of same broadcast shape.

    σ_low used on the slow side of v_target, σ_high on the fast side.
    Baseline g(0, v_target) uses σ_low (since 0 is on the slow side
    when v_target > 0). For v_target ≤ ε we fall back to a peak-at-zero
    Gaussian (obstacle case)."""
    two_sigma_low_sq = 2.0 * sigma_low * sigma_low
    two_sigma_high_sq = 2.0 * sigma_high * sigma_high

    # Pick σ² element-wise based on whether v is below or above v_target.
    sigma_sq = np.where(v < v_target, two_sigma_low_sq, two_sigma_high_sq)
    g_v = np.exp(-((v - v_target) ** 2) / sigma_sq)

    # Baseline at v=0 always uses σ_low (since 0 < v_target when v_target > ε).
    g_0 = np.exp(-(v_target ** 2) / two_sigma_low_sq)
    denom = np.where(np.abs(1.0 - g_0) < 1e-9, 1.0, 1.0 - g_0)
    main_branch = np.clip((g_v - g_0) / denom, 0.0, None)

    # Obstacle branch (v_target ≤ ε): peak-at-zero Gaussian with σ_low.
    obstacle_branch = np.exp(-(v ** 2) / two_sigma_low_sq)

    return np.where(v_target > epsilon, main_branch, obstacle_branch)


def v_target_curve(kappa_hat: np.ndarray, v_min: float, v_max: float, kappa_max: float) -> np.ndarray:
    frac = np.clip(kappa_hat / kappa_max, 0.0, 1.0)
    return v_max + (v_min - v_max) * frac


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--target-speed-bonus", dest="alpha", type=float, default=8.0)
    parser.add_argument("--target-speed-sigma-low", dest="sigma_low", type=float, default=0.2,
                        help="σ on the slow side of v_target (default: 0.2).")
    parser.add_argument("--target-speed-sigma-high", dest="sigma_high", type=float, default=0.5,
                        help="σ on the fast side of v_target (default: 0.5).")
    parser.add_argument("--target-speed-v-min", dest="v_target_min", type=float, default=0.6)
    parser.add_argument("--target-speed-v-max", dest="v_target_max", type=float, default=2.0)
    parser.add_argument("--kappa-max", dest="kappa_max", type=float, default=0.3,
                        help="Matches Environments.LineFollowing.compute_curvature's clip.")
    parser.add_argument("--v-action-min", dest="v_action_min", type=float, default=0.5,
                        help="Lower x-axis bound (matches policy action_space.low[1]).")
    parser.add_argument("--v-action-max", dest="v_action_max", type=float, default=3.0,
                        help="Upper x-axis bound (matches policy action_space.high[1]).")
    parser.add_argument("--out", default=None,
                        help="Output PNG path. Default: ./reward_surface_a{α}_s{σ}_vmin{vmin}_vmax{vmax}.png")
    parser.add_argument("--show", action="store_true",
                        help="Open an interactive matplotlib window instead of saving.")
    args = parser.parse_args()

    # ---- Build the grid ----
    vs = np.linspace(args.v_action_min, args.v_action_max, 200)
    ks = np.linspace(0.0, args.kappa_max * 1.2, 200)  # slight headroom past κ_max
    V, K = np.meshgrid(vs, ks, indexing="xy")  # shape (len(ks), len(vs))

    Vt = v_target_curve(K, args.v_target_min, args.v_target_max, args.kappa_max)
    Z = args.alpha * speed_shape(V, Vt, args.sigma_low, args.sigma_high)

    # v_target curve (1-D over κ)
    vt_line = v_target_curve(ks, args.v_target_min, args.v_target_max, args.kappa_max)

    # ---- Plot ----
    if not args.show:
        matplotlib.use("Agg")

    fig, (ax_heat, ax_slice) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: heatmap with v_target overlay.
    im = ax_heat.pcolormesh(V, K, Z, shading="auto", cmap="viridis", vmin=0.0, vmax=args.alpha)
    cbar = fig.colorbar(im, ax=ax_heat)
    cbar.set_label("speed_shape · α  (per-step reward at perfect path/hitch)")
    ax_heat.plot(vt_line, ks, color="white", linewidth=2.0, label=r"$v_{\mathrm{target}}(\hat{\kappa})$")
    ax_heat.axhline(args.kappa_max, color="white", linestyle="--", alpha=0.5,
                    label=r"$\hat{\kappa} = \kappa_{\max}$")
    ax_heat.set_xlabel(r"$|\dot{x}|$ (m/s)")
    ax_heat.set_ylabel(r"$\hat{\kappa}$ (1/m)")
    title_l = (
        f"reward = α · path · hitch · speed_shape   "
        f"(α={args.alpha},  σ_low={args.sigma_low},  σ_high={args.sigma_high},  "
        f"v_target ∈ [{args.v_target_min}, {args.v_target_max}])"
    )
    ax_heat.set_title(title_l)
    ax_heat.legend(loc="upper right", framealpha=0.85)

    # Right: 1-D slice at three κ samples.
    kappa_samples = [0.0, args.kappa_max * 0.5, args.kappa_max]
    labels = [
        rf"$\hat{{\kappa}}={kappa_samples[0]:.2f}$  (straight,  $v_{{tgt}}={args.v_target_max:.2f}$)",
        rf"$\hat{{\kappa}}={kappa_samples[1]:.2f}$  (medium,    $v_{{tgt}}={(args.v_target_max + args.v_target_min)/2:.2f}$)",
        rf"$\hat{{\kappa}}={kappa_samples[2]:.2f}$  (sharp,     $v_{{tgt}}={args.v_target_min:.2f}$)",
    ]
    for k_val, lbl in zip(kappa_samples, labels):
        vt = v_target_curve(np.array(k_val), args.v_target_min, args.v_target_max, args.kappa_max)
        z = args.alpha * speed_shape(vs, vt, args.sigma_low, args.sigma_high)
        ax_slice.plot(vs, z, label=lbl, linewidth=2.0)
        # Mark the peak.
        ax_slice.axvline(float(vt), linestyle=":", alpha=0.4)
    ax_slice.set_xlabel(r"$|\dot{x}|$ (m/s)")
    ax_slice.set_ylabel("per-step reward")
    ax_slice.set_title("reward(v) at fixed curvatures")
    ax_slice.grid(alpha=0.3)
    ax_slice.legend(loc="upper right", framealpha=0.85)
    ax_slice.set_ylim(0.0, args.alpha * 1.05)

    fig.tight_layout()

    if args.show:
        plt.show()
    else:
        out = args.out or (
            f"reward_surface_a{args.alpha}_sl{args.sigma_low}_sh{args.sigma_high}_"
            f"vmin{args.v_target_min}_vmax{args.v_target_max}.png"
        )
        out_path = Path(out).resolve()
        fig.savefig(out_path, dpi=140)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
