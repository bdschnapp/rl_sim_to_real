#!/usr/bin/env python3
"""
Render sample training paths to PNG so we can visually verify each
path-kind generator produces the geometry it advertises.

These match the path-generation logic in train_lab_model._patch_path_generator
exactly. If a kind's PNG doesn't look like its name, the training-time
generator is the bug — fix it here, propagate to train_lab_model.

Usage
-----
    python plot_path_samples.py                # all kinds, 3 samples each
    python plot_path_samples.py --kind sharp   # just one kind
    python plot_path_samples.py --n-samples 5  # more samples per kind
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline


# Match the env's path-x grid: arange(int(x0), int(x_end), 1) with
# x0=5, x_end=145 (canvas width 150 m, 5 m margin).
X0 = 5
X_END = 145
XX = np.arange(X0, X_END, 1).astype(float)
N = len(XX)

# Canvas extent (must match train_lab_model + e2e_rl defaults at lab scale):
VERT_OFFSET = 45.0
CANVAS_Y_MAX = 90.0
CANVAS_Y_MIN = 0.0


def make_straight(rng: np.random.Generator) -> np.ndarray:
    return np.zeros(N, dtype=float)


def make_gentle(rng: np.random.Generator) -> np.ndarray:
    n_ctrl = 6
    ctrl_x = np.linspace(X0, X_END, n_ctrl)
    ctrl_y = -1.0 + 2.0 * rng.random(n_ctrl)
    ctrl_y[0] = (-0.25 / 50) + rng.random() * (0.5 / 50)
    cs = CubicSpline(ctrl_x, ctrl_y, bc_type=((1, 0.0), "not-a-knot"))
    return cs(XX) * 5.0


def make_sharp(rng: np.random.Generator) -> np.ndarray:
    n_bends = int(rng.integers(4, 9))
    margin = 20.0
    base_centers = np.linspace(X0 + margin, X_END - margin, n_bends)
    gap = (X_END - X0 - 2 * margin) / max(1, n_bends - 1)
    jitter = (rng.random(n_bends) - 0.5) * gap * 0.4
    bend_xs = sorted((base_centers + jitter).tolist())
    sign = 1.0 if rng.random() < 0.5 else -1.0
    y = np.zeros(N, dtype=float)
    for bend_x in bend_xs:
        width = float(rng.uniform(1.0, 1.5))
        dy = sign * float(rng.uniform(2.5, 4.0))
        sign = -sign
        ramp = (np.tanh((XX - bend_x) / width) + 1.0) * 0.5
        y += dy * ramp
    return y


def make_sustained_turn(rng: np.random.Generator) -> np.ndarray:
    """Legacy single-bend sustained turn — kept for reference / eval."""
    theta_final = float(rng.uniform(np.deg2rad(25.0), np.deg2rad(45.0)))
    sign = 1.0 if rng.random() < 0.5 else -1.0
    final_slope = sign * float(np.tan(theta_final))
    width = float(rng.uniform(0.8, 1.5))

    safety = 5.0
    max_post_bend = (CANVAS_Y_MAX - VERT_OFFSET) - safety  # 40 m
    min_bend_x = max(X0 + 25.0, X_END - max_post_bend / abs(final_slope))
    max_bend_x = X_END - 10.0
    if min_bend_x >= max_bend_x:
        bend_x = min_bend_x
    else:
        bend_x = float(rng.uniform(min_bend_x, max_bend_x))

    arg = (XX - bend_x) / width
    arg_0 = (X0 - bend_x) / width
    ln_cosh = np.log(np.cosh(arg))
    ln_cosh_0 = float(np.log(np.cosh(arg_0)))
    return (final_slope / 2.0) * ((XX - X0) + width * (ln_cosh - ln_cosh_0))


def make_winding(rng: np.random.Generator) -> np.ndarray:
    """Chained alternating-sign sustained turns. The active training-time
    sharp/turning mode in v15+: vehicle is continuously either in a
    transition (high κ) or driving at a sustained ±s slope."""
    n_bends = int(rng.choice([3, 5]))
    s = float(rng.uniform(0.5, 0.8))
    width = float(rng.uniform(0.8, 1.5))
    initial_sign = 1.0 if rng.random() < 0.5 else -1.0

    margin = 18.0
    base_xs = np.linspace(X0 + margin, X_END - margin, n_bends)
    gap = (X_END - X0 - 2.0 * margin) / max(1, n_bends - 1)
    jitter = (rng.random(n_bends) - 0.5) * gap * 0.3
    bend_xs = sorted((base_xs + jitter).tolist())

    interval_slopes = [0.0]
    for i in range(n_bends - 1):
        sign = initial_sign if (i % 2 == 0) else -initial_sign
        interval_slopes.append(sign * s)
    interval_slopes.append(0.0)

    y = np.zeros(N, dtype=float)
    for bend_x, sp, sn in zip(bend_xs, interval_slopes[:-1], interval_slopes[1:]):
        delta = sn - sp
        if abs(delta) < 1e-9:
            continue
        arg = (XX - bend_x) / width
        arg_0 = (X0 - bend_x) / width
        ln_cosh = np.log(np.cosh(arg))
        ln_cosh_0 = float(np.log(np.cosh(arg_0)))
        y += (delta / 2.0) * ((XX - X0) + width * (ln_cosh - ln_cosh_0))
    return y


GENERATORS = {
    "straight":       make_straight,
    "gentle":         make_gentle,
    "sharp":          make_sharp,
    "sustained_turn": make_sustained_turn,
    "winding":        make_winding,
}


def compute_curvature(y: np.ndarray) -> np.ndarray:
    """Numerical curvature κ(x) for y(x). Matches the env's finite-diff
    convention so the values are directly comparable to obs κ̂."""
    x_p = np.gradient(XX)
    y_p = np.gradient(y)
    x_pp = np.gradient(x_p)
    y_pp = np.gradient(y_p)
    denom = (x_p ** 2 + y_p ** 2) ** 1.5 + 1e-9
    kappa = (x_p * y_pp - y_p * x_pp) / denom
    return np.clip(kappa, -0.6, 0.6)


def plot_kind(kind: str, n_samples: int, out_dir: Path, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    gen = GENERATORS[kind]

    fig, (ax_path, ax_curv) = plt.subplots(2, 1, figsize=(14, 7),
                                            gridspec_kw={"height_ratios": [3, 1]})
    for i in range(n_samples):
        y_local = gen(rng)
        y_world = y_local + VERT_OFFSET
        kappa = compute_curvature(y_local)
        spawn_idx = int(0.1 * (len(XX) - 1))

        ax_path.plot(XX, y_world, linewidth=1.8, alpha=0.85,
                     label=f"sample {i}: κ_max={np.abs(kappa).max():.2f}")
        ax_path.plot(XX[spawn_idx], y_world[spawn_idx], "o",
                     markersize=8, markeredgecolor="black",
                     markerfacecolor="white", zorder=5)
        ax_curv.plot(XX, np.abs(kappa), alpha=0.85)

    # Canvas / lane indication
    ax_path.axhline(VERT_OFFSET, color="gray", linestyle=":", alpha=0.4,
                    label=f"vert_offset = {VERT_OFFSET}")
    ax_path.axhline(CANVAS_Y_MAX, color="red", linestyle="--", alpha=0.4,
                    label=f"canvas y = {CANVAS_Y_MAX} m (top)")
    ax_path.axhline(CANVAS_Y_MIN, color="red", linestyle="--", alpha=0.4)
    ax_path.set_xlabel("x (m)")
    ax_path.set_ylabel("y (m)")
    ax_path.set_xlim(X0, X_END)
    ax_path.set_ylim(CANVAS_Y_MIN - 5, CANVAS_Y_MAX + 5)
    ax_path.set_title(f"path kind: {kind!r}  (spawn point = white circle)")
    ax_path.legend(loc="upper left", fontsize=8, framealpha=0.85)
    ax_path.grid(alpha=0.3)

    ax_curv.axhline(0.3, color="orange", linestyle=":", alpha=0.5,
                    label="κ obs clip = 0.3")
    ax_curv.axhline(0.5, color="red", linestyle=":", alpha=0.5,
                    label="MVSL corner κ = 0.5")
    ax_curv.set_xlabel("x (m)")
    ax_curv.set_ylabel("|κ| (1/m)")
    ax_curv.set_xlim(X0, X_END)
    ax_curv.set_ylim(0, 0.6)
    ax_curv.legend(loc="upper right", fontsize=8, framealpha=0.85)
    ax_curv.grid(alpha=0.3)

    fig.tight_layout()
    out = out_dir / f"path_{kind}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--kind", choices=["all"] + list(GENERATORS),
                        default="all")
    parser.add_argument("--n-samples", dest="n_samples", type=int, default=3)
    parser.add_argument("--out-dir", default="/tmp")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    kinds = list(GENERATORS) if args.kind == "all" else [args.kind]
    for k in kinds:
        out = plot_kind(k, args.n_samples, out_dir, seed=args.seed)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
