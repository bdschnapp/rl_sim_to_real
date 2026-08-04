#!/usr/bin/env python3
"""Retune the reverse pure-pursuit gains for the DEPLOYED lab geometry.

Ben's sim observation (2026-07-30): with the e2e_rl fpp_rev gains the tractor
slaloms around the path to servo the trailer instead of making small
hitch-stabilising corrections. Same failure mode the cupy shunt-truck retune
fixed (REV_GAINS there: k_hitch flipped negative, k_y cut 14x) — and the shunt
trailer/wheelbase ratio (12.5/2.95=4.2) matches the lab robot (2.8/0.65=4.3).

This script replicates the bridge's env exactly (same e2erl_config overrides
as rl_bridge_node.py L249-300 + ros_env_adapter._patch_vehicle_params) and
CEM-searches gains at the deployed reverse speed. Objective per episode:
  J = 3*failed + rms(e_y_t) + 0.5*rms(e_y) + 4*mean|Δsteer| + rms(hitch)
(fail = terminated early = off-path/jackknife). Lower is better. mean|Δsteer|
is the anti-slalom term (steering activity per step).

Run:  cd Electrans_project && ./src/electrans_rl_bridge/scripts/tune_reverse_pp.py
"""
import os
import sys
import json

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, "/home/ben/Ben/Thesis/e2e_rl")

import numpy as np

# ---- replicate the bridge's config overrides (rl_bridge_node L249-300) -----
from e2erl_utils import config as e2erl_config
e2erl_config.window_width_px = 500
e2erl_config.window_height_px = 400
e2erl_config.meters_per_pixel = 0.05
e2erl_config.tractor_length_m = 1.0
e2erl_config.tractor_width_m = 0.65
e2erl_config.trailer_length_m = 2.8
e2erl_config.trailer_width_m = 0.65
e2erl_config.lane_centerline_half_width_m = 1.41
e2erl_config.lane_shoulder_m = 0.20
e2erl_config.grid_res_m = 0.05
e2erl_config.lane_sample_ds_m = 0.10
e2erl_config.tesla_model_s_vehicle_params = dict(
    e2erl_config.tesla_model_s_vehicle_params, lf=0.33, lr=0.32,
)

from Environments.LineFollowing import ReverseLidarStateObservationLineFollowingEnv
from Environments.LineFollowing import compute_curvature

SPEED = 0.4          # deployed reverse speed (bridge_velocity_max_reverse)
MAX_RATE = float(np.deg2rad(25))
N_SEARCH_PATHS = 6
N_FINAL_PATHS = 24
OUT = os.path.join(os.path.dirname(__file__), "tuned_reverse_pp.json")

E2ERL_GAINS = dict(k_hitch=0.7981827582320544, k_y=1.246169343066824,
                   k_theta=1.846962929234278, k_ff=-1.995183948132352)
SHUNT_GAINS = dict(k_hitch=-2.591, k_y=0.087, k_theta=1.663, k_ff=0.711)
KEYS = ["k_hitch", "k_y", "k_theta", "k_ff"]


def make_env():
    env = ReverseLidarStateObservationLineFollowingEnv(lidar_beams=24)
    env.vehicle.lf = 0.325   # ros_env_adapter._patch_vehicle_params
    env.vehicle.lr = 0.325
    return env


def episode(env, g, seed):
    try:
        env.reset(seed=int(seed))
    except TypeError:
        np.random.seed(int(seed)); env.reset()
    v = env.vehicle
    # reset() spawns at -config.initial_xd = -5 m/s (semi-truck scale); at lab
    # geometry that transient jackknifes anything. Start at deployment speed.
    v.xd = -SPEED
    dt = float(v.dt)
    eyt_l, ey_l, hitch_l, ds_l = [], [], [], []
    prev_s = float(v.s)
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated) and steps < 4000:
        psi2 = float(v.p - v.trailer.yaw)
        e_y_t, e_th_t = env.get_trailer_errors()
        kappa = float(compute_curvature(env, lookahead_steps=10))
        delta_des = (g["k_hitch"] * psi2 + g["k_ff"] * kappa
                     + g["k_y"] * float(e_y_t) + g["k_theta"] * float(e_th_t))
        rate = float(np.clip((delta_des - float(v.s)) / dt, -MAX_RATE, MAX_RATE))
        _, _, terminated, truncated, _ = env.step(np.array([rate, -SPEED], dtype=np.float32))
        e_y, _ = env.get_vehicle_errors()
        eyt_l.append(float(e_y_t)); ey_l.append(float(e_y))
        hitch_l.append(psi2); ds_l.append(abs(float(v.s) - prev_s))
        prev_s = float(v.s)
        steps += 1
    # terminated=True fires for BOTH failure (off-path/jackknife) and success
    # (env sets .success when crossing the far end) — disambiguate via .success.
    failed = bool(terminated and not getattr(env, "success", False))
    rms = lambda a: float(np.sqrt(np.mean(np.square(a)))) if a else 9.9
    J = (3.0 * failed + rms(eyt_l) + 0.5 * rms(ey_l)
         + 4.0 * float(np.mean(ds_l) if ds_l else 9.9) + rms(hitch_l))
    return J, failed, rms(eyt_l), rms(ey_l), float(np.mean(ds_l) if ds_l else 9.9)


def score(env, g, seeds):
    r = [episode(env, g, s) for s in seeds]
    return float(np.mean([x[0] for x in r])), r


def main():
    env = make_env()
    seeds = list(range(1000, 1000 + N_SEARCH_PATHS))

    print("== baselines ==", flush=True)
    results = {}
    for name, g in [("e2erl", E2ERL_GAINS), ("shunt", SHUNT_GAINS)]:
        J, r = score(env, g, seeds)
        fails = sum(x[1] for x in r)
        print(f"{name}: J={J:.3f} fails={fails}/{len(seeds)} "
              f"rms_eyt={np.mean([x[2] for x in r]):.3f} rms_ey={np.mean([x[3] for x in r]):.3f} "
              f"steer_act={np.mean([x[4] for x in r]):.4f}", flush=True)
        results[name] = (J, dict(g))

    # CEM around the better baseline
    best_name = min(results, key=lambda k: results[k][0])
    mu = np.array([results[best_name][1][k] for k in KEYS])
    sigma = np.abs(mu) * 0.5 + np.array([0.5, 0.1, 0.4, 0.4])
    rng = np.random.default_rng(0)
    best_J, best_g = results[best_name][0], dict(results[best_name][1])
    print(f"== CEM init from {best_name} ==", flush=True)
    for it in range(6):
        cand = rng.normal(mu, sigma, size=(16, len(KEYS)))
        scored = []
        for c in cand:
            g = dict(zip(KEYS, map(float, c)))
            J, r = score(env, g, seeds)
            scored.append((J, g, sum(x[1] for x in r)))
        scored.sort(key=lambda t: t[0])
        elite = scored[:4]
        mu = np.mean([[e[1][k] for k in KEYS] for e in elite], axis=0)
        sigma = np.std([[e[1][k] for k in KEYS] for e in elite], axis=0) + 0.02
        if elite[0][0] < best_J:
            best_J, best_g = elite[0][0], dict(elite[0][1])
        print(f"iter {it}: best J={elite[0][0]:.3f} fails={elite[0][2]} "
              f"gains={ {k: round(v,3) for k, v in elite[0][1].items()} }", flush=True)

    print("== final eval (24 fresh paths) ==", flush=True)
    fseeds = list(range(5000, 5000 + N_FINAL_PATHS))
    for name, g in [("e2erl", E2ERL_GAINS), ("shunt", SHUNT_GAINS), ("cem", best_g)]:
        J, r = score(env, g, fseeds)
        fails = sum(x[1] for x in r)
        print(f"{name}: J={J:.3f} fails={fails}/{len(fseeds)} "
              f"rms_eyt={np.mean([x[2] for x in r]):.3f} rms_ey={np.mean([x[3] for x in r]):.3f} "
              f"steer_act={np.mean([x[4] for x in r]):.4f}", flush=True)
        if name == "cem":
            json.dump({"gains": g, "J": J, "fails": fails, "speed": SPEED},
                      open(OUT, "w"), indent=2)
    print(f"wrote {OUT}", flush=True)
    print("TUNE_DONE", flush=True)


main()
