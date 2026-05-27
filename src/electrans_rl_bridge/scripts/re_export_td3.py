#!/usr/bin/env python3
"""Re-export a TD3 SB3 checkpoint as a portable policy .pth.

Run with the numpy-2.x venv that pickled the source checkpoint:
  /home/ben/Ben/Thesis/e2e_rl/venv/bin/python3 \
      src/electrans_rl_bridge/scripts/re_export_td3.py \
      /home/ben/Ben/Thesis/e2e_rl/models/forward/bev_scaled_cnn/dense/best_model.zip \
      /home/ben/Ben/Thesis/e2e_rl/models/forward/bev_scaled_cnn/dense/best_model.policy.pth

The output .pth is loadable from any numpy version because torch.save() doesn't
go through cloudpickle for tensor state dicts. The accompanying
.policy_kwargs.pkl (next to the .pth) carries the kwargs the bridge needs to
re-instantiate the policy.

This script must run in the venv that originally produced the checkpoint --
otherwise the load itself will fail with the same numpy errors the bridge hit.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import torch


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("src_zip", help="SB3 TD3 .zip checkpoint to re-export")
    p.add_argument("dst_pth", help="output .pth (companion .policy_kwargs.pkl is written next to it)")
    p.add_argument(
        "--e2e-rl-path",
        default="/home/ben/Ben/Thesis/e2e_rl",
        help="path containing Models/ and Environments/ for SB3 unpickling",
    )
    args = p.parse_args()

    if args.e2e_rl_path not in sys.path:
        sys.path.insert(0, args.e2e_rl_path)

    # Required to be importable for SB3 to unpickle the features extractor.
    from Models.CNNFeatureExtractor import CNNFeatureExtractor  # noqa: F401
    from stable_baselines3 import TD3

    print(f"Loading {args.src_zip} ...")
    model = TD3.load(args.src_zip, env=None, device="cpu")
    print(f"  policy class: {type(model.policy).__name__}")
    print(f"  action_space: {model.action_space}")
    print(f"  observation_space: {model.observation_space}")

    torch.save(model.policy.state_dict(), args.dst_pth)
    print(f"Wrote {args.dst_pth}")

    # Pick the matching e2e_rl env class from the observation space shape, so
    # the bridge can re-instantiate the right env at runtime without the user
    # having to remember which checkpoint goes with which obs pipeline.
    env_class_module, env_class_name, env_kwargs = _detect_env(model)
    print(f"  env: {env_class_module}.{env_class_name} kwargs={env_kwargs}")

    # The policy needs the same constructor kwargs at load time. Save them as
    # plain pickle (numpy not involved -- they're Python primitives + the
    # features_extractor_class which lives in e2e_rl, importable on both sides).
    meta_path = os.path.splitext(args.dst_pth)[0] + ".policy_kwargs.pkl"
    meta = {
        "policy_class_name": type(model.policy).__name__,
        "policy_kwargs": model.policy_kwargs,
        "env_class_module": env_class_module,
        "env_class_name": env_class_name,
        "env_kwargs": env_kwargs,
        # Spaces re-built by the bridge from the live env adapter, not from here.
    }
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f, protocol=4)
    print(f"Wrote {meta_path}")

    return 0


def _detect_env(model):
    """Pick the e2e_rl env class + kwargs that matches the model's observation
    space. The three forward lane-following options are:

      - BevObservationLineFollowingEnv (Dict obs: image 32x32x1 + 8-dim vector)
      - StateObservationLineFollowingEnv (Box(8,))
      - LidarStateObservationLineFollowingEnv (Box(8 + lidar_beams,))
    """
    from gymnasium import spaces

    obs = model.observation_space
    if isinstance(obs, spaces.Dict) and {"image", "vector"} <= set(obs.spaces):
        return "Environments.LineFollowing", "BevObservationLineFollowingEnv", {}
    if isinstance(obs, spaces.Box) and obs.shape == (8,):
        return "Environments.LineFollowing", "StateObservationLineFollowingEnv", {}
    if isinstance(obs, spaces.Box) and len(obs.shape) == 1 and obs.shape[0] > 8:
        return (
            "Environments.ObstacleAvoidance",
            "LidarStateObservationLineFollowingEnv",
            {"lidar_beams": int(obs.shape[0] - 8)},
        )
    raise RuntimeError(
        f"re_export_td3: cannot map observation_space {obs!r} to a known env class. "
        "Add a new branch to _detect_env() if this is a new obs pipeline."
    )


if __name__ == "__main__":
    sys.exit(main())
