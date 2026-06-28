"""Clean ROS -> policy-observation transform (replaces ROSLineFollowingAdapter).

The OLD adapter instantiated a full e2e_rl pygame env and mutated its members,
with the coordinate logic scattered across four places (centerline Y-flip,
steering/hitch negation, a reverse-only beta-sign negation, and a steer-rate
negation in the node). This consolidates the ENTIRE map->truck-local convention
into ONE documented function that calls the pure tractor_trailer_rl.build_observation
— no env instantiation, no pygame, no global config.

The convention (proven equivalent to the old adapter by test_bridge_obs_parity.py
across many random spawns):
  * rotate the map-frame centerline into truck-local by (-yaw [+ pi if reverse]),
    so +X is the direction of travel;
  * apply the documented Y reflection that aligns the ROS map handedness with the
    training env frame (and negate steering + mirror the hitch accordingly);
  * re-order the centerline so it flows AHEAD of the truck (curvature reads forward);
  * place the truck at the world centre so the occupancy grid (spanning the world
    extent) contains it.

NOTE: this reproduces the convention the v22 checkpoints were TRAINED under (via
e2e_rl). Models retrained natively in tractor_trailer_rl may use a simpler
convention; this function is specifically the v22-compatible deploy transform.
"""

from __future__ import annotations

import numpy as np

from tractor_trailer_rl import build_observation, EgoState, TrailerState


def _orient_forward(xs_local, ys_local):
    """Reverse centerline order if its local tangent at the truck points -X."""
    if xs_local.size < 2:
        return xs_local, ys_local
    i = int(np.argmin(xs_local * xs_local + ys_local * ys_local))
    if 0 < i < xs_local.size - 1:
        dx_tan = xs_local[i + 1] - xs_local[i - 1]
    elif i == 0:
        dx_tan = xs_local[1] - xs_local[0]
    else:
        dx_tan = xs_local[-1] - xs_local[-2]
    if dx_tan < 0.0:
        return xs_local[::-1].copy(), ys_local[::-1].copy()
    return xs_local, ys_local


def observation_from_ros(centerline_xs, centerline_ys, ego_x, ego_y, ego_yaw,
                         steering, xd, hitch_angle, cfg, *, world_scale=1.0,
                         occ_grid=None, occ_meta=None):
    """Build the policy observation from ROS-map-frame inputs. `cfg` is a
    tractor_trailer_rl Config (its direction sets forward/reverse, vehicle_kind
    sets trailer/tractor-only, world sets the canvas/occupancy extent)."""
    reverse = cfg.is_reverse
    xs_map = np.asarray(centerline_xs, dtype=np.float64)
    ys_map = np.asarray(centerline_ys, dtype=np.float64)

    rot = -ego_yaw + (np.pi if reverse else 0.0)
    c, s = np.cos(rot), np.sin(rot)
    dx = xs_map - ego_x
    dy = ys_map - ego_y
    xs_l = ((dx * c - dy * s) * world_scale)
    ys_l = (-(dx * s + dy * c) * world_scale)          # documented Y reflection
    xs_l, ys_l = _orient_forward(xs_l.astype(np.float32), ys_l.astype(np.float32))

    ox = cfg.world.width_m / 2.0
    oy = cfg.world.height_m / 2.0
    xs = (xs_l + ox).astype(np.float32)
    ys = (ys_l + oy).astype(np.float32)

    vp = np.pi if reverse else 0.0
    ego = EgoState(x=ox, y=oy, yaw=vp, steer=-steering, xd=xd * world_scale)

    lr = cfg.vehicle.lr
    L = cfg.vehicle.trailer_length_m
    hitch_x = ox - lr * np.cos(vp)
    hitch_y = oy - lr * np.sin(vp)
    tyaw = vp + hitch_angle           # mirror of v.p - (-hitch) in the old adapter
    trailer = TrailerState(x=hitch_x - L * np.cos(tyaw),
                           y=hitch_y - L * np.sin(tyaw), yaw=tyaw)

    obs = build_observation(xs, ys, ego, trailer, cfg,
                            occ_grid=occ_grid, occ_meta=occ_meta).vector
    # Reverse hitch-sign convention (matches old adapter). ONLY valid for the
    # TRAILER layout, where obs[1] is the hitch angle. For TRACTOR_ONLY the layout
    # is [s, e_y, e_psi, k1, k2] -- obs[1] is the cross-track error, so negating it
    # flips e_y and makes the reverse policy steer away from centre. Gate on kind.
    if reverse and not cfg.is_tractor_only:
        obs = obs.copy()
        obs[1] = -obs[1]
    return obs
