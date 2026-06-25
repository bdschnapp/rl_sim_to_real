#!/usr/bin/env python3
"""Convert a rosbag2 sqlite recording of robot driving into a training
dataset for the learned dynamics model.

Reads each relevant topic, deserialises messages, time-aligns onto a
uniform 10 Hz grid (linear interpolation for continuous signals,
zero-order hold for commands), derives any missing geometry (trailer
pose from truck pose + hitch angle), and writes the resulting tensors
to an `.npz` file.

Missing topics are tolerated — the script will warn and emit zeros for
the affected columns, so the pipeline can be exercised on the existing
short bags before a full-topic bag is captured.

Usage
-----
    # ROS env must be sourced first so rclpy/autoware msgs are importable.
    source /opt/ros/humble/setup.bash
    source ~/Ben/Thesis/Electrans_project/install/setup.bash  # if local install

    /home/ben/Ben/Thesis/e2e_rl/venv/bin/python3 \\
      src/electrans_rl_bridge/scripts/dynamics/bag_to_dataset.py \\
      --bag bags/sample_existing/dynamics_topics_only \\
      --out bags/sample_existing/dataset.npz \\
      --action-window 11

Outputs (`.npz`):
    state          (T, 8)         body-frame state per timestep
    action         (T, 2)         command at this timestep
    action_history (T, N+1, 2)    last N+1 actions (oldest-first)
    state_next     (T, 8)         body-frame state at the NEXT timestep
    state_delta    (T, 8)         state_next - state (target for the NN)
    timestamps     (T,)           grid timestamps (ns)
    topic_status   (dict)         which topics were found
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Default config maps each *role* needed by the dynamics model to one or
# more candidate ROS topic names. The first match in the bag wins. We
# keep this flexible so different deployments (sim vs real, autoware vs
# others) can be ingested with the same script.
TOPIC_ROLES: Dict[str, List[str]] = {
    "control_cmd": [
        "/control/command/control_cmd",
        "/control/command/control_cmd_filtered",
    ],
    "steering_status": ["/vehicle/status/steering_status"],
    "velocity_status": ["/vehicle/status/velocity_status"],
    "trailer_state": [
        "/vehicle/trailer_state",
        "/vehicle/status/trailer_state",
    ],
    "kinematic_state": [
        "/localization/kinematic_state",
        "/odom",
    ],
    "imu": ["/sensing/imu/imu_data", "/imu/data"],
    "gear_cmd": ["/control/command/gear_cmd"],
}

# Topic type strings as they appear in rosbag metadata. Used to pick the
# right deserialiser. The role → type mapping is fixed.
TOPIC_TYPES: Dict[str, str] = {
    "control_cmd": "autoware_control_msgs/msg/Control",
    "steering_status": "autoware_vehicle_msgs/msg/SteeringReport",
    "velocity_status": "autoware_vehicle_msgs/msg/VelocityReport",
    "trailer_state": "autoware_vehicle_msgs/msg/TrailerState",
    "kinematic_state": "nav_msgs/msg/Odometry",
    "imu": "sensor_msgs/msg/Imu",
    "gear_cmd": "autoware_vehicle_msgs/msg/GearCommand",
}

# Add autoware-msgs install paths if present in the workspace.
_AUTOWARE_MSG_PATHS = [
    "/home/ben/Ben/Thesis/Electrans_project/install/autoware_vehicle_msgs/local/lib/python3.10/dist-packages",
    "/home/ben/Ben/Thesis/Electrans_project/install/autoware_control_msgs/local/lib/python3.10/dist-packages",
]
for p in _AUTOWARE_MSG_PATHS:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)


def _import_message_classes() -> Dict[str, type]:
    """Lazy-import the message classes only for topics we'll actually
    read. Lets the script still inspect a bag's topic list when some
    autoware packages aren't installed."""
    classes: Dict[str, type] = {}
    type_to_role: Dict[str, str] = {v: k for k, v in TOPIC_TYPES.items()}
    for type_str, role in type_to_role.items():
        try:
            pkg_name, _, msg_name = type_str.replace("/msg/", "/").rpartition("/")
            mod = __import__(f"{pkg_name}.msg", fromlist=[msg_name])
            classes[type_str] = getattr(mod, msg_name)
        except Exception as e:
            print(f"[bag_to_dataset] skip {type_str}: {e}")
    return classes


# ----------------------------------------------------------------------------
# Bag reading
# ----------------------------------------------------------------------------


def _open_bag(bag_path: str):
    """Open a rosbag2 sqlite bag. Returns (SequentialReader, topic_metadata)."""
    import rosbag2_py
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions

    # If the user pointed us at the dir, find the .db3 inside it.
    p = Path(bag_path)
    if p.is_dir():
        # rosbag2 wants the bag DIRECTORY, not the .db3 file.
        storage_uri = str(p)
    else:
        storage_uri = str(p)

    storage_options = StorageOptions(uri=storage_uri, storage_id="sqlite3")
    converter_options = ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader = SequentialReader()
    reader.open(storage_options, converter_options)
    topic_meta = reader.get_all_topics_and_types()
    return reader, topic_meta


def _assign_topics_to_roles(
    topic_meta, msg_classes: Dict[str, type]
) -> Dict[str, Tuple[str, type]]:
    """Returns role → (topic_name, msg_class). Topics not matching any
    role we care about are ignored."""
    name_to_type = {tm.name: tm.type for tm in topic_meta}
    role_assignments: Dict[str, Tuple[str, type]] = {}
    for role, candidates in TOPIC_ROLES.items():
        expected_type = TOPIC_TYPES[role]
        cls = msg_classes.get(expected_type)
        if cls is None:
            continue
        for candidate in candidates:
            if candidate in name_to_type and name_to_type[candidate] == expected_type:
                role_assignments[role] = (candidate, cls)
                break
    return role_assignments


def _read_messages(
    bag_path: str, role_assignments: Dict[str, Tuple[str, type]]
) -> Dict[str, List[Tuple[int, object]]]:
    """Pulls every relevant message from the bag and deserialises it.
    Returns role → list of (timestamp_ns, deserialised_msg) tuples,
    sorted ascending by timestamp.
    """
    from rclpy.serialization import deserialize_message

    reader, _ = _open_bag(bag_path)
    # Map topic name → (role, msg_class) for fast lookup in the loop.
    topic_to_role = {tn: (role, cls) for role, (tn, cls) in role_assignments.items()}

    bins: Dict[str, List[Tuple[int, object]]] = {role: [] for role in role_assignments}
    while reader.has_next():
        topic, data, t = reader.read_next()
        match = topic_to_role.get(topic)
        if match is None:
            continue
        role, cls = match
        try:
            msg = deserialize_message(data, cls)
        except Exception as e:
            print(f"[bag_to_dataset] deserialise fail on {topic}: {e}")
            continue
        bins[role].append((int(t), msg))
    for role in bins:
        bins[role].sort(key=lambda x: x[0])
    return bins


# ----------------------------------------------------------------------------
# Per-role extractors
# ----------------------------------------------------------------------------
#
# Each extractor turns a list of (t_ns, msg) into one or more
# (timestamps, values) timeseries dictionaries. Keys here are the
# field names that the alignment stage uses.


def _quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Standard quaternion → yaw."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return float(math.atan2(siny_cosp, cosy_cosp))


def _extract_role(role: str, msgs: List[Tuple[int, object]]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Convert messages of a given role into named timeseries.

    Returns dict of {field_name: (timestamps_ns[N], values[N])} arrays.
    """
    if not msgs:
        return {}
    ts = np.fromiter((m[0] for m in msgs), dtype=np.int64)

    if role == "control_cmd":
        steer = np.fromiter(
            (float(m[1].lateral.steering_tire_angle) for m in msgs), dtype=np.float32
        )
        vel = np.fromiter(
            (float(m[1].longitudinal.velocity) for m in msgs), dtype=np.float32
        )
        return {"u_steer_cmd": (ts, steer), "u_vel_cmd": (ts, vel)}

    if role == "steering_status":
        s = np.fromiter(
            (float(m[1].steering_tire_angle) for m in msgs), dtype=np.float32
        )
        return {"steer": (ts, s)}

    if role == "velocity_status":
        # VelocityReport has longitudinal_velocity, lateral_velocity, heading_rate.
        vx = np.fromiter(
            (float(m[1].longitudinal_velocity) for m in msgs), dtype=np.float32
        )
        vy = np.fromiter(
            (float(m[1].lateral_velocity) for m in msgs), dtype=np.float32
        )
        wz = np.fromiter(
            (float(m[1].heading_rate) for m in msgs), dtype=np.float32
        )
        return {"xd": (ts, vx), "yd": (ts, vy), "pd_status": (ts, wz)}

    if role == "trailer_state":
        beta = np.fromiter(
            (float(m[1].hitch_angle) for m in msgs), dtype=np.float32
        )
        beta_rate = np.fromiter(
            (float(getattr(m[1], "hitch_rate", 0.0)) for m in msgs), dtype=np.float32
        )
        return {"beta": (ts, beta), "beta_rate": (ts, beta_rate)}

    if role == "kinematic_state":
        # nav_msgs/Odometry: pose.position.x/y, pose.orientation, twist.linear, twist.angular
        x = np.fromiter(
            (float(m[1].pose.pose.position.x) for m in msgs), dtype=np.float32
        )
        y = np.fromiter(
            (float(m[1].pose.pose.position.y) for m in msgs), dtype=np.float32
        )
        yaw = np.fromiter(
            (
                _quat_to_yaw(
                    m[1].pose.pose.orientation.x,
                    m[1].pose.pose.orientation.y,
                    m[1].pose.pose.orientation.z,
                    m[1].pose.pose.orientation.w,
                )
                for m in msgs
            ),
            dtype=np.float32,
        )
        return {"x": (ts, x), "y": (ts, y), "p": (ts, yaw)}

    if role == "imu":
        # IMU angular_velocity.z is the most reliable yaw_rate signal.
        wz = np.fromiter(
            (float(m[1].angular_velocity.z) for m in msgs), dtype=np.float32
        )
        return {"pd_imu": (ts, wz)}

    if role == "gear_cmd":
        gear = np.fromiter((int(m[1].command) for m in msgs), dtype=np.int16)
        return {"gear": (ts, gear)}

    return {}


# ----------------------------------------------------------------------------
# Time alignment
# ----------------------------------------------------------------------------


def _align_to_grid(
    series: Dict[str, Tuple[np.ndarray, np.ndarray]],
    grid_ts: np.ndarray,
    zero_order_hold_keys: set,
) -> Dict[str, np.ndarray]:
    """Resample every timeseries onto the shared grid.

    Continuous signals: linear interpolation.
    Step / command signals: zero-order hold (use the most recent value).
    """
    out: Dict[str, np.ndarray] = {}
    for key, (t, v) in series.items():
        if t.size == 0:
            out[key] = np.zeros_like(grid_ts, dtype=np.float32)
            continue
        if key in zero_order_hold_keys:
            # Zero-order hold (commands): for each grid point, find the
            # most recent sample at or before it.
            idx = np.searchsorted(t, grid_ts, side="right") - 1
            idx = np.clip(idx, 0, t.size - 1)
            out[key] = v[idx].astype(np.float32)
        else:
            # Linear interpolation in float64 to avoid precision loss on
            # large timestamps. Clip the grid to the available range; out-
            # of-range queries get the boundary value (np.interp default).
            t_f = t.astype(np.float64)
            v_f = v.astype(np.float64)
            g_f = grid_ts.astype(np.float64)
            out[key] = np.interp(g_f, t_f, v_f).astype(np.float32)
    return out


def _build_grid(
    series: Dict[str, Tuple[np.ndarray, np.ndarray]],
    grid_hz: float,
) -> np.ndarray:
    """Pick a [t_start, t_end] window covered by ALL essential timeseries
    and return uniform timestamps at the requested rate."""
    if not series:
        return np.array([], dtype=np.int64)
    # Start = max of each series' first ts; End = min of each series' last ts.
    starts = [t[0] for (t, _) in series.values() if t.size > 0]
    ends = [t[-1] for (t, _) in series.values() if t.size > 0]
    if not starts:
        return np.array([], dtype=np.int64)
    t_start = max(starts)
    t_end = min(ends)
    if t_end <= t_start:
        return np.array([], dtype=np.int64)
    step_ns = int(1e9 / grid_hz)
    return np.arange(t_start, t_end + 1, step_ns, dtype=np.int64)


# ----------------------------------------------------------------------------
# Body-frame state assembly
# ----------------------------------------------------------------------------


def _assemble_state(
    aligned: Dict[str, np.ndarray],
    trailer_length: float,
    truck_lr: float,
    have: Dict[str, bool],
) -> np.ndarray:
    """Build the (T, 8) body-frame state matrix from aligned signals.

    Body-frame state columns: [xd, yd, pd, s, dx_t, dy_t, beta, t_yaw_rate]

    Where data is missing (e.g. no trailer topic) the column is zero —
    the caller's `topic_status` dict records the gap so downstream code
    can ignore those samples or those output dimensions.
    """
    T = next(iter(aligned.values())).size if aligned else 0
    if T == 0:
        return np.zeros((0, 8), dtype=np.float32)

    state = np.zeros((T, 8), dtype=np.float32)
    state[:, 0] = aligned.get("xd", np.zeros(T))
    state[:, 1] = aligned.get("yd", np.zeros(T))
    # Yaw rate: prefer IMU (high rate) over VelocityReport (often
    # missing). If we have kinematic_state but no IMU we'll later
    # finite-difference the yaw column.
    pd = aligned.get("pd_imu")
    if pd is None or not have.get("imu"):
        pd = aligned.get("pd_status")
    if pd is None or pd.size == 0:
        pd = np.zeros(T, dtype=np.float32)
    state[:, 2] = pd

    state[:, 3] = aligned.get("steer", np.zeros(T))

    # Trailer position relative to truck (body frame), computed from
    # hitch angle. If we don't have hitch_angle, leave at zero.
    beta = aligned.get("beta", np.zeros(T, dtype=np.float32))
    state[:, 6] = beta

    if have.get("kinematic_state") and have.get("trailer_state"):
        # We know the truck pose and the hitch angle. Trailer axle is
        # `trailer_length` behind the hitch in trailer-yaw direction;
        # hitch is `lr` behind the truck CG in truck-yaw direction.
        p = aligned["p"]
        cos_p = np.cos(p)
        sin_p = np.sin(p)
        # Hitch (world)
        hx = aligned["x"] - truck_lr * cos_p
        hy = aligned["y"] - truck_lr * sin_p
        # Trailer yaw (world) and axle position (world)
        trailer_yaw = p - beta
        ct = np.cos(trailer_yaw)
        st = np.sin(trailer_yaw)
        tx = hx - trailer_length * ct
        ty = hy - trailer_length * st
        # World → body-frame trailer offset
        dx_w = tx - aligned["x"]
        dy_w = ty - aligned["y"]
        state[:, 4] = cos_p * dx_w + sin_p * dy_w
        state[:, 5] = -sin_p * dx_w + cos_p * dy_w

    # Trailer yaw rate: derive from beta_rate if we have it. β = p − tψ,
    # so tψ̇ = ṗ − β̇. ṗ is just the truck yaw rate (col 2).
    if "beta_rate" in aligned:
        state[:, 7] = state[:, 2] - aligned["beta_rate"]

    return state


def _assemble_actions(aligned: Dict[str, np.ndarray]) -> np.ndarray:
    """Build the (T, 2) command matrix."""
    T = next(iter(aligned.values())).size if aligned else 0
    if T == 0:
        return np.zeros((0, 2), dtype=np.float32)
    actions = np.zeros((T, 2), dtype=np.float32)
    actions[:, 0] = aligned.get("u_steer_cmd", np.zeros(T))
    actions[:, 1] = aligned.get("u_vel_cmd", np.zeros(T))
    return actions


def _build_action_history(actions: np.ndarray, window: int) -> np.ndarray:
    """Stack a (T, window, 2) tensor whose row t is [u_{t-window+1}, …, u_t].

    For t < window-1 we pad with zeros at the front (so the model
    sees the equivalent of "no commands yet" at episode start)."""
    T, D = actions.shape
    out = np.zeros((T, window, D), dtype=np.float32)
    for k in range(window):
        # Position k in the output history = action at time (t - (window-1-k))
        src_t = np.arange(T) - (window - 1 - k)
        valid = src_t >= 0
        out[valid, k, :] = actions[src_t[valid], :]
    return out


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True, help="Path to rosbag2 directory")
    ap.add_argument("--out", required=True, help="Output .npz path")
    ap.add_argument("--action-window", type=int, default=11)
    ap.add_argument("--grid-hz", type=float, default=10.0,
                    help="Target sample rate (10 Hz matches the bridge tick)")
    ap.add_argument("--trailer-length", type=float, default=2.0)
    ap.add_argument("--truck-lr", type=float, default=0.32)
    args = ap.parse_args()

    msg_classes = _import_message_classes()
    reader, topic_meta = _open_bag(args.bag)
    role_assignments = _assign_topics_to_roles(topic_meta, msg_classes)

    print(f"[bag_to_dataset] reading bag at {args.bag}")
    print(f"[bag_to_dataset] role -> topic mapping:")
    for role in TOPIC_ROLES:
        if role in role_assignments:
            print(f"    {role:<18} <- {role_assignments[role][0]}")
        else:
            print(f"    {role:<18} <- (MISSING — column will be zero)")

    bins = _read_messages(args.bag, role_assignments)
    print(f"[bag_to_dataset] message counts:")
    for role, msgs in bins.items():
        print(f"    {role:<18} {len(msgs):>8} msgs")

    # Per-role extract
    series: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for role, msgs in bins.items():
        series.update(_extract_role(role, msgs))

    if not series:
        print("[bag_to_dataset] ERROR: no usable timeseries extracted from bag")
        return 1

    grid_ts = _build_grid(series, args.grid_hz)
    if grid_ts.size < 2:
        print("[bag_to_dataset] ERROR: no overlapping time window across topics")
        return 1
    print(f"[bag_to_dataset] grid: {grid_ts.size} samples at {args.grid_hz} Hz "
          f"({(grid_ts[-1] - grid_ts[0])/1e9:.1f} s)")

    # Command signals use zero-order hold; everything else interpolates.
    zoh_keys = {"u_steer_cmd", "u_vel_cmd", "gear"}
    aligned = _align_to_grid(series, grid_ts, zoh_keys)

    have = {role: (role in role_assignments) for role in TOPIC_ROLES}

    state = _assemble_state(aligned, args.trailer_length, args.truck_lr, have)
    actions = _assemble_actions(aligned)
    action_history = _build_action_history(actions, args.action_window)

    # State delta target: next-step state minus current state.
    state_next = np.roll(state, -1, axis=0)
    state_next[-1] = state[-1]  # last row repeats (will be dropped by training)
    state_delta = (state_next - state).astype(np.float32)

    # Wrap-aware delta for the hitch angle so β ∈ (−π, π].
    state_delta[:, 6] = (state_delta[:, 6] + np.pi) % (2 * np.pi) - np.pi

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        state=state,
        action=actions,
        action_history=action_history,
        state_next=state_next,
        state_delta=state_delta,
        timestamps=grid_ts,
        topic_status=np.array(
            [(k, str(int(v))) for k, v in have.items()], dtype=object
        ),
    )
    print(f"[bag_to_dataset] wrote {out_path}  state={state.shape}  "
          f"action={actions.shape}  history={action_history.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
