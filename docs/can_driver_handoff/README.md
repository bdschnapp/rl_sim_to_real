# Hunter CAN driver — handoff package

This package documents the ROS 2 ↔ CAN vehicle interface for the Electrans
robot's AgileX Hunter chassis, and includes the exact bridge node source that
runs on the robot today. Use it to write a CAN driver that matches the deployed
behaviour.

## Start here

1. **`CAN_INTERFACE.md`** — the full specification. Read this first. It defines
   the wire protocol (CAN IDs, byte layouts, endianness, scaling, the enable
   handshake, feedback decode) and the ROS 2 topic contract the driver must
   preserve. Sections 3–7 are everything you need to reproduce the behaviour;
   section 11 is a self-contained reference implementation.

2. **`canbridge_package/`** — the authoritative source, as deployed. This is a
   ROS 2 (Humble) `ament_cmake` package with two nodes:
   - `canbridge/canbridge.py` — **the protocol translator** (Autoware `Control`
     ↔ Hunter CAN frames, scaling, feedback → vehicle status). This is the file
     the spec is derived from; read it alongside the spec.
     - `src/ros2socketcan.cpp` / `.h` — a thin raw SocketCAN ↔ `can_msgs/Frame`
     bridge (no protocol logic; applies no scaling).
   - `launch/can_bridge.launch.py` — launches both nodes, param `can_interface`
     (default `can1`).
   - `config/can_params.yaml` — interface parameters.

## The two-layer design in one line

`ros2socketcan.cpp` moves raw CAN frames on/off the bus 1:1; `canbridge.py`
contains all the Hunter-specific meaning (frame IDs, packing, scaling). A
replacement driver may keep this split or merge both into one process — the
wire protocol on `can1` is identical either way.

## Key facts (see CAN_INTERFACE.md for the full detail)

- Bus: Peak PCAN-USB as `can1`, **500 kbps**, standard 11-bit IDs, classic CAN.
- Command `0x111` @ **50 Hz**: `[speed_hi speed_lo 0 0 0 0 turn_hi turn_lo]`,
  big-endian signed int16 fields.
- Startup handshake (once): `0x441 00` clear faults → `0x421 01` enter CAN
  control → `0x131 00` brakes off.
- Feedback: `0x211` (mode in `data[1]`), `0x221` (speed `data[0:2]`, steer
  `data[6:8]`).
- Scaling is **empirically calibrated, not datasheet units**:
  `SPEED_SCALE = 1.804`, `STEER_SCALE = 0.657`, `WHEEL_BASE = 0.65 m`. Same
  constant is used for both decode and encode so commands execute 1:1.
- Safety: speed hard-capped at ±0.6 m/s (±333 CAN units); steering ±640 units.

## Bring-up (once per boot, needs root)

```bash
sudo ip link set can1 up type can bitrate 500000
ip -br link show can1        # expect UP
candump can1                 # expect 0x211, 0x221, 0x241, 0x251-3, 0x261, 0x311
```

## Build (ROS 2 Humble)

```bash
# from a colcon workspace with this package under src/
colcon build --packages-select canbridge --symlink-install
source install/setup.bash
ros2 launch canbridge can_bridge.launch.py can_interface:=can1
```

Package dependencies (see `package.xml`): `rclcpp`, `rclpy`, `can_msgs`,
`autoware_control_msgs`, `autoware_vehicle_msgs`.
