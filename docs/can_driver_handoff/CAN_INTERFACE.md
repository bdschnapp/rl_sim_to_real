# Hunter Chassis CAN Interface — Driver Specification

**Purpose.** This document fully specifies the CAN protocol used to drive and
read back the AgileX Hunter chassis on the Electrans robot, so that a CAN
driver can be (re)written to match the behaviour that is currently in the repo
and deployed on the robot. It is derived from the authoritative source in this
handoff: `canbridge_package/canbridge/canbridge.py` (the protocol/translation
node) and `canbridge_package/src/ros2socketcan.cpp` (the raw SocketCAN bridge).

Everything a replacement driver must reproduce is in sections 3–7. Sections 1–2
give the physical/architecture context, sections 8–10 cover state machine,
timing, and a self-contained reference implementation.

> **Byte convention used throughout.** `data[0]` is the first byte on the wire.
> All multi-byte values are **big-endian** (most-significant byte first) and
> **signed 16-bit two's complement** unless stated otherwise.

---

## 1. Architecture

The vehicle interface is split into two ROS 2 nodes. A replacement CAN driver
can either keep this split or collapse both layers into one process — the wire
protocol is identical either way.

```
  Autoware                     canbridge.py                 ros2socketcan.cpp            Hunter chassis
 (controller)              (protocol translator)          (raw SocketCAN bridge)          (PCAN / can1)
     │  Control msg              │                                │                            │
     ├──────────────────────────►│ pack 0x111 / 0x421 / …        │                            │
     │  /control/command/         │  as can_msgs/Frame            │                            │
     │  control_cmd               ├───────────────────────────────►│ write(2) raw CAN frame   │
     │                            │  /CAN/can1/transmit            ├────────────────────────────►│
     │                            │                                │                            │
     │  VelocityReport            │                                │  read(2) raw CAN frame     │
     │  SteeringReport            │◄───────────────────────────────┤◄────────────────────────────┤
     │  ControlModeReport         │  decode 0x211 / 0x221          │  /CAN/can1/receive          │
     │◄───────────────────────────┤                                │                            │
```

- **`ros2socketcan.cpp`** — a thin, protocol-agnostic bridge. It opens a
  `SOCK_RAW`/`CAN_RAW` socket on the configured interface and does a 1:1
  translation between Linux `struct can_frame` and the `can_msgs/msg/Frame`
  ROS topic. It applies **no scaling and no protocol logic**. It publishes every
  received frame on `/CAN/<iface>/receive` and transmits every frame it receives
  on `/CAN/<iface>/transmit`.
- **`canbridge.py`** — all Hunter-specific logic: the enable/clear handshake,
  packing the 0x111 motion command, and decoding the 0x211/0x221 feedback into
  Autoware vehicle-status messages. **This is the file a new driver must match.**

---

## 2. Physical / bus layer

| Property | Value | Notes |
|---|---|---|
| Adapter | Peak **PCAN-USB** | Enumerates as SocketCAN interface `can1`. |
| Interface name | `can1` | The Jetson's built-in Tegra `can0` is **unused**. Configurable via the `can_interface` param. |
| Bitrate | **500 000 bit/s** (500 kbps) | AgileX Hunter standard. |
| CAN ID format | **Standard 11-bit** | All IDs below fit in 11 bits; `is_extended = 0`, `is_rtr = 0`, `is_error = 0`. |
| Frame format | Classic CAN 2.0 (not CAN-FD) | 8-byte max payload. |

Bring the interface up before any traffic (once per boot, needs root):

```bash
sudo ip link set can1 up type can bitrate 500000
ip -br link show can1        # expect: UP
candump can1                 # expect a stream of 0x211, 0x221, 0x241, 0x251-3, 0x261, 0x311 frames
```

If `can1` does not exist, the PCAN adapter is unplugged. If it shows
`NO-CARRIER`, the bitrate is wrong or the Hunter is powered off / e-stopped.

---

## 3. Command frames (driver → chassis)

### 3.1 Startup handshake (sent once, in order)

Before motion commands are accepted, the chassis must be taken out of fault and
put into CAN-control mode. `canbridge.py` sends these three frames one time on
the first timer tick:

| Order | CAN ID | DLC | Bytes on wire | Meaning |
|---|---|---|---|---|
| 1 | `0x441` | 1 | `00` | Clear faults |
| 2 | `0x421` | 1 | `01` | Enter **CAN control** mode |
| 3 | `0x131` | 1 | `00` | Brakes off |

> **DLC note.** The repo sends these as **DLC = 1** — only `data[0]` reaches the
> wire. The chassis only inspects `data[0]` for these enable/clear commands, so
> a DLC of 1 and a DLC of 8 (with the remaining bytes zero) are equivalent. If
> your CAN stack requires DLC 8, pad with zeros: `01 00 00 00 00 00 00 00`.

> **Important ordering/state caveat.** The mode-enable (`0x421 01`) is sent
> **only once, at startup**. If any e-stop is engaged when the driver starts,
> the chassis is in DISENGAGED, the one-shot enable is rejected, and the chassis
> will **not** enter CAN control even after the e-stop is released. The operator
> fix is: release all e-stops, then restart the driver. A more robust driver may
> re-send `0x421 01` whenever it observes the chassis is not in CAN_CONTROL
> (mode byte ≠ `0x01`; see §4.1). The current repo does not do this.

### 3.2 Motion command — `0x111` (sent continuously at 50 Hz)

This is the only per-tick command. **DLC = 8.** It carries the commanded linear
speed and steering angle as two big-endian signed int16 fields:

| Byte | `data[0]` | `data[1]` | `data[2]` | `data[3]` | `data[4]` | `data[5]` | `data[6]` | `data[7]` |
|---|---|---|---|---|---|---|---|---|
| Field | speed **hi** | speed **lo** | 0 | 0 | 0 | 0 | turn **hi** | turn **lo** |

- **speed** = signed int16, big-endian = `(data[0] << 8) | data[1]`, CAN units.
- **turn**  = signed int16, big-endian = `(data[6] << 8) | data[7]`, CAN units.
- Positive speed = forward; negative = reverse.
- Positive turn = left; negative = right (standard Ackermann tire-angle sign).
- Bytes 2–5 are reserved and sent as zero.

**Command rate.** Published every **20 ms (50 Hz)** by a ROS timer. The chassis
expects a steady stream; if commands stop the chassis will time out and stop.
The driver publishes `0x111` every tick even before the first Autoware command
arrives — in that case the payload is all zeros (speed 0, turn 0 = stopped).

**Encoding from physical units → CAN units** (see §5 for the constants):

```
speed_can = clamp( round( velocity_mps * 1000 / SPEED_SCALE ), -333, +333 )
turn_can  = clamp( round( steer_rad    * 1000 / STEER_SCALE ), -640, +640 )
```

with `SPEED_SCALE = 1.804`, `STEER_SCALE = 0.657`.

- Speed is **first** hard-limited to ±`MAX_SPEED_MPS` = **0.6 m/s** in physical
  units, then converted, then the CAN value is capped at **±333** units.
  (Hardware absolute max is ±1500 units; 333 is the software safety cap that
  corresponds to 0.6 m/s in the map frame.)
- Turn is capped at **±640** CAN units (≈ the physical steering lock of ±637
  units ≈ ±0.42 rad true tire angle). Firmware clamps anything beyond.

> **Sign handling in code.** `canbridge.py` computes the speed magnitude as
> `min(abs(int(velocity*1000/SPEED_SCALE)), 333)` and then re-applies the sign
> for reverse. The result is identical to the clamped formula above.

---

## 4. Feedback frames (chassis → driver)

The chassis broadcasts several status frames. The repo **decodes two of them**;
the rest are observed on the bus but ignored.

### 4.1 System state — `0x211`

| Byte | `data[0]` | `data[1]` | … |
|---|---|---|---|
| Field | — | **chassis mode** | — |

`data[1]` (chassis mode) values:

| Value | Name | Meaning | Mapped Autoware `ControlModeReport` |
|---|---|---|---|
| `0x00` | STANDBY | E-stops released, not yet in CAN control | DISENGAGED |
| `0x01` | CAN_CONTROL | Drive ready (accepts `0x111`) | **AUTONOMOUS** |
| `0x02` | REMOTE | RC controller active | MANUAL |
| `0x03` | DISENGAGED | At least one e-stop engaged | DISENGAGED |

The repo maps `0x01 → AUTONOMOUS`, `0x02 → MANUAL`, and **anything else**
(`0x00`, `0x03`) → `DISENGAGED`. Published on `/vehicle/status/control_mode`.

### 4.2 Motion state — `0x221`

Carries the measured speed and steering angle:

| Byte | `data[0]` | `data[1]` | `data[2]` | `data[3]` | `data[4]` | `data[5]` | `data[6]` | `data[7]` |
|---|---|---|---|---|---|---|---|---|
| Field | speed **hi** | speed **lo** | — | — | — | — | steer **hi** | steer **lo** |

- **speed_raw** = signed int16 BE = `(data[0] << 8) | data[1]`
- **steer_raw** = signed int16 BE = `(data[6] << 8) | data[7]`

**Decoding to physical units:**

```
velocity_mps = speed_raw / 1000 * SPEED_SCALE     # m/s, map frame
steer_rad    = steer_raw / 1000 * STEER_SCALE     # true tire angle, rad
```

These feed `/vehicle/status/velocity_status` and
`/vehicle/status/steering_status` (see §7). The driver also computes a kinematic
yaw rate from them (see §6).

### 4.3 Other observed frames (not decoded by the repo)

`candump` shows `0x241`, `0x251`, `0x252`, `0x253`, `0x261`, `0x311` in addition
to `0x211`/`0x221`. These are standard Hunter status frames (motor RPM, driver
state, BMS, etc.). The current driver does **not** read them; a new driver only
needs them if you want to expose extra diagnostics. They are listed here so you
recognise them on the bus and do not mistake them for the motion feedback.

---

## 5. Scaling constants & calibration

These are **not** the datasheet units — they were empirically calibrated on the
robot (2026-07-01) and both feedback decode and command encode use the same
constant so commands execute 1:1. **A matching driver must use these exact
values.**

| Constant | Value | Applies to | Meaning / derivation |
|---|---|---|---|
| `SPEED_SCALE` | **1.804** | speed field (`0x111`, `0x221`) | The CAN speed field is *not* mm/s. Calibrated against tape-measured 16 ft straight drives in the **map frame** (NDT distance ÷ raw wheel distance = 4.98 m / 2.76 m). `m/s = raw/1000 × 1.804`. |
| `STEER_SCALE` | **0.657** | steer field (`0x111`, `0x221`) | The CAN steer field is *not* milli-rad of tire angle. Calibrated with a constant-steering full circle: reported 0.637 "rad" drove a 1.46 m radius ⇒ true tire angle `atan(0.65/1.46) = 0.419 rad`, so `true_rad = raw/1000 × 0.657`. |
| `WHEEL_BASE_M` | **0.65** | yaw-rate derivation | Ackermann wheelbase; must match Autoware `vehicle_info.wheel_base`. |
| `MAX_SPEED_MPS` | **0.6** | command clamp | Hard software speed cap (both directions), in map-frame m/s. |
| speed CAN cap | **±333** | `0x111` speed field | `0.6 / 1.804 × 1000 ≈ 333`. Hardware absolute max is ±1500. |
| turn CAN cap | **±640** | `0x111` turn field | ≈ physical steering lock (±637 raw ≈ ±0.42 rad). |
| max tire angle | **0.436 rad** | reference | Vehicle-info `max_steer_angle`; matches the physical lock. |

> Physical steering lock ≈ ±637 raw units ≈ ±0.42 rad true tire angle, which is
> close to the vehicle-info `max_steer_angle` of 0.436 rad.

---

## 6. Derived quantity — kinematic yaw rate

From decoded speed and steering, the driver computes a bicycle-model yaw rate
and publishes it as `heading_rate` (and as `angular.z` downstream). Downstream
consumers (lidar deskew, pointcloud-concat motion compensation, gyro-odometry
fallback) rely on it — leaving it 0 makes every turn look like straight-line
motion.

```
heading_rate = velocity_mps * tan(steer_rad) / WHEEL_BASE_M      # rad/s
```

---

## 7. ROS 2 interface contract

A replacement driver must preserve these topic names, types, and directions so
the rest of the Autoware stack keeps working.

### Subscribes (inputs)

| Topic | Type | Meaning |
|---|---|---|
| `/control/command/control_cmd` | `autoware_control_msgs/msg/Control` | Command source. Reads `msg.longitudinal.velocity` (m/s) and `msg.lateral.steering_tire_angle` (rad). |
| `/CAN/<iface>/receive` | `can_msgs/msg/Frame` | Raw frames from the SocketCAN bridge (if you keep the two-node split). |

### Publishes (outputs)

| Topic | Type | Fields populated |
|---|---|---|
| `/CAN/<iface>/transmit` | `can_msgs/msg/Frame` | Frames to send to the chassis (`0x441`, `0x421`, `0x131`, `0x111`). |
| `/vehicle/status/velocity_status` | `autoware_vehicle_msgs/msg/VelocityReport` | `header.frame_id = "base_link"`, `longitudinal_velocity` (m/s), `heading_rate` (rad/s). |
| `/vehicle/status/steering_status` | `autoware_vehicle_msgs/msg/SteeringReport` | `steering_tire_angle` (rad). |
| `/vehicle/status/control_mode` | `autoware_vehicle_msgs/msg/ControlModeReport` | `mode` (AUTONOMOUS / MANUAL / DISENGAGED). |

`<iface>` defaults to `can1`; it is a node parameter (`can_interface`).

`can_msgs/msg/Frame` fields used: `id` (uint32), `dlc` (uint8), `data` (uint8[8]),
`is_extended`/`is_rtr`/`is_error` (all 0 here).

---

## 8. Mode / safety state machine

```
                 release e-stops
   DISENGAGED ──────────────────────►  STANDBY
   (0x03)                               (0x00)
      ▲                                    │  driver sends 0x421 01 (once at startup)
      │ e-stop engaged                     ▼
      │                                CAN_CONTROL ──── accepts 0x111 motion cmds
      │                                 (0x01)
      │                                    │  operator flips RC to manual
      └──────────── REMOTE ◄───────────────┘
                    (0x02)
```

Read the current mode from `0x211` `data[1]`. Only in **CAN_CONTROL (0x01)**
does the chassis act on `0x111`. See §3.1 for the one-shot-enable caveat.

---

## 9. Timing summary

| Action | Rate / timing |
|---|---|
| Node startup delay | 5 s sleep before the command timer starts (lets the SocketCAN bridge and chassis feedback settle). |
| Startup handshake (`0x441`, `0x421`, `0x131`) | Once, on the first timer tick. |
| Motion command `0x111` | Every **20 ms (50 Hz)**, continuously. |
| Feedback decode (`0x211`, `0x221`) | On receipt (event-driven), whatever rate the chassis broadcasts. |

---

## 10. Quick-reference CAN ID table

| ID | Dir | DLC | Purpose | Payload |
|---|---|---|---|---|
| `0x441` | TX | 1 | Clear faults | `00` |
| `0x421` | TX | 1 | Enter CAN control | `01` |
| `0x131` | TX | 1 | Brakes off | `00` |
| `0x111` | TX | 8 | Motion command | `[spd_hi spd_lo 00 00 00 00 turn_hi turn_lo]` |
| `0x211` | RX | 8 | System state | `data[1]` = chassis mode |
| `0x221` | RX | 8 | Motion state | `[spd_hi spd_lo _ _ _ _ steer_hi steer_lo]` |
| `0x241`,`0x251`-`0x253`,`0x261`,`0x311` | RX | 8 | Other status (RPM/BMS/etc.) | Not decoded |

---

## 11. Reference implementation (pure SocketCAN, no ROS)

The following self-contained Python is behaviourally equivalent to the repo's
motion path, using `python-can` directly. It exists to remove any ambiguity in
byte packing/scaling; it is **not** what runs on the robot (the robot runs the
two-node ROS split above), but any driver that reproduces this wire behaviour
matches the repo.

```python
import can, struct, time, math

SPEED_SCALE = 1.804
STEER_SCALE = 0.657
WHEEL_BASE  = 0.65
MAX_SPEED_MPS = 0.6
SPEED_CAN_CAP = 333
TURN_CAN_CAP  = 640

bus = can.Bus(channel="can1", interface="socketcan", bitrate=500000)

def send(arb_id, data, dlc=None):
    b = bytes(data)
    bus.send(can.Message(arbitration_id=arb_id, data=b,
                         dlc=dlc if dlc is not None else len(b),
                         is_extended_id=False))

# ---- startup handshake (once) ----
send(0x441, [0x00], dlc=1)   # clear faults
send(0x421, [0x01], dlc=1)   # enter CAN control
send(0x131, [0x00], dlc=1)   # brakes off

def pack_motion(velocity_mps, steer_rad):
    v = max(-MAX_SPEED_MPS, min(MAX_SPEED_MPS, velocity_mps))
    spd = int(v * 1000 / SPEED_SCALE)
    spd = max(-SPEED_CAN_CAP, min(SPEED_CAN_CAP, spd))
    trn = int(steer_rad * 1000 / STEER_SCALE)
    trn = max(-TURN_CAN_CAP, min(TURN_CAN_CAP, trn))
    # big-endian signed int16 for each field; bytes 2..5 = 0
    return struct.pack(">h", spd) + b"\x00\x00\x00\x00" + struct.pack(">h", trn)

def decode_motion(data):  # from a 0x221 frame
    spd_raw, steer_raw = struct.unpack(">h", data[0:2])[0], struct.unpack(">h", data[6:8])[0]
    v = spd_raw / 1000 * SPEED_SCALE          # m/s
    d = steer_raw / 1000 * STEER_SCALE        # rad
    yaw_rate = v * math.tan(d) / WHEEL_BASE   # rad/s
    return v, d, yaw_rate

# ---- 50 Hz command loop ----
velocity_cmd, steer_cmd = 0.0, 0.0   # updated from your controller
while True:
    send(0x111, pack_motion(velocity_cmd, steer_cmd), dlc=8)
    # (in parallel, read frames; on 0x211 read data[1] for mode,
    #  on 0x221 call decode_motion)
    time.sleep(0.02)
```

---

*Derived from `canbridge_package/canbridge/canbridge.py` and
`canbridge_package/src/ros2socketcan.cpp`. Calibration constants
(`SPEED_SCALE`, `STEER_SCALE`) were tuned on the physical robot 2026-07-01; if
you recalibrate, update both the feedback-decode and command-encode paths with
the same value so commands execute 1:1.*
