# `rl_sim_to_real` setup notes

Distilled from a long pairing session on the sibling repo
`/home/electrans_robot/Electrans_project` (branch `robot-real-launch`).
That repo runs the full Autoware planning + control stack on the real
AgileX Hunter trailer robot; this repo (`rl_sim_to_real`) keeps Autoware
**only for perception + localization** and uses an external TD3/RL policy
(via `electrans_rl_bridge`) for control. The list below is what was
needed to get the sibling repo driving the robot, with notes on what
still applies here.

## 0. TL;DR — what is *not* needed any more

- **`~/lidar_ws` is no longer a dependency.** Every package that was
  being pulled from `lidar_ws` (canbridge, ros2_socketcan, ros2can_bridge,
  nebula, witmotion_ros, ublox*) is already present under
  `src/sensor_component/` and `src/launcher/`. Source only this repo's
  `install/setup.bash`.
- **Dummy perception publishers are not needed.** The full Autoware
  perception pipeline runs in this repo. The sibling project had to
  fake `/perception/object_recognition/objects`,
  `/perception/occupancy_grid_map/map`, and
  `/perception/obstacle_segmentation/pointcloud` because it launches with
  `perception:=false`; this repo runs real perception, so skip that.
- **No need to engage Autoware autonomous mode for driving.** Control
  comes from `electrans_rl_bridge` directly, not from
  `/api/operation_mode/change_to_autonomous`. The hacks around the
  diagnostic graph aggregator and the `OperationModeState` QoS workaround
  (a manual `ros2 topic pub` republisher) are only relevant if you want
  to flip Autoware control on.

## 1. Patches already imported from the sibling repo

I `diff`'d the critical files. All of these are **already identical** in
both repos — no porting needed:

| File | Why it was changed |
|---|---|
| `src/sensor_component/agilex_hunter_sensors/canbridge/canbridge/canbridge.py` | Publishes `/vehicle/status/control_mode` (required for `operation_mode_transition_manager` to engage), hard 0.5 m/s speed cap, steering scale 1320 (full range), reads `can_interface` param so it can target `can1`. |
| `src/launcher/autoware_launch/vehicle/electrans_robot_vehicle_launch/.../vehicle_interface.launch.xml` | Default `can_interface` is `can1`; forwards param to both `ros2can_bridge` and `canbridge`. |
| `src/launcher/autoware_launch/sensor_kit/electrans_robot_sensor_kit_launch/.../launch/imu.launch.xml` | Tamagawa driver was commented out; Witmotion WT905 driver is now wired in with `/imu → tamagawa/imu_raw` remap. |
| `src/launcher/autoware_launch/sensor_kit/electrans_robot_sensor_kit_launch/.../launch/gnss.launch.xml` | `ublox_gps_node` no longer has `respawn="true"`. With our F9P unit it throws `Could not configure serial baud rate` on startup and would otherwise burn CPU in a respawn loop. |
| `src/launcher/autoware_launch/autoware_launch/config/system/diagnostics/localization.yaml` | `accuracy` and `sensor_fusion_status` links commented out of the `/autoware/localization` aggregate so autonomous mode is not gated on EKF/IMU-fusion health. *(Only relevant if you re-enable Autoware autonomous.)* |
| `src/launcher/autoware_launch/autoware_launch/config/planning/vehicle_overrides/robot/common.param.yaml` | `max_vel: 0.2 m/s`. *(Only relevant if you re-enable Autoware planning.)* |
| `src/universe/autoware_universe/launch/tier4_localization_launch/launch/pose_twist_estimator/pose_twist_estimator.launch.xml` | This repo goes further than the sibling: it has been pruned to drop yabloc, eagleye, AR-tag, lidar-marker, and the ADAPI `automatic_pose_initializer`. Net effect is the same as setting `gnss_enabled:=false` in the sibling — manual initial-pose only, no auto-init loop fighting the user's pose publish. |

## 2. Hardware bring-up (every robot session)

These steps apply regardless of which stack you're running.

1. **CAN interface (sudo, once per boot).** The robot's Hunter chassis
   talks over a Peak PCAN USB adapter that enumerates as `can1`. The
   built-in Jetson Tegra `can0` is unused.
   ```
   sudo ip link set can1 up type can bitrate 500000
   ip -br link show can1                                  # expect: UP
   timeout 2 candump can1 | head                          # expect: frames at 0x211, 0x221, 0x241, 0x251-3, 0x261, 0x311
   ```
   If `can1` doesn't exist at all: the PCAN adapter is unplugged. If it
   shows `NO-CARRIER`, the bitrate is wrong or the Hunter is off.

2. **WT905 IMU.** It enumerates as `/dev/ttyCH341USB0` (CH340 USB-serial)
   with the symlink `/dev/ttyUSB_IMU`. It must be put into a continuous
   output mode at 115200 baud via the Witmotion config GUI / button
   sequence before the ROS driver will get any data. The driver opens
   the port cleanly even when the device is silent, so the symptom of a
   wrong-mode device is "no error in the witmotion log but
   `/sensing/imu/tamagawa/imu_raw` has zero publishers".

3. **u-blox ZED-F9P GNSS.** Enumerates as `/dev/ttyACM0`. The current
   `zed_f9p.yaml` mismatches the device baud, so `ublox_gps_node` dies
   on startup with `Could not configure serial baud rate`. The respawn
   loop is disabled (see §1) — it dies once and stays dead. This is
   harmless for the localization pipeline because GNSS pose isn't being
   used (see the `pose_twist_estimator.launch.xml` pruning).

4. **Robosense Helios lidar.** Driver expects packets on the Jetson
   wired interface `enP8p1s0` (host IP `192.168.1.102`, driver binds to
   `192.168.1.200`). If `ros2 topic hz /sensing/lidar/front/pointcloud`
   produces nothing after launch, check the cable / device IP — symptom
   in the log is `Missed pointcloud output deadline` repeating and NDT
   reporting `No InputSource. Please check the input lidar topic`.

5. **Hunter chassis "mode" lever / e-stops.** The chassis advertises its
   mode in byte 1 of CAN frame 0x211:
   - `0` STANDBY — e-stops released, but chassis isn't in CAN_CONTROL yet
   - `1` CAN_CONTROL — drive ready (only entered after canbridge sends
     `0x421 0x01 ...`, which canbridge does **once at startup**)
   - `2` REMOTE — RC controller active
   - `3` DISENGAGED — at least one e-stop is engaged

   Implication: if you bring the robot up with any e-stop active, the
   chassis will be in `3` when canbridge starts. canbridge's one-shot
   `CanMode` send is rejected and the chassis stays in `0`/`3` forever
   even after the e-stop is released. **Fix: release all e-stops, then
   restart canbridge.** A bash one-liner to repro:
   ```
   pkill -KILL -9 -f "/canbridge/canbridge"
   /home/electrans_robot/Ben/rl_sim_to_real/install/canbridge/lib/canbridge/canbridge \
     --ros-args -r __node:=controller_canbridge -p can_interface:=can1
   ```
   (Same idea as the Electrans_project version — adjust the path if the
   workspace install dir differs.)

## 3. Python / system dependencies

The kept Autoware stack still pulls in some non-standard packages:

- **`osqp`** — required by `autoware_trailer_ltv_mpc_controller`. Since
  control is ripped out here this may not apply, but if anything pulls
  in `qp_solver.py` you'll see
  `RuntimeError: OSQP is required for Trailer LTV MPC solving.`
  Install with `pip3 install --user osqp`.
- `python3-numpy`, `python3-scipy`, `tf_transformations` — already
  declared as exec deps in `electrans_rl_bridge/package.xml`.
- ROS 2 Humble, CUDA 11.6 on Ubuntu 22.04 per the existing README.

External resources `electrans_rl_bridge` expects (override via launch args):
- `e2e_rl_path` — path to the `e2e_rl` source tree on this machine. The
  current default in `electrans_rl_bridge.launch.xml` is
  `/home/ben/Ben/Thesis/e2e_rl` (Ben's dev box). On this Jetson
  (`electrans_robot`) update or pass `e2e_rl_path:=<actual path>`.
- `td3_model_path`, `td3_reverse_model_path` — same comment; the
  defaults point at `lab_models_v15` / `lab_models_v16` under Ben's
  laptop layout. The Jetson has `lab_models_v*` under
  `/home/electrans_robot/Ben/rl_sim_to_real/` already, so the path just
  needs the prefix updated.
- `map_path` — `~/autoware_map/mvsl` on this Jetson (same as sibling repo).

## 4. Build

```
cd /home/electrans_robot/Ben/rl_sim_to_real
# colcon build the whole tree. Use --symlink-install so config/launch
# edits are picked up without rebuilding (sibling repo relies on this).
colcon build --symlink-install
```

If the build is slow on the Jetson and you only edited python/launch/yaml,
re-running the launch is enough because `--symlink-install` makes those
files live.

## 5. Launch sequence (Autoware perception + localization + RL bridge)

```bash
# 0. CAN up (sudo, once per boot)
sudo ip link set can1 up type can bitrate 500000

# 1. Source ONLY this repo (no ~/lidar_ws)
cd /home/electrans_robot/Ben/rl_sim_to_real
source install/setup.bash

# 2. Launch Autoware. Perception is ON here; planning/control args don't
#    matter because those packages aren't in this tree. rviz on the
#    Jetson is heavy, prefer the foxglove_bridge that's already in the
#    launch and connect from a laptop.
FASTDDS_BUILTIN_TRANSPORTS=UDPv4 ros2 launch autoware_launch \
  electrans_robot_real.launch.xml \
  map_path:=$HOME/autoware_map/mvsl rviz:=false

# 3. In a second shell, after the launch settles, start the RL bridge
source install/setup.bash
ros2 launch electrans_rl_bridge electrans_rl_bridge.launch.xml \
  e2e_rl_path:=<absolute path to e2e_rl on this machine> \
  td3_model_path:=<absolute path to forward model .pth> \
  td3_reverse_model_path:=<absolute path to reverse model .pth> \
  map_path:=$HOME/autoware_map/mvsl
```

In Foxglove (`ws://<jetson-ip>:8765`):
1. **3D panel Fixed Frame must be `map`.** With any other Fixed Frame
   the published `/initialpose` arrives at the adaptor with
   `frame_id != map`, the height fitter fails its TF lookup, and NDT's
   align service rejects with `Please publish TF map to base_link`.
   That symptom looks like "the initial pose was ignored" but is
   actually a frame-id bug.
2. Publish initial pose. After NDT aligns you should see
   `EKF Activation succeeded` and `map → base_link` TF live.
3. Hand off to whatever workflow the RL bridge expects (goal pose,
   lane reference, etc.).

If the chassis byte 1 is not `1` (CAN_CONTROL) at this point, restart
canbridge as described in §2.5 above.

## 6. Known issues / gotchas carried over from the sibling repo

- After Autoware reaches a goal and the planner emits a degenerate
  trajectory, `motion_velocity_planner` in the sibling repo crashed
  inside `SplineInterpolationPoints2d::getSplineInterpolatedYaw`
  (planning_container then dies with SIGABRT). It is in the autoware
  universe code (`autoware_motion_velocity_planner`) and is not patched.
  This won't affect this repo as long as the planning packages aren't
  active. If you ever build/enable them: relaunch is currently the only
  workaround.
- The `operation_mode_transition_manager` and its `change_to_autonomous`
  service ARE still present in this tree because the localization
  pipeline references them. Don't call that service from the RL bridge
  workflow — leaving the system in `mode=STOP` is fine. If you ever
  want to flip it on, the sibling needed:
  1. clear `duplicated_node_checker` (kill stragglers from prior launches whose PIDs are lower than the current `ros2 launch` PID),
  2. wait for chassis byte 1 = `1`,
  3. retry the service call until it returns success.
- `ros2 topic echo` and `ros2 topic hz` are flaky on this Jetson under
  load (the CLI client times out long before the topic is actually
  silent). Don't trust a `Terminated` from those tools as proof that
  data isn't flowing — check the publisher count or grep the launch
  log instead.
- The Foxglove 3D panel will NOT render `autoware_planning_msgs/Trajectory`
  or `autoware_planning_msgs/Path` natively. The lanelet route renders
  via `/planning/mission_planning/route_marker` (MarkerArray). For the
  driven path the only options are the MarkerArray debug topics under
  `behavior_path_planner/debug/` etc. (Probably moot here since planning
  is ripped out.)

## 7. Summary of work done in the sibling session

For context if you want to continue this work in a new Claude session:

- Got the AgileX Hunter trailer driving under full Autoware lane-driving
  with the `trailer_ltv_mpc` controller preset by editing the files in
  §1 and installing `osqp` via pip.
- Debugged every step from kernel-level (CAN interface bring-up, PCAN
  USB enumeration, e-stop / chassis mode handshake) up through the
  Autoware operation_mode handshake, NDT initial-pose alignment, and
  per-module RTC auto-mode enables.
- Confirmed the lidar driver, IMU driver, GNSS driver wire-up, and the
  Foxglove 3D-panel `Fixed Frame = map` requirement for initial-pose
  publishing.
- All those fixes have been carried over into this repo's tree.

So for this repo, the only new things to wire up are:
1. The `electrans_rl_bridge` launch paths (`e2e_rl_path`, model paths,
   map_path) for the actual machine being used.
2. Whatever the RL bridge expects on top of localization (`/initialpose`,
   `/planning/mission_planning/goal`, etc.) — check the bridge's
   `lane_reference_node` and `rl_bridge_node` subscriptions.
