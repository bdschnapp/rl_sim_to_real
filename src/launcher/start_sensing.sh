#!/bin/bash
# SENSING-ONLY bring-up for MAP CAPTURE (raw sensor recording).
#
# Brings up ONLY the raw sensors + tf + vehicle interface — NO localization
# (no NDT/lidar-matching against a map), NO planning, NO control/RL. You drive
# the robot manually (physical Hunter remote) and record a bag of raw sensor
# data; the pointcloud map is built OFFLINE from that bag. This is the correct
# flow for (re)creating a map: there is no map yet, so we cannot localize — we
# capture raw lidar + IMU + wheel-odom + tf and register them offline.
#
# Usage (ROS-style args, optional):
#   ./src/launcher/start_sensing.sh                 # sensors, no trailer, CAN up
#   ./src/launcher/start_sensing.sh trailer:=true   # with trailer self-filter on
#   AUTO_CAN_UP=0 ./src/launcher/start_sensing.sh   # skip CAN (no wheel odom)
#
# Then, in a SECOND terminal (sourced), record the bag — see the printed
# `ros2 bag record` command below (confirm topic names with `ros2 topic list`).
set -eo pipefail

# ----- args -----
TRAILER="false"
for arg in "$@"; do
  case "$arg" in
    trailer:=*) TRAILER="${arg#trailer:=}" ;;
    *) echo "WARN: ignoring '$arg' (expected trailer:=true|false)" >&2 ;;
  esac
done

# ----- CAN (for wheel odometry via the vehicle interface) -----
CAN_IFACE="${CAN_IFACE:-can1}"
CAN_BITRATE="${CAN_BITRATE:-500000}"
SUDO_PASS="${SUDO_PASS:-a}"
AUTO_CAN_UP="${AUTO_CAN_UP:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
MAP_PATH="${MAP_PATH:-$HOME/Ben/Thesis/Electrans_project/maps/tractor_trailer_rl_lab_map}"   # only for sensor_kit/vehicle descriptions; map module is OFF

export ELECTRANS_TRAILER="$TRAILER"

source /opt/ros/humble/setup.bash
source "$WORKSPACE_ROOT/install/setup.bash"
echo "✓ Sourced workspace; sensing-only (trailer=$TRAILER)"

if [ "$AUTO_CAN_UP" = "1" ]; then
  if ip link show "$CAN_IFACE" >/dev/null 2>&1; then
    if ! ip -br link show "$CAN_IFACE" | grep -q '\bUP\b'; then
      echo "→ Bringing $CAN_IFACE up at $CAN_BITRATE bps..."
      if sudo -n true 2>/dev/null; then sudo ip link set "$CAN_IFACE" up type can bitrate "$CAN_BITRATE"
      else echo "$SUDO_PASS" | sudo -S ip link set "$CAN_IFACE" up type can bitrate "$CAN_BITRATE"; fi
    fi
    echo "✓ $CAN_IFACE up (wheel odom available)."
  else
    echo "ℹ  $CAN_IFACE missing — continuing without wheel odom (lidar+IMU+tf only)."
  fi
fi

cat <<'REC'
────────────────────────────────────────────────────────────────────────────
 To RECORD the mapping bag, in a SECOND sourced terminal run:

   cd ~/Ben/<workspace> && source install/setup.bash
   ros2 bag record -o ~/map_capture_$(date +%Y%m%d_%H%M%S) \
     /sensing/pointcloud/robosense \
     /tf /tf_static \
     /sensing/imu/imu_data \
     /vehicle/status/velocity_status

 First confirm exact names with: ros2 topic list | grep -E 'point|imu|tf|velocity'
 Drive the FULL drivable space slowly (close the loop, revisit start). Ctrl-C to stop.
────────────────────────────────────────────────────────────────────────────
REC

echo "→ Launching sensing-only (sensors + tf + vehicle interface; NO localization/planning/control)..."
exec env FASTDDS_BUILTIN_TRANSPORTS=UDPv4 ELECTRANS_TRAILER="$TRAILER" \
  ros2 launch autoware_launch electrans_robot_real.launch.xml \
    map_path:="$MAP_PATH" \
    sensing:=true \
    vehicle:=true \
    launch_sensing_driver:=true \
    map:=false \
    localization:=false \
    perception:=false \
    planning:=false \
    control:=false \
    system:=false \
    rviz:=false \
    trailer:="$TRAILER"
