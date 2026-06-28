#!/bin/bash
# Single entry point for bringing up the real robot.
#
# What it does (in order):
#   1. Sources ROS Humble + this workspace's install (the ONE consolidated
#      workspace — no more lidar_ws / Electrans_project juggling).
#   2. Brings up the PCAN-USB interface (`can1` by default) at 500 kbps
#      if it isn't already UP. Verifies vehicle status frames are flowing
#      on the bus before launching anything (low RX bytes = Hunter
#      powered off OR e-stop engaged OR bitrate mismatch).
#   3. Exec's the autoware real-robot launch file. The RL bridge,
#      sensors, localization, map, and Foxglove bridge all come up as
#      part of that include graph — no other launch commands needed.
#
# Usage:
#   ./src/launcher/start_robot.sh                 # full bring-up
#   CAN_IFACE=can0 ./src/launcher/start_robot.sh  # override interface
#   AUTO_CAN_UP=0 ./src/launcher/start_robot.sh   # skip CAN bring-up
#   SKIP_RX_CHECK=1 ./src/launcher/start_robot.sh # skip the bus-traffic gate
#
# Designed to be run on the robot itself OR via:
#   ssh electrans_robot@agilex './Ben/rl_sim_to_real/src/launcher/start_robot.sh'
#
# Designed to survive SSH disconnect when invoked via the wrapper at
# scripts/dynamics/robot/start_autoware_real.sh — that wrapper does the
# nohup+setsid plumbing.
set -eo pipefail  # NOT -u: ROS setup scripts trip nounset

# ----- config ----------------------------------------------------------------
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$HOME/Ben/rl_sim_to_real}"
CAN_IFACE="${CAN_IFACE:-can1}"
CAN_BITRATE="${CAN_BITRATE:-500000}"
SUDO_PASS="${SUDO_PASS:-a}"          # fallback only; sudo is bypassed if `ip` has CAP_NET_ADMIN
AUTO_CAN_UP="${AUTO_CAN_UP:-1}"
SKIP_RX_CHECK="${SKIP_RX_CHECK:-0}"
MAP_PATH="${MAP_PATH:-$HOME/Ben/Thesis/Electrans_project/maps/tractor_trailer_rl_lab_map}"
LAUNCH_RVIZ="${LAUNCH_RVIZ:-true}"
LAUNCH_PERCEPTION="${LAUNCH_PERCEPTION:-false}"

# RViz needs an X display. When this script runs over an SSH-detached
# session (start_autoware_real.sh wrapper), there's no DISPLAY and RViz
# would crash on launch with "could not connect to display". Disable RViz
# automatically in that case so the rest of the stack still comes up.
if [ "$LAUNCH_RVIZ" = "true" ] && [ -z "${DISPLAY:-}" ]; then
    echo "ℹ  DISPLAY not set — auto-disabling RViz (set LAUNCH_RVIZ=true && export DISPLAY=:0 to force)"
    LAUNCH_RVIZ="false"
fi

# ----- 1. environment --------------------------------------------------------
source /opt/ros/humble/setup.bash
WS_SETUP="$WORKSPACE_ROOT/install/setup.bash"
if [ ! -f "$WS_SETUP" ]; then
    echo "ERROR: $WS_SETUP not found." >&2
    echo "       Did you 'colcon build' in $WORKSPACE_ROOT first?" >&2
    exit 1
fi
source "$WS_SETUP"
echo "✓ Sourced $WS_SETUP"

# ----- 2. CAN bring-up -------------------------------------------------------
if [ "$AUTO_CAN_UP" = "1" ]; then
    if ! ip link show "$CAN_IFACE" >/dev/null 2>&1; then
        echo "ERROR: CAN interface '$CAN_IFACE' does not exist." >&2
        echo "       PCAN-USB unplugged? Kernel module not loaded?" >&2
        echo "       Check: lsusb | grep -i peak  &&  dmesg | grep -i peak" >&2
        exit 1
    fi

    if ip -br link show "$CAN_IFACE" | grep -q '\bUP\b'; then
        echo "✓ $CAN_IFACE already UP."
    else
        echo "→ Bringing $CAN_IFACE up at $CAN_BITRATE bps..."
        if sudo -n true 2>/dev/null; then
            # Passwordless sudo available — preferred path.
            sudo ip link set "$CAN_IFACE" up type can bitrate "$CAN_BITRATE"
        else
            echo "  (using SUDO_PASS env / default fallback)"
            echo "$SUDO_PASS" | sudo -S ip link set "$CAN_IFACE" up type can bitrate "$CAN_BITRATE"
        fi
        echo "✓ $CAN_IFACE is up."
    fi

    if [ "$SKIP_RX_CHECK" != "1" ]; then
        # Hunter vehicle broadcasts status frames at ~430 B/s. Anything
        # below 100 B over 2 s ⇒ vehicle off / e-stop / bitrate mismatch.
        echo "→ Checking bus traffic on $CAN_IFACE (2 s sample)..."
        B=$(ip -s link show "$CAN_IFACE" | awk '/RX:/{getline; print $2}')
        sleep 2
        A=$(ip -s link show "$CAN_IFACE" | awk '/RX:/{getline; print $2}')
        DELTA=$((A - B))
        echo "  RX delta: ${DELTA} bytes (expect ≥860)"
        if (( DELTA < 100 )); then
            echo "ERROR: very low CAN traffic. Vehicle off, e-stop engaged, or bitrate mismatch?" >&2
            echo "       Set SKIP_RX_CHECK=1 to bypass this gate." >&2
            exit 1
        fi
    fi
fi

