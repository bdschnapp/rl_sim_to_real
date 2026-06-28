#!/bin/bash
# SINGLE ENTRY POINT for the whole stack — real robot OR planning simulator.
#
# Usage (ROS-style args, in any order; both optional):
#   ./src/launcher/start_robot.sh                       # real robot, no trailer
#   ./src/launcher/start_robot.sh trailer:=true         # real robot + trailer
#   ./src/launcher/start_robot.sh sim:=true             # sim (RViz), trailer on
#   ./src/launcher/start_robot.sh sim:=true trailer:=false
#
#   sim:=false (default) → real robot: bring up CAN, then
#                          electrans_robot_real.launch.xml.
#   sim:=true            → planning_simulator.launch.xml (no CAN, RViz on).
#   trailer:=false (default for real) / trailer:=true
#                          → single switch for ALL trailer subsystems:
#                            lidar hitch-angle estimator + trailer self-filter
#                            (sensing chain) and the trailer mesh visualizer.
#                            Exported as ELECTRANS_TRAILER so the deep sensing
#                            leaf (robosense_Helios.launch.xml) sees it without
#                            threading an arg through 6 include levels.
#
# EVERYTHING ELSE is hard-coded below (behaviour knobs) or comes from each
# launch file's defaults (machine-specific paths: e2e_rl tree, maps, models —
# these legitimately differ between the dev desktop running sim and the Jetson
# running the real robot, so we do NOT override them here).
#
# Designed to survive SSH disconnect when invoked via the nohup+setsid wrapper
# at scripts/dynamics/robot/start_autoware_real.sh.
set -eo pipefail  # NOT -u: ROS setup scripts trip nounset

# ----- parse ROS-style args (name:=value) ------------------------------------
SIM="false"
TRAILER=""   # empty → default per-mode below (real: false, sim: true)
for arg in "$@"; do
    case "$arg" in
        sim:=*)     SIM="${arg#sim:=}" ;;
        trailer:=*) TRAILER="${arg#trailer:=}" ;;
        *) echo "WARN: ignoring unrecognised arg '$arg' (expected sim:=… / trailer:=…)" >&2 ;;
    esac
done
# Per-mode trailer default if not given explicitly.
if [ -z "$TRAILER" ]; then
    if [ "$SIM" = "true" ]; then TRAILER="true"; else TRAILER="false"; fi
fi

# ----- hard-coded behaviour knobs (edit here, one place) ---------------------
# Machine-independent RL bridge knobs. Machine-specific paths (e2e_rl, models,
# map) are intentionally left to each launch file's defaults.
ACTION_SPACE="${ACTION_SPACE:-stop_signal}"   # fixed_speed | variable_speed
CONTROL_RATE_HZ="${CONTROL_RATE_HZ:-10.0}"
DEFAULT_VELOCITY_MPS="${DEFAULT_VELOCITY_MPS:-0.6}"

# Real-robot CAN config.
CAN_IFACE="${CAN_IFACE:-can1}"
CAN_BITRATE="${CAN_BITRATE:-500000}"
SUDO_PASS="${SUDO_PASS:-a}"          # fallback only; bypassed if `ip` has CAP_NET_ADMIN
AUTO_CAN_UP="${AUTO_CAN_UP:-1}"
SKIP_RX_CHECK="${SKIP_RX_CHECK:-0}"
LAUNCH_RVIZ="${LAUNCH_RVIZ:-true}"
LAUNCH_PERCEPTION="${LAUNCH_PERCEPTION:-false}"
# RL=0 launches WITHOUT the control component (RL bridge) — sensing +
# localization + map only. Used to free the Jetson's CPU and test whether NDT
# localization works with headroom (the RL bridge + BEV are a big CPU consumer).
# Combine with LAUNCH_RVIZ=false to also drop rviz (~2 cores) for the leanest
# localization-only stack:  RL=0 LAUNCH_RVIZ=false ./start_robot.sh
RL="${RL:-1}"
if [ "$RL" = "0" ]; then CONTROL=false; else CONTROL=true; fi

# Workspace root: derive from THIS script's location (src/launcher/), so the
# same script works on the dev desktop and the Jetson regardless of where the
# repo lives. Override with WORKSPACE_ROOT if needed.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

# Tractor-trailer RL lab map (2.8 m bidirectional lane), vendored in-repo under
# maps/ so it travels with the workspace (desktop + Jetson). Override MAP_PATH=...
# to point elsewhere.
MAP_PATH="${MAP_PATH:-$WORKSPACE_ROOT/maps/tractor_trailer_rl_lab_map}"

# Workspace root exported so the launch files resolve in-repo assets (deployed
# models under lab_models_ttrl_deploy/, vendored map under maps/) via
# $(env ELECTRANS_REPO ...). This is what makes the model/map defaults portable
# across the dev desktop and the Jetson without hardcoding either machine's path.
export ELECTRANS_REPO="$WORKSPACE_ROOT"

# e2e_rl is a RUNTIME dependency (the bridge reconstructs the TD3 policies
# against e2e_rl env classes and runs their _get_obs every tick — the deployed
# models' metadata points at Environments.*). It lives as a SIBLING of the
# workspace on both machines (~/Ben/e2e_rl on the Jetson, ~/Ben/Thesis/e2e_rl on
# the desktop), so derive it from WORKSPACE_ROOT. Override with E2E_RL_PATH=...
E2E_RL_PATH="${E2E_RL_PATH:-$(dirname "$WORKSPACE_ROOT")/e2e_rl}"
if [ ! -d "$E2E_RL_PATH" ]; then
    echo "WARN: e2e_rl not found at $E2E_RL_PATH — the RL bridge will fail to import" >&2
    echo "      Environments/Models/e2erl_utils. Set E2E_RL_PATH=... to its location." >&2
fi

# Trailer flag exported for the sensing-chain leaf + read by <set_env> in the
# launch files. Belt-and-suspenders: exporting here is 100% reliable even if
# launch-file <set_env> propagation timing ever changed.
export ELECTRANS_TRAILER="$TRAILER"

# ----- environment -----------------------------------------------------------
source /opt/ros/humble/setup.bash
WS_SETUP="$WORKSPACE_ROOT/install/setup.bash"
if [ ! -f "$WS_SETUP" ]; then
    echo "ERROR: $WS_SETUP not found." >&2
    echo "       Did you 'colcon build' in $WORKSPACE_ROOT first?" >&2
    exit 1
fi
source "$WS_SETUP"
echo "✓ Sourced $WS_SETUP"
echo "✓ Mode: sim=$SIM  trailer=$TRAILER  (ELECTRANS_TRAILER=$ELECTRANS_TRAILER)"

# =============================================================================
# SIM path — no CAN, no sensors; planning_simulator brings up RViz.
# =============================================================================
if [ "$SIM" = "true" ]; then
    echo
    echo "→ Launching planning_simulator (map=$MAP_PATH, trailer=$TRAILER)..."
    echo
    exec env SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
        TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1 PYTORCH_NO_TRITON=1 \
        ELECTRANS_TRAILER="$TRAILER" ELECTRANS_REPO="$ELECTRANS_REPO" \
        ros2 launch autoware_launch planning_simulator.launch.xml \
        map_path:="$MAP_PATH" \
        e2e_rl_path:="$E2E_RL_PATH" \
        trailer:="$TRAILER" \
        action_space:="$ACTION_SPACE" \
        control_rate_hz:="$CONTROL_RATE_HZ" \
        default_velocity_mps:="$DEFAULT_VELOCITY_MPS"
fi

# =============================================================================
# REAL path — CAN bring-up + bus sanity gate, then the real launch.
# =============================================================================
# RViz needs an X display. Over an SSH-detached session there's no DISPLAY and
# RViz would crash; auto-disable so the rest of the stack still comes up.
if [ "$LAUNCH_RVIZ" = "true" ] && [ -z "${DISPLAY:-}" ]; then
    echo "ℹ  DISPLAY not set — auto-disabling RViz (export DISPLAY=:0 to force)"
    LAUNCH_RVIZ="false"
fi

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
            sudo ip link set "$CAN_IFACE" up type can bitrate "$CAN_BITRATE"
        else
            echo "  (using SUDO_PASS env / default fallback)"
            echo "$SUDO_PASS" | sudo -S ip link set "$CAN_IFACE" up type can bitrate "$CAN_BITRATE"
        fi
        echo "✓ $CAN_IFACE is up."
    fi

    if [ "$SKIP_RX_CHECK" != "1" ]; then
        # Hunter vehicle broadcasts status frames at ~430 B/s. Anything below
        # 100 B over 2 s ⇒ vehicle off / e-stop / bitrate mismatch.
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

echo
echo "→ Launching electrans_robot_real (map=$MAP_PATH, rviz=$LAUNCH_RVIZ, control/RL=$CONTROL, perception=$LAUNCH_PERCEPTION, trailer=$TRAILER)..."
echo
exec env FASTDDS_BUILTIN_TRANSPORTS=UDPv4 \
    SDL_VIDEODRIVER=dummy \
    SDL_AUDIODRIVER=dummy \
    TORCHDYNAMO_DISABLE=1 \
    TORCH_COMPILE_DISABLE=1 \
    PYTORCH_NO_TRITON=1 \
    ELECTRANS_TRAILER="$TRAILER" ELECTRANS_REPO="$ELECTRANS_REPO" \
    ros2 launch autoware_launch electrans_robot_real.launch.xml \
    map_path:="$MAP_PATH" \
    e2e_rl_path:="$E2E_RL_PATH" \
    rviz:="$LAUNCH_RVIZ" \
    perception:="$LAUNCH_PERCEPTION" \
    control:="$CONTROL" \
    trailer:="$TRAILER" \
    action_space:="$ACTION_SPACE" \
    control_rate_hz:="$CONTROL_RATE_HZ" \
    default_velocity_mps:="$DEFAULT_VELOCITY_MPS"
