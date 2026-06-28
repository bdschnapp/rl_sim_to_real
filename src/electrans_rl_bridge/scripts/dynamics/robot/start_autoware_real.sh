#!/bin/bash
# SSH-disconnect-proof wrapper around src/launcher/start_robot.sh.
#
# The actual bring-up logic (workspace sourcing, CAN bring-up, ros2 launch
# invocation) all lives in start_robot.sh — this just runs that script
# detached (nohup + setsid) so it survives the SSH session closing,
# saves the PID to a known file, and verifies the process is still alive
# a few seconds after launch.
#
# Mirrors the start_dynamics_bag.sh pattern. Stop with stop_autoware_real.sh.
#
# Any args (e.g. trailer:=true) are forwarded verbatim to start_robot.sh.
set -eo pipefail

# ---- paths ----------------------------------------------------------------
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$HOME/Ben/rl_sim_to_real}"
ENTRY_POINT="$WORKSPACE_ROOT/src/launcher/start_robot.sh"

if [ ! -x "$ENTRY_POINT" ]; then
    echo "ERROR: $ENTRY_POINT missing or not executable." >&2
    exit 1
fi

mkdir -p "$HOME/.electrans_runtime"
PID_FILE="$HOME/.electrans_runtime/autoware_real.pid"
LOG_FILE="$HOME/.electrans_runtime/autoware_real.log"

# ---- refuse to start a duplicate -----------------------------------------
if [ -f "$PID_FILE" ]; then
    OLDPID=$(cat "$PID_FILE")
    if kill -0 "$OLDPID" 2>/dev/null; then
        echo "ERROR: autoware launch already running (PID $OLDPID)."
        echo "       Stop it first with stop_autoware_real.sh"
        exit 1
    else
        echo "Stale PID file ($OLDPID not running); removing."
        rm -f "$PID_FILE"
    fi
fi

# ---- launch detached ------------------------------------------------------
echo "Detaching launch via nohup+setsid → $LOG_FILE"
nohup setsid bash "$ENTRY_POINT" "$@" > "$LOG_FILE" 2>&1 < /dev/null &
LAUNCH_PID=$!
echo "$LAUNCH_PID" > "$PID_FILE"

echo "PID $LAUNCH_PID — waiting 10 s for initial bring-up..."
sleep 10
if ! kill -0 "$LAUNCH_PID" 2>/dev/null; then
    echo "ERROR: launch died within 10 s. Last 40 lines of log:" >&2
    tail -40 "$LOG_FILE" >&2
    rm -f "$PID_FILE"
    exit 1
fi

echo
echo "✓ Autoware launch active."
echo "  PID:  $LAUNCH_PID  ($PID_FILE)"
echo "  Log:  $LOG_FILE"
echo
echo "Topics typically take 20-30 s to start publishing. Check with:"
echo "  source $WORKSPACE_ROOT/install/setup.bash"
echo "  ros2 topic hz /vehicle/status/steering_status"
echo
echo "Stop later with:"
echo "  $HOME/Ben/rl_sim_to_real/src/electrans_rl_bridge/scripts/dynamics/robot/stop_autoware_real.sh"
