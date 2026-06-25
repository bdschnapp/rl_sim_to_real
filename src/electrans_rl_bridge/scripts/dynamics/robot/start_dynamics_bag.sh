#!/bin/bash
# Start a detached `ros2 bag record` of the dynamics-relevant topics.
#
# Designed to survive SSH disconnect: uses `nohup setsid` so the process
# leaves its own session/process-group and ignores SIGHUP. The PID is
# saved to ~/bags/dynamics_active.pid so the companion stop script can
# find it after reconnection.
#
# IMPORTANT: `ros2 bag record` finalises its SQLite database and writes
# metadata.yaml ONLY on a clean SIGINT. Use stop_dynamics_bag.sh — do
# NOT `kill -9`, the bag will be incomplete.
set -eo pipefail  # NOT -u: ROS setup scripts trip nounset

# ---- environment ----------------------------------------------------------
source /opt/ros/humble/setup.bash
WS_SETUP="$HOME/Ben/rl_sim_to_real/install/setup.bash"
if [ -f "$WS_SETUP" ]; then
    source "$WS_SETUP"
fi

# ---- paths ----------------------------------------------------------------
mkdir -p "$HOME/bags"
TS=$(date +%Y%m%d-%H%M%S)
BAG_DIR="$HOME/bags/dynamics_${TS}"
PID_FILE="$HOME/bags/dynamics_active.pid"
LOG_FILE="$HOME/bags/dynamics_active.log"

# ---- refuse to start a duplicate -----------------------------------------
if [ -f "$PID_FILE" ]; then
    OLDPID=$(cat "$PID_FILE")
    if kill -0 "$OLDPID" 2>/dev/null; then
        echo "ERROR: a bag recording is already running (PID $OLDPID)."
        echo "       Stop it first with stop_dynamics_bag.sh"
        exit 1
    else
        echo "Stale PID file ($OLDPID not running); removing."
        rm -f "$PID_FILE"
    fi
fi

# ---- topics ---------------------------------------------------------------
TOPICS=(
    /control/command/control_cmd
    /control/command/gear_cmd
    /localization/kinematic_state
    /vehicle/status/steering_status
    /vehicle/status/velocity_status
    /vehicle/trailer_state
    /sensing/imu/imu_data
)

# ---- launch detached ------------------------------------------------------
echo "Starting recording → $BAG_DIR"
echo "Topics:"
printf '  %s\n' "${TOPICS[@]}"

# nohup    : ignore SIGHUP when SSH closes
# setsid   : new session — process becomes session leader, PID == PGID
# < /dev/null : detach stdin so the process can't be tied to the SSH tty
# &        : background it from this shell
nohup setsid ros2 bag record -o "$BAG_DIR" "${TOPICS[@]}" \
    > "$LOG_FILE" 2>&1 < /dev/null &
BAG_PID=$!
echo "$BAG_PID" > "$PID_FILE"

# ---- verify it's alive after a couple of seconds --------------------------
sleep 3
if ! kill -0 "$BAG_PID" 2>/dev/null; then
    echo "ERROR: process died within 3s. Tail of log:"
    tail -30 "$LOG_FILE" >&2
    rm -f "$PID_FILE"
    exit 1
fi

# Sanity: count topics it's subscribed to (rosbag2 prints "Listening for...")
echo
echo "✓ Recording active."
echo "  PID:     $BAG_PID  (saved to $PID_FILE)"
echo "  Bag:     $BAG_DIR"
echo "  Log:     $LOG_FILE"
echo
echo "To verify topics are being captured:"
echo "  tail -f $LOG_FILE"
echo "To check size while driving (must be over wifi):"
echo "  ssh $USER@$(hostname) 'ls -la $BAG_DIR/'"
echo "When done, stop with:"
echo "  $HOME/Ben/rl_sim_to_real/src/electrans_rl_bridge/scripts/dynamics/robot/stop_dynamics_bag.sh"
echo
echo "Safe to disconnect SSH now."
