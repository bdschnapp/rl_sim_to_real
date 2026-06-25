#!/bin/bash
# Stop the autoware launch started by start_autoware_real.sh.
# Sends SIGINT to the entire process group so all composable nodes /
# child launches exit cleanly.
set -eo pipefail

PID_FILE="$HOME/.electrans_runtime/autoware_real.pid"
LOG_FILE="$HOME/.electrans_runtime/autoware_real.log"

if [ ! -f "$PID_FILE" ]; then
    echo "No active autoware launch (no PID file)."
    ORPHANS=$(pgrep -af "ros2 launch autoware" || true)
    if [ -n "$ORPHANS" ]; then
        echo "Found orphaned ros2-launch processes:"
        echo "$ORPHANS"
    fi
    exit 0
fi

LAUNCH_PID=$(cat "$PID_FILE")

if ! kill -0 "$LAUNCH_PID" 2>/dev/null; then
    echo "PID $LAUNCH_PID is not running. Cleaning up PID file."
    rm -f "$PID_FILE"
    exit 0
fi

echo "Sending SIGINT to PID $LAUNCH_PID and its process group..."
kill -INT -- "-$LAUNCH_PID" 2>/dev/null || kill -INT "$LAUNCH_PID"

for i in $(seq 1 30); do
    if ! kill -0 "$LAUNCH_PID" 2>/dev/null; then
        echo "✓ Launch exited after ${i}s."
        rm -f "$PID_FILE"
        # Best-effort: kill any leftover ros2 processes (composable nodes
        # sometimes outlive the parent launch).
        STRAGGLERS=$(pgrep -af "ros2 launch autoware_launch electrans_robot_real" || true)
        if [ -n "$STRAGGLERS" ]; then
            echo "Stragglers still running:"
            echo "$STRAGGLERS"
        fi
        exit 0
    fi
    sleep 1
done

echo "WARNING: launch did not exit within 30s. Sending SIGTERM..."
kill -TERM -- "-$LAUNCH_PID" 2>/dev/null || kill -TERM "$LAUNCH_PID"
sleep 5
if kill -0 "$LAUNCH_PID" 2>/dev/null; then
    echo "ERROR: still alive after SIGTERM. Force-kill: kill -9 -$LAUNCH_PID" >&2
    exit 1
fi
rm -f "$PID_FILE"
echo "Launch terminated via SIGTERM."
