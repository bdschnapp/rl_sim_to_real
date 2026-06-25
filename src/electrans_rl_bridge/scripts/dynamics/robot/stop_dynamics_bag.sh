#!/bin/bash
# Gracefully stop the `ros2 bag record` started by start_dynamics_bag.sh.
#
# Sends SIGINT to the recorder process group, waits up to 15 s for the
# bag to be finalised, then verifies the metadata.yaml is present.
set -eo pipefail  # NOT -u: ROS shells trip nounset

PID_FILE="$HOME/bags/dynamics_active.pid"
LOG_FILE="$HOME/bags/dynamics_active.log"

if [ ! -f "$PID_FILE" ]; then
    echo "No active recording PID file at $PID_FILE."
    echo "Checking for orphaned ros2 bag record processes..."
    ORPHANS=$(pgrep -af "ros2 bag record" || true)
    if [ -n "$ORPHANS" ]; then
        echo "Found:"
        echo "$ORPHANS"
        echo
        echo "If you want to stop an orphan, run:"
        echo "  kill -INT <PID>"
    else
        echo "Nothing to stop."
    fi
    exit 0
fi

BAG_PID=$(cat "$PID_FILE")

if ! kill -0 "$BAG_PID" 2>/dev/null; then
    echo "PID $BAG_PID is no longer running."
    rm -f "$PID_FILE"
    LATEST_BAG=$(find "$HOME/bags" -maxdepth 1 -mindepth 1 -type d -name 'dynamics_*' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2- || true)
    if [ -n "$LATEST_BAG" ]; then
        echo "Most recent bag dir: $LATEST_BAG"
        ls -la "$LATEST_BAG"
        if [ ! -f "$LATEST_BAG/metadata.yaml" ]; then
            echo "WARNING: no metadata.yaml — bag may need 'ros2 bag reindex'"
        fi
    fi
    exit 0
fi

# Sending SIGINT to the entire process group (PGID == PID since we used
# setsid in the start script). Some rosbag2 builds spawn child threads /
# processes; signalling the group is the safe move.
echo "Sending SIGINT to PID $BAG_PID (process group -$BAG_PID)..."
kill -INT -- "-$BAG_PID" 2>/dev/null || kill -INT "$BAG_PID"

# Wait up to 15 s for clean exit.
for i in $(seq 1 15); do
    if ! kill -0 "$BAG_PID" 2>/dev/null; then
        echo "✓ Process exited cleanly after ${i}s."
        rm -f "$PID_FILE"

        LATEST_BAG=$(find "$HOME/bags" -maxdepth 1 -mindepth 1 -type d -name 'dynamics_*' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2- || true)
        if [ -n "$LATEST_BAG" ]; then
            echo
            echo "Bag dir: $LATEST_BAG"
            du -sh "$LATEST_BAG"
            ls -la "$LATEST_BAG"
            if [ -f "$LATEST_BAG/metadata.yaml" ]; then
                echo "✓ metadata.yaml present — bag finalised cleanly."
                echo
                echo "Topic message counts:"
                grep -E "name:|message_count:" "$LATEST_BAG/metadata.yaml" \
                    | paste - - | sed 's/^/  /'
            else
                echo "WARNING: no metadata.yaml — try: ros2 bag reindex $LATEST_BAG"
            fi
        fi
        exit 0
    fi
    sleep 1
done

# Didn't exit on SIGINT — escalate.
echo "WARNING: process did not exit within 15s of SIGINT. Sending SIGTERM..."
kill -TERM "$BAG_PID" 2>/dev/null || true
sleep 5
if kill -0 "$BAG_PID" 2>/dev/null; then
    echo "ERROR: SIGTERM ignored. The bag may already be corrupted." >&2
    echo "Force-kill with:  kill -9 $BAG_PID" >&2
    echo "Then try:         ros2 bag reindex \$(ls -dt $HOME/bags/dynamics_* | head -1)" >&2
    exit 1
fi
rm -f "$PID_FILE"
echo "Process terminated via SIGTERM. Run 'ros2 bag reindex' on the bag dir."
