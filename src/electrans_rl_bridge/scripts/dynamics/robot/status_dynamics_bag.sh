#!/bin/bash
# Report whether a dynamics bag recording is currently active.
PID_FILE="$HOME/bags/dynamics_active.pid"
LOG_FILE="$HOME/bags/dynamics_active.log"

if [ ! -f "$PID_FILE" ]; then
    echo "No active recording (no PID file)."
    LATEST_BAG=$(find "$HOME/bags" -maxdepth 1 -mindepth 1 -type d -name 'dynamics_*' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2- 2>/dev/null | head -1 || true)
    if [ -n "$LATEST_BAG" ]; then
        echo "Most recent bag: $LATEST_BAG"
        du -sh "$LATEST_BAG"
    fi
    exit 0
fi

BAG_PID=$(cat "$PID_FILE")
if ! kill -0 "$BAG_PID" 2>/dev/null; then
    echo "Stale PID file (PID $BAG_PID not running). Run stop_dynamics_bag.sh to clean up."
    exit 1
fi

LATEST_BAG=$(find "$HOME/bags" -maxdepth 1 -mindepth 1 -type d -name 'dynamics_*' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2- 2>/dev/null | head -1 || true)
echo "✓ Recording active"
echo "  PID:     $BAG_PID"
echo "  Started: $(stat -c '%y' "$PID_FILE")"
if [ -n "$LATEST_BAG" ]; then
    echo "  Bag:     $LATEST_BAG"
    SIZE=$(du -sh "$LATEST_BAG" 2>/dev/null | cut -f1)
    echo "  Size:    $SIZE"
fi
echo
echo "Last 5 lines of log:"
tail -5 "$LOG_FILE" 2>/dev/null | sed 's/^/  /'
