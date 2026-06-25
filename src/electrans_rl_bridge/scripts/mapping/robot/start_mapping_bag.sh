#!/bin/bash
# Start a detached `ros2 bag record` of the topics needed to BUILD A NEW LAB MAP
# (pointcloud map + a fresh NDT alignment reference) from a manual drive.
#
# This is SEPARATE from start_dynamics_bag.sh (that one records control/state
# for the NN dynamics model and does NOT capture lidar). For mapping we need the
# raw/undistorted lidar cloud + IMU + TF (extrinsics) + GNSS, recorded while you
# drive the robot slowly around the lab covering all the moved equipment.
#
# Survives SSH disconnect (nohup + setsid). PID saved to ~/bags/mapping_active.pid.
# Stop ONLY with stop_mapping_bag.sh (clean SIGINT) — kill -9 leaves a corrupt bag.
#
# >>> Per repo rule: this file is authored on the dev desktop and scp'd to the
#     robot under ~/Ben/rl_sim_to_real/. Edit it on the desktop, not the robot. <<<
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
BAG_DIR="$HOME/bags/mapping_${TS}"
PID_FILE="$HOME/bags/mapping_active.pid"
LOG_FILE="$HOME/bags/mapping_active.log"

# ---- refuse to start a duplicate -----------------------------------------
if [ -f "$PID_FILE" ]; then
    OLDPID=$(cat "$PID_FILE")
    if kill -0 "$OLDPID" 2>/dev/null; then
        echo "ERROR: a mapping bag is already recording (PID $OLDPID)."
        echo "       Stop it first with stop_mapping_bag.sh"
        exit 1
    else
        echo "Stale PID file ($OLDPID not running); removing."
        rm -f "$PID_FILE"
    fi
fi

# ---- topics ---------------------------------------------------------------
# VERIFY against `ros2 topic list` before the real run — names marked (verify)
# were inferred while the stack was down. A topic that doesn't exist is harmless
# (record just waits for it), but a wrong name means missing data in the bag.
#
# Mapping pipeline needs, at minimum: one dense lidar cloud + /tf_static
# (sensor extrinsics) + IMU. GNSS + wheel odom help global georeferencing and
# motion priors for SLAM/offline NDT map generation.
TOPICS=(
    # --- lidar (primary mapping input) ---
    /sensing/lidar/front/pointcloud            # dense undistorted single-lidar cloud (confirmed ~7 Hz)
    /sensing/lidar/concatenate_data            # concatenated cloud, if the pipeline emits it (verify)
    # --- motion / pose priors ---
    /sensing/imu/imu_data                       # IMU for scan deskew / SLAM
    /vehicle/status/velocity_status             # wheel odometry (motion prior)
    /sensing/gnss/pose_with_covariance          # GNSS pose for global georef (verify exact name)
    /sensing/gnss/nav_sat_fix                   # raw GNSS fix (verify exact name)
    # --- frames (REQUIRED: extrinsics live in tf_static) ---
    /tf
    /tf_static
)

# ---- launch detached ------------------------------------------------------
echo "Starting MAPPING recording → $BAG_DIR"
echo "Topics:"
printf '  %s\n' "${TOPICS[@]}"

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

echo
echo "✓ Mapping recording active."
echo "  PID:  $BAG_PID  (saved to $PID_FILE)"
echo "  Bag:  $BAG_DIR"
echo "  Log:  $LOG_FILE"
echo
echo "Now drive the robot slowly around the lab, covering all areas + the moved"
echo "equipment, returning near the start to help loop closure. Then:"
echo "  stop_mapping_bag.sh"
