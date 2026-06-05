#!/usr/bin/env bash
# Wait for both v3 trainings to exit, then re-export the resulting best_model.zip
# files to .pth and repoint the launch defaults at lab_models_v3.
set -uo pipefail

FWD_PID=1554520
REV_PID=1556368
REPO=/home/ben/Ben/Thesis/Electrans_project
E2E_PY=/home/ben/Ben/Thesis/e2e_rl/venv/bin/python3
REEXPORT=$REPO/src/electrans_rl_bridge/scripts/re_export_td3.py
V3_FWD_ZIP=$REPO/lab_models_v3/models/forward/lidar_24/multiplicative/best_model.zip
V3_REV_ZIP=$REPO/lab_models_v3/models/reverse/lidar_24/guided/best_model.zip
V3_FWD_PTH=$REPO/lab_models_v3/models/forward/lidar_24/multiplicative/best_model.policy.pth
V3_REV_PTH=$REPO/lab_models_v3/models/reverse/lidar_24/guided/best_model.policy.pth
V2_FWD_PTH=$REPO/lab_models_v2/models/forward/lidar_24/multiplicative/best_model.policy.pth
V2_REV_PTH=$REPO/lab_models_v2/models/reverse/lidar_24/guided/best_model.policy.pth

LOG=$REPO/lab_models_v3/.train_logs/post_train.log
exec > "$LOG" 2>&1

echo "[post_train] waiting for PIDs $FWD_PID $REV_PID to exit"
while kill -0 "$FWD_PID" 2>/dev/null || kill -0 "$REV_PID" 2>/dev/null; do
  sleep 60
done
echo "[post_train] both training PIDs exited at $(date)"

echo "[post_train] forward final eval lines:"
grep -E "Eval num_timesteps|New best" "$REPO/lab_models_v3/.train_logs/forward.log" | tail -8
echo "[post_train] reverse final eval lines:"
grep -E "Eval num_timesteps|New best" "$REPO/lab_models_v3/.train_logs/reverse.log" | tail -8

echo
echo "[post_train] re-exporting forward"
"$E2E_PY" "$REEXPORT" "$V3_FWD_ZIP" "$V3_FWD_PTH" 2>&1 | tail -4

echo "[post_train] re-exporting reverse"
"$E2E_PY" "$REEXPORT" --reverse "$V3_REV_ZIP" "$V3_REV_PTH" 2>&1 | tail -4

echo
echo "[post_train] repointing launches: v2 -> v3"
LAUNCHES=(
  "$REPO/src/launcher/autoware_launch/autoware_launch/launch/planning_simulator.launch.xml"
  "$REPO/src/launcher/autoware_launch/autoware_launch/launch/autoware.launch.xml"
  "$REPO/src/electrans_rl_bridge/launch/electrans_rl_bridge.launch.xml"
  "$REPO/src/launcher/autoware_launch/autoware_launch/launch/components/tier4_control_component.launch.xml"
)
for f in "${LAUNCHES[@]}"; do
  sed -i \
    -e "s|$V2_FWD_PTH|$V3_FWD_PTH|g" \
    -e "s|$V2_REV_PTH|$V3_REV_PTH|g" \
    "$f"
done

echo "[post_train] rebuilding affected ROS packages"
cd "$REPO"
colcon build --symlink-install --packages-select electrans_rl_bridge autoware_launch 2>&1 | tail -4

echo
echo "[post_train] done at $(date)"
