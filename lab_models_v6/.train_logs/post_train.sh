#!/usr/bin/env bash
# Forward-only v6 post-train: re-export forward and repoint forward launch
# arg from whatever its current value is (v3 path) to v6.
set -uo pipefail

FWD_PID=2477110
REPO=/home/ben/Ben/Thesis/Electrans_project
E2E_PY=/home/ben/Ben/Thesis/e2e_rl/venv/bin/python3
REEXPORT=$REPO/src/electrans_rl_bridge/scripts/re_export_td3.py
V_FWD_ZIP=$REPO/lab_models_v6/models/forward/lidar_24/multiplicative/best_model.zip
V_FWD_PTH=$REPO/lab_models_v6/models/forward/lidar_24/multiplicative/best_model.policy.pth
PREV_FWD_PTH=$REPO/lab_models_v3/models/forward/lidar_24/multiplicative/best_model.policy.pth

LOG=$REPO/lab_models_v6/.train_logs/post_train.log
exec > "$LOG" 2>&1

echo "[v6 post_train] waiting for PID $FWD_PID to exit"
while kill -0 "$FWD_PID" 2>/dev/null; do
  sleep 60
done
echo "[v6 post_train] forward PID exited at $(date)"

echo "[v6 post_train] forward final eval lines:"
grep -E "Eval num_timesteps|New best" "$REPO/lab_models_v6/.train_logs/forward.log" | tail -10

echo
echo "[v6 post_train] re-exporting forward"
"$E2E_PY" "$REEXPORT" "$V_FWD_ZIP" "$V_FWD_PTH" 2>&1 | tail -4

echo
echo "[v6 post_train] repointing forward launches: v3 -> v6"
LAUNCHES=(
  "$REPO/src/launcher/autoware_launch/autoware_launch/launch/planning_simulator.launch.xml"
  "$REPO/src/launcher/autoware_launch/autoware_launch/launch/autoware.launch.xml"
  "$REPO/src/electrans_rl_bridge/launch/electrans_rl_bridge.launch.xml"
  "$REPO/src/launcher/autoware_launch/autoware_launch/launch/components/tier4_control_component.launch.xml"
)
for f in "${LAUNCHES[@]}"; do
  sed -i "s|$PREV_FWD_PTH|$V_FWD_PTH|g" "$f"
done

echo "[v6 post_train] bumping eval_lab_model default model dir to v6"
sed -i 's|lab_models_v5|lab_models_v6|g; s|lab_models_v4|lab_models_v6|g; s|lab_models_v3|lab_models_v6|g' "$REPO/src/electrans_rl_bridge/scripts/eval_lab_model.py"

echo "[v6 post_train] rebuilding affected ROS packages"
cd "$REPO"
colcon build --symlink-install --packages-select electrans_rl_bridge autoware_launch 2>&1 | tail -4

echo
echo "[v6 post_train] done at $(date)"
echo "[v6 post_train] NOTE: reverse launches still point at v3 fixed-speed checkpoint. To test reverse, override action_space:=fixed_speed at launch."
