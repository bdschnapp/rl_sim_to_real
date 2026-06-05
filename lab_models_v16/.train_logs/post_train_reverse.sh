#!/usr/bin/env bash
# v16 reverse post-train watcher.
# Waits for PID to exit, re-exports best_model.zip → best_model.policy.pth,
# repoints the launch defaults from v15 reverse to v16 reverse, rebuilds
# the bridge packages.
set -uo pipefail

REV_PID=287305
REPO=/home/ben/Ben/Thesis/Electrans_project
E2E_PY=/home/ben/Ben/Thesis/e2e_rl/venv/bin/python3
REEXPORT=$REPO/src/electrans_rl_bridge/scripts/re_export_td3.py
V_REV_ZIP=$REPO/lab_models_v16/models/reverse/lidar_24/multiplicative/best_model.zip
V_REV_PTH=$REPO/lab_models_v16/models/reverse/lidar_24/multiplicative/best_model.policy.pth
PREV_REV_PTH=$REPO/lab_models_v15/models/reverse/lidar_24/multiplicative/best_model.policy.pth

LOG=$REPO/lab_models_v16/.train_logs/post_train_reverse.log
exec > "$LOG" 2>&1

echo "[v16-rev post_train] waiting for PID $REV_PID"
while kill -0 "$REV_PID" 2>/dev/null; do sleep 60; done
echo "[v16-rev post_train] training exited at $(date)"

"$E2E_PY" - <<'PY'
import numpy as np
d = np.load('/home/ben/Ben/Thesis/Electrans_project/lab_models_v16/models/reverse/lidar_24/multiplicative/logs/evaluations.npz')
r = d['results'].mean(axis=1)
ts = d['timesteps']
print(f'n_evals={len(ts)}, best={r.max():.1f} @ ts={ts[r.argmax()]}, final={r[-1]:.1f}')
for t, rv in zip(ts[-12:], r[-12:]):
    print(f'  ts={t:>7}  reward={rv:>8.1f}')
PY

echo "[v16-rev post_train] re-exporting reverse"
"$E2E_PY" "$REEXPORT" --reverse "$V_REV_ZIP" "$V_REV_PTH" 2>&1 | tail -4

echo "[v16-rev post_train] repointing reverse launch path (v15 -> v16)"
LAUNCHES=(
  "$REPO/src/launcher/autoware_launch/autoware_launch/launch/planning_simulator.launch.xml"
  "$REPO/src/launcher/autoware_launch/autoware_launch/launch/autoware.launch.xml"
  "$REPO/src/electrans_rl_bridge/launch/electrans_rl_bridge.launch.xml"
  "$REPO/src/launcher/autoware_launch/autoware_launch/launch/components/tier4_control_component.launch.xml"
)
for f in "${LAUNCHES[@]}"; do
  sed -i "s|$PREV_REV_PTH|$V_REV_PTH|g" "$f"
done

# Update eval script default too so eval_lab_model.py picks v16 by default
sed -i 's|lab_models_v15|lab_models_v16|g' "$REPO/src/electrans_rl_bridge/scripts/eval_lab_model.py"

cd "$REPO"
colcon build --symlink-install --packages-select electrans_rl_bridge autoware_launch 2>&1 | tail -4
echo "[v16-rev post_train] done at $(date)"
echo
echo "v16 reverse model is now installed. To test:"
echo "  1. Restart the launch"
echo "  2. Click initial pose + goal in RViz, or"
echo "     bash /tmp/reverse_debug/clean_runner.py --label v16_test --duration 25"
