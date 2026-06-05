#!/usr/bin/env bash
set -uo pipefail

FWD_PID=1324887
REPO=/home/ben/Ben/Thesis/Electrans_project
E2E_PY=/home/ben/Ben/Thesis/e2e_rl/venv/bin/python3
REEXPORT=$REPO/src/electrans_rl_bridge/scripts/re_export_td3.py
V_FWD_ZIP=$REPO/lab_models_v15/models/forward/lidar_24/multiplicative/best_model.zip
V_FWD_PTH=$REPO/lab_models_v15/models/forward/lidar_24/multiplicative/best_model.policy.pth
PREV_FWD_PTH=$REPO/lab_models_v13/models/forward/lidar_24/multiplicative/best_model.policy.pth

LOG=$REPO/lab_models_v15/.train_logs/post_train.log
exec > "$LOG" 2>&1

echo "[v15 post_train] waiting for PID $FWD_PID"
while kill -0 "$FWD_PID" 2>/dev/null; do sleep 60; done
echo "[v15 post_train] training exited at $(date)"

"$E2E_PY" - <<'PY'
import numpy as np
d = np.load('/home/ben/Ben/Thesis/Electrans_project/lab_models_v15/models/forward/lidar_24/multiplicative/logs/evaluations.npz')
r = d['results'].mean(axis=1)
ts = d['timesteps']
print(f'n_evals={len(ts)}, best={r.max():.1f} @ ts={ts[r.argmax()]}, final={r[-1]:.1f}')
for t, rv in zip(ts[-12:], r[-12:]):
    print(f'  ts={t:>7}  reward={rv:>8.1f}')
PY

echo "[v15 post_train] re-exporting forward"
"$E2E_PY" "$REEXPORT" "$V_FWD_ZIP" "$V_FWD_PTH" 2>&1 | tail -4

LAUNCHES=(
  "$REPO/src/launcher/autoware_launch/autoware_launch/launch/planning_simulator.launch.xml"
  "$REPO/src/launcher/autoware_launch/autoware_launch/launch/autoware.launch.xml"
  "$REPO/src/electrans_rl_bridge/launch/electrans_rl_bridge.launch.xml"
  "$REPO/src/launcher/autoware_launch/autoware_launch/launch/components/tier4_control_component.launch.xml"
)
for f in "${LAUNCHES[@]}"; do
  sed -i "s|$PREV_FWD_PTH|$V_FWD_PTH|g" "$f"
done

sed -i 's|lab_models_v14|lab_models_v15|g; s|lab_models_v13|lab_models_v15|g' "$REPO/src/electrans_rl_bridge/scripts/eval_lab_model.py"

cd "$REPO"
colcon build --symlink-install --packages-select electrans_rl_bridge autoware_launch 2>&1 | tail -4
echo "[v15 post_train] done at $(date)"
