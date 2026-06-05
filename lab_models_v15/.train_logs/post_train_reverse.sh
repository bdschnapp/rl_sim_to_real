#!/usr/bin/env bash
# Reverse-only post-train watcher. Re-exports reverse best_model and
# repoints the launch defaults' td3_reverse_model_path. Bridge code may
# need updating separately to handle the variable-speed reverse model
# (see Task #41).
set -uo pipefail

REV_PID=368205
REPO=/home/ben/Ben/Thesis/Electrans_project
E2E_PY=/home/ben/Ben/Thesis/e2e_rl/venv/bin/python3
REEXPORT=$REPO/src/electrans_rl_bridge/scripts/re_export_td3.py
V_REV_ZIP=$REPO/lab_models_v15/models/reverse/lidar_24/multiplicative/best_model.zip
V_REV_PTH=$REPO/lab_models_v15/models/reverse/lidar_24/multiplicative/best_model.policy.pth
PREV_REV_PTH=$REPO/lab_models_v4/models/reverse/lidar_24/guided/best_model.policy.pth

LOG=$REPO/lab_models_v15/.train_logs/post_train_reverse.log
exec > "$LOG" 2>&1

echo "[v15-rev post_train] waiting for PID $REV_PID"
while kill -0 "$REV_PID" 2>/dev/null; do sleep 60; done
echo "[v15-rev post_train] training exited at $(date)"

"$E2E_PY" - <<'PY'
import numpy as np
d = np.load('/home/ben/Ben/Thesis/Electrans_project/lab_models_v15/models/reverse/lidar_24/multiplicative/logs/evaluations.npz')
r = d['results'].mean(axis=1)
ts = d['timesteps']
print(f'n_evals={len(ts)}, best={r.max():.1f} @ ts={ts[r.argmax()]}, final={r[-1]:.1f}')
for t, rv in zip(ts[-12:], r[-12:]):
    print(f'  ts={t:>7}  reward={rv:>8.1f}')
PY

echo "[v15-rev post_train] re-exporting reverse"
"$E2E_PY" "$REEXPORT" --reverse "$V_REV_ZIP" "$V_REV_PTH" 2>&1 | tail -4

echo "[v15-rev post_train] repointing reverse launch path (v4 -> v15-multiplicative)"
LAUNCHES=(
  "$REPO/src/launcher/autoware_launch/autoware_launch/launch/planning_simulator.launch.xml"
  "$REPO/src/launcher/autoware_launch/autoware_launch/launch/autoware.launch.xml"
  "$REPO/src/electrans_rl_bridge/launch/electrans_rl_bridge.launch.xml"
  "$REPO/src/launcher/autoware_launch/autoware_launch/launch/components/tier4_control_component.launch.xml"
)
for f in "${LAUNCHES[@]}"; do
  sed -i "s|$PREV_REV_PTH|$V_REV_PTH|g" "$f"
done

cd "$REPO"
colcon build --symlink-install --packages-select electrans_rl_bridge autoware_launch 2>&1 | tail -4
echo "[v15-rev post_train] done at $(date)"
echo
echo "NOTE: bridge's _patch_variable_speed_envs deliberately skips the"
echo "reverse env so the legacy v4 1-D model could load. The new v15"
echo "reverse is 2-D (variable-speed). To actually USE this model from"
echo "the bridge, update _patch_variable_speed_envs to also patch the"
echo "reverse env class (or detect action shape per checkpoint)."
