#!/bin/bash
# Overnight sequential training of the 4 lab RL models at the corrected 2.8 m
# trailer length. All use the optimal observation: lidar_24 + multiplicative
# reward + variable-speed, n_envs=1 (low footprint to share the GPU with other
# running training). Runs sequentially; a failure is logged but does NOT stop
# the rest, so partial progress survives.
#
#   1. forward truck          (tractor-only) -> lab_models_tractor_only/models/forward
#   2. reverse truck          (tractor-only) -> lab_models_tractor_only/models/reverse
#   3. forward truck+trailer  (trailer)      -> lab_models_v21/models/forward
#   4. reverse truck+trailer  (trailer base) -> lab_models_v21/models/reverse
#   5. reverse truck+trailer curriculum (warm-start from #4, easy->hard recovery
#      scenarios — the one model that needs sequential learning)
#                                            -> lab_models_v21_curriculum/...
#
# Outputs (.zip + best_model.zip per model) still need re_export_td3.py -> .pth
# before the bridge can load them (tractor-only also needs the 29-dim deploy
# plumbing). Test in sim RViz tomorrow.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"
E2E="$(dirname "$REPO")/e2e_rl"
PY="$E2E/venv/bin/python"
TS="$(date +%Y%m%d-%H%M%S)"
LOGDIR="$REPO/.train_logs/overnight_$TS"
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/summary.log"

export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy

STEPS="${STEPS:-200000}"
COMMON="--n-envs 1 --lidar-beams 24 --reward multiplicative --variable-speed --e2e-rl-path $E2E"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$SUMMARY"; }

run() {
    local name="$1"; shift
    local logf="$LOGDIR/$name.log"
    log "START $name"
    if "$@" > "$logf" 2>&1; then
        log "OK    $name"
    else
        log "FAIL  $name (exit $?, see $logf)"
    fi
}

log "Overnight training: REPO=$REPO  E2E=$E2E  STEPS=$STEPS  logs=$LOGDIR"
cd "$SCRIPT_DIR"

run forward_truck   "$PY" train_lab_tractor_only.py --scenario forward $COMMON --timesteps "$STEPS" --out-dir "$REPO/lab_models_tractor_only"
run reverse_truck   "$PY" train_lab_tractor_only.py --scenario reverse $COMMON --timesteps "$STEPS" --out-dir "$REPO/lab_models_tractor_only"
run forward_trailer "$PY" train_lab_model.py        --scenario forward $COMMON --timesteps "$STEPS" --out-dir "$REPO/lab_models_v21"
run reverse_trailer "$PY" train_lab_model.py        --scenario reverse $COMMON --timesteps "$STEPS" --out-dir "$REPO/lab_models_v21"

BASE_ZIP="$REPO/lab_models_v21/models/reverse/lidar_24/multiplicative/best_model.zip"
if [ -f "$BASE_ZIP" ]; then
    run reverse_trailer_curriculum "$PY" train_lab_recovery_curriculum.py \
        --init-from-zip "$BASE_ZIP" --n-envs 1 --lidar-beams 24 --reward multiplicative \
        --e2e-rl-path "$E2E" --out-dir "$REPO/lab_models_v21_curriculum"
else
    log "SKIP  reverse_trailer_curriculum (base zip missing: $BASE_ZIP)"
fi

log "ALL DONE"
echo "==== summary ===="; cat "$SUMMARY"
