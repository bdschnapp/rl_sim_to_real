#!/bin/bash
# Sequential training of the 4 lab RL models at the corrected 2.8 m trailer
# length, all lidar_24 + multiplicative reward. Runs sequentially; a failure is
# logged but does NOT stop the rest, so partial progress survives.
#
#   1. forward truck          (tractor-only) -> lab_models_tractor_only/models/forward
#   2. reverse truck          (tractor-only) -> lab_models_tractor_only/models/reverse
#   3. forward truck+trailer  (trailer)      -> lab_models_v21/models/forward
#   4. reverse truck+trailer  (trailer)      -> lab_models_v21/models/reverse
#
# Knobs (env vars):
#   MODE     = stop_signal (default) | variable_speed
#              stop_signal -> --stop-signal: constant speed + 2-D [steer, stop],
#              a stop is penalized in these no-obstacle envs.
#   N_ENVS   = parallel envs (default 8).
#   STEPS    = timesteps per model (default 200000; fwd converges ~30k, rev ~60k).
#
# Training is DELAY-FREE by design (no actuator-lag patch) — sim-to-real latency
# is compensated at deploy by the Smith Predictor, never learned here.
#
# The reverse-trailer recovery CURRICULUM (train_lab_recovery_curriculum.py) is
# variable-speed-wired, so it only runs in MODE=variable_speed. A stop-signal
# recovery curriculum is a TODO once base stop-signal models look good.
#
# Outputs (.zip + best_model.zip) still need re_export_td3.py -> .pth for the
# bridge (tractor-only also needs the 29-dim deploy plumbing). Test in pygame
# with eval_lab_model.py.
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

MODE="${MODE:-stop_signal}"
N_ENVS="${N_ENVS:-8}"
STEPS="${STEPS:-200000}"
case "$MODE" in
    stop_signal)    ACTION_FLAG="--stop-signal" ;;
    variable_speed) ACTION_FLAG="--variable-speed" ;;
    *) echo "MODE must be stop_signal or variable_speed (got '$MODE')"; exit 2 ;;
esac
COMMON="--n-envs $N_ENVS --lidar-beams 24 --reward multiplicative $ACTION_FLAG --e2e-rl-path $E2E"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$SUMMARY"; }
run() {
    local name="$1"; shift
    local logf="$LOGDIR/$name.log"
    log "START $name"
    if "$@" > "$logf" 2>&1; then log "OK    $name"; else log "FAIL  $name (exit $?, see $logf)"; fi
}

log "Training: MODE=$MODE  N_ENVS=$N_ENVS  STEPS=$STEPS  REPO=$REPO  logs=$LOGDIR"
cd "$SCRIPT_DIR"

run forward_truck   "$PY" train_lab_tractor_only.py --scenario forward $COMMON --timesteps "$STEPS" --out-dir "$REPO/lab_models_tractor_only"
run reverse_truck   "$PY" train_lab_tractor_only.py --scenario reverse $COMMON --timesteps "$STEPS" --out-dir "$REPO/lab_models_tractor_only"
run forward_trailer "$PY" train_lab_model.py        --scenario forward $COMMON --timesteps "$STEPS" --out-dir "$REPO/lab_models_v21"
run reverse_trailer "$PY" train_lab_model.py        --scenario reverse $COMMON --timesteps "$STEPS" --out-dir "$REPO/lab_models_v21"

if [ "$MODE" = "variable_speed" ]; then
    BASE_ZIP="$REPO/lab_models_v21/models/reverse/lidar_24/multiplicative/best_model.zip"
    if [ -f "$BASE_ZIP" ]; then
        run reverse_trailer_curriculum "$PY" train_lab_recovery_curriculum.py \
            --init-from-zip "$BASE_ZIP" --n-envs "$N_ENVS" --lidar-beams 24 --reward multiplicative \
            --e2e-rl-path "$E2E" --out-dir "$REPO/lab_models_v21_curriculum"
    else
        log "SKIP  reverse_trailer_curriculum (base zip missing: $BASE_ZIP)"
    fi
else
    log "SKIP  reverse_trailer_curriculum (MODE=$MODE; curriculum is variable-speed-only for now)"
fi

log "ALL DONE"
echo "==== summary ===="; cat "$SUMMARY"
