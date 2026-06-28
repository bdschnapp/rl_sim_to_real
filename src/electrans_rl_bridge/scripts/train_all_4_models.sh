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

# Pin BLAS/OpenMP to 1 thread per process. With SubprocVecEnv (N_ENVS workers)
# each numpy/torch import otherwise grabs a thread pool sized to all 24 cores,
# so N_ENVS x 24 threads thrash the box and env stepping crawls (this was the
# "training got much slower" regression — measured 24 threads/worker, ~23 of 24
# cores lost to contention). Per-step env work is single-threaded anyway; the
# parallelism is ACROSS the N_ENVS processes. TD3 gradient updates run on GPU
# (device=auto -> cuda), so capping CPU threads does not slow learning. Must be
# exported BEFORE python starts so it takes effect before the first numpy import.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# Unbuffered stdout so SB3 progress tables flush to the per-model logs live
# (otherwise block-buffering hides FPS/reward for many minutes when not a TTY).
export PYTHONUNBUFFERED=1

# Subset selector: space-separated list of which of the 4 to (re)train. Lets you
# retrain only the converged cases without clobbering the rest, e.g.
#   MODELS="forward_truck reverse_truck forward_trailer" MODE=stop_signal ...
MODELS="${MODELS:-forward_truck reverse_truck forward_trailer reverse_trailer}"
MODE="${MODE:-fixed}"
N_ENVS="${N_ENVS:-8}"
STEPS="${STEPS:-200000}"
SPEED_MIN="${SPEED_MIN:-0.5}"
SPEED_MAX="${SPEED_MAX:-0.8}"
# At deploy speed (0.5-0.8 m/s) episodes truncate at the full 1000-step cap
# (never reach the far goal in the 150x90 m world), so each eval pass costs
# n_eval_episodes x ~1000 steps. Bump eval-freq to keep eval overhead modest.
EVAL_FREQ="${EVAL_FREQ:-20000}"
NORM_EVAL_FREQ="${NORM_EVAL_FREQ:-50000}"
case "$MODE" in
    fixed)          ACTION_FLAG="" ;;
    stop_signal)    ACTION_FLAG="--stop-signal" ;;
    variable_speed) ACTION_FLAG="--variable-speed" ;;
    *) echo "MODE must be fixed | stop_signal | variable_speed (got '$MODE')"; exit 2 ;;
esac
# Easy-case deploy-speed training: mild paths only, lab-scale world (set in the
# trainer config override), per-episode speed in [SPEED_MIN, SPEED_MAX].
COMMON="--n-envs $N_ENVS --lidar-beams 24 --reward multiplicative $ACTION_FLAG --mild-paths --speed-min $SPEED_MIN --speed-max $SPEED_MAX --eval-freq $EVAL_FREQ --normalized-eval-freq $NORM_EVAL_FREQ --e2e-rl-path $E2E"

# Output roots (overridable so a stop_signal run can land in fresh dirs instead
# of clobbering the validated fixed-speed checkpoints in the defaults).
TRACTOR_OUT="${TRACTOR_OUT:-$REPO/../previous_models/lab_models_tractor_only}"
TRAILER_OUT="${TRAILER_OUT:-$REPO/../previous_models/lab_models_v21}"

want() { [[ " $MODELS " == *" $1 "* ]]; }

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$SUMMARY"; }
run() {
    local name="$1"; shift
    local logf="$LOGDIR/$name.log"
    log "START $name"
    if "$@" > "$logf" 2>&1; then log "OK    $name"; else log "FAIL  $name (exit $?, see $logf)"; fi
}

log "Training: MODE=$MODE  N_ENVS=$N_ENVS  STEPS=$STEPS  MODELS='$MODELS'  REPO=$REPO  logs=$LOGDIR"
log "Outputs: tractor=$TRACTOR_OUT  trailer=$TRAILER_OUT"
cd "$SCRIPT_DIR"

want forward_truck   && run forward_truck   "$PY" train_lab_tractor_only.py --scenario forward $COMMON --timesteps "$STEPS" --out-dir "$TRACTOR_OUT"
want reverse_truck   && run reverse_truck   "$PY" train_lab_tractor_only.py --scenario reverse $COMMON --timesteps "$STEPS" --out-dir "$TRACTOR_OUT"
want forward_trailer && run forward_trailer "$PY" train_lab_model.py        --scenario forward $COMMON --timesteps "$STEPS" --out-dir "$TRAILER_OUT"
want reverse_trailer && run reverse_trailer "$PY" train_lab_model.py        --scenario reverse $COMMON --timesteps "$STEPS" --out-dir "$TRAILER_OUT"

if [ "$MODE" = "variable_speed" ]; then
    BASE_ZIP="$TRAILER_OUT/models/reverse/lidar_24/multiplicative/best_model.zip"
    if [ -f "$BASE_ZIP" ]; then
        run reverse_trailer_curriculum "$PY" train_lab_recovery_curriculum.py \
            --init-from-zip "$BASE_ZIP" --n-envs "$N_ENVS" --lidar-beams 24 --reward multiplicative \
            --e2e-rl-path "$E2E" --out-dir "$REPO/../previous_models/lab_models_v21_curriculum"
    else
        log "SKIP  reverse_trailer_curriculum (base zip missing: $BASE_ZIP)"
    fi
else
    log "SKIP  reverse_trailer_curriculum (MODE=$MODE; curriculum is variable-speed-only for now)"
fi

log "ALL DONE"
echo "==== summary ===="; cat "$SUMMARY"
