#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Two-phase Docker build for the Electrans Autoware stack.
#
# Phase 1 (base image):   Installs ROS Humble + builds all standard Autoware
#                         packages.  Skipped if the image already exists unless
#                         --rebuild-base is passed.
#
# Phase 2 (overlay image): Builds the custom Electrans packages on top of the
#                          base.  Always runs.
#
# Usage:
#   ./docker_image_build.sh                  # skip Phase 1 if base image exists
#   ./docker_image_build.sh --rebuild-base   # force Phase 1 rebuild
# ─────────────────────────────────────────────────────────────────────────────
set -e

BASE_IMAGE="autoware:latest"
APP_IMAGE="electrans:latest"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Argument parsing ──────────────────────────────────────────────────────────
REBUILD_BASE=false
for arg in "$@"; do
    case $arg in
        --rebuild-base)
            REBUILD_BASE=true
            ;;
        --help|-h)
            sed -n '2,14p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            exit 1
            ;;
    esac
done

cd "$SCRIPT_DIR"

# ── Phase 1: Base image ───────────────────────────────────────────────────────
if $REBUILD_BASE || ! docker image inspect "$BASE_IMAGE" > /dev/null 2>&1; then
    echo "======================================================================"
    echo " Phase 1: Building base image ($BASE_IMAGE)"
    echo " This compiles ~479 packages and will take a long time."
    echo "======================================================================"
    docker build \
        -f Dockerfile.base \
        -t "$BASE_IMAGE" \
        .
    echo "======================================================================"
    echo " Phase 1 complete: $BASE_IMAGE"
    echo "======================================================================"
else
    echo "======================================================================"
    echo " Phase 1 skipped: $BASE_IMAGE already exists."
    echo " Pass --rebuild-base to force a full base rebuild."
    echo "======================================================================"
fi

# ── Phase 2: Overlay image ────────────────────────────────────────────────────
echo "======================================================================"
echo " Phase 2: Building overlay image ($APP_IMAGE)"
echo "======================================================================"
docker build \
    -f Dockerfile \
    -t "$APP_IMAGE" \
    .
echo "======================================================================"
echo " Phase 2 complete: $APP_IMAGE"
echo ""
echo " Run with: ./run_image.sh"
echo "======================================================================"
