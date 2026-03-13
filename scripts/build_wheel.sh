#!/usr/bin/env bash
# build_wheel.sh — Build the pyGpufit wheel inside a Docker container.
#
# All source is bind-mounted from the host; nothing is baked into the image.
#
# Usage:
#   bash scripts/build_wheel.sh [options]
#
# Options:
#   --cuda-version VER   CUDA base image version  (default: 12.4.0 / env: CUDA_VERSION)
#   --cuda-arch ARCH     CMake CUDA_ARCH value     (default: All    / env: CUDA_ARCH)
#   --no-gpu             Build without --gpus flag (pass explicit --cuda-arch)
#   --no-cache           Force rebuild of the Docker image
#   --jobs N             Parallel make jobs        (default: nproc inside container)
#
# Environment:
#   WHEEL_OUT_DIR        Where to write the .whl   (default: <repo-root>/dist/gpufit)
#
# Output:
#   $WHEEL_OUT_DIR/pyGpufit-*.whl
set -euo pipefail

GPUFIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CUDA_VERSION="${CUDA_VERSION:-12.4.0}"
CUDA_ARCH="${CUDA_ARCH:-All}"
WHEEL_OUT_DIR="${WHEEL_OUT_DIR:-${GPUFIT_DIR}/dist/gpufit}"
USE_GPU="1"
NO_CACHE=""
CMAKE_JOBS="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda-version) CUDA_VERSION="$2"; shift 2 ;;
    --cuda-arch)    CUDA_ARCH="$2";    shift 2 ;;
    --no-gpu)       USE_GPU="0";       shift ;;
    --no-cache)     NO_CACHE="--no-cache"; shift ;;
    --jobs)         CMAKE_JOBS="$2";   shift 2 ;;
    *) echo "[build_wheel] Unknown argument: $1" >&2; exit 1 ;;
  esac
done

IMAGE_TAG="gpufit-builder:${CUDA_VERSION}"

if ! command -v docker &>/dev/null; then
  echo "[build_wheel] ERROR: docker not found on PATH." >&2; exit 1
fi

echo "[build_wheel] CUDA version : ${CUDA_VERSION}"
echo "[build_wheel] CUDA_ARCH    : ${CUDA_ARCH}"
echo "[build_wheel] GPU          : $([ "${USE_GPU}" = "1" ] && echo yes || echo no)"
echo "[build_wheel] Output       : ${WHEEL_OUT_DIR}"

mkdir -p "${WHEEL_OUT_DIR}"

# ── Build the image (uses the Dockerfile at the repo root) ──────────────────
TMPCTX="$(mktemp -d)"
trap 'rm -rf "${TMPCTX}"' EXIT

echo "[build_wheel] Building image ${IMAGE_TAG}…"
# shellcheck disable=SC2086
docker build \
  ${NO_CACHE} \
  --build-arg "CUDA_VERSION=${CUDA_VERSION}" \
  -t "${IMAGE_TAG}" \
  -f "${GPUFIT_DIR}/Dockerfile" \
  "${TMPCTX}"

# ── Run CMake + wheel packaging inside the container ────────────────────────
GPU_FLAG=""
[ "${USE_GPU}" = "1" ] && GPU_FLAG="--gpus all"

[ "${CMAKE_JOBS}" = "0" ] && JOBS_EXPR="\$(nproc)" || JOBS_EXPR="${CMAKE_JOBS}"

echo "[build_wheel] Running build inside container…"
# shellcheck disable=SC2086
docker run --rm \
  ${GPU_FLAG} \
  -v "${GPUFIT_DIR}:/src/Gpufit:ro" \
  -v "${WHEEL_OUT_DIR}:/dist/gpufit" \
  "${IMAGE_TAG}" \
  bash -c "
    set -euo pipefail
    cmake -S /src/Gpufit -B /build/gpufit \
      -DCMAKE_BUILD_TYPE=Release \
      -DCUDA_ARCH=${CUDA_ARCH}
    cmake --build /build/gpufit --config Release --parallel ${JOBS_EXPR}
    WHEEL=\$(find /build/gpufit -name '*.whl' -print -quit 2>/dev/null || true)
    [ -z \"\${WHEEL}\" ] && { echo '[build_wheel] ERROR: no .whl found.' >&2; exit 1; }
    cp \"\${WHEEL}\" /dist/gpufit/
    echo \"[build_wheel] wheel ready: \$(basename \${WHEEL})\"
  "

echo "[build_wheel] Done. Wheel in: ${WHEEL_OUT_DIR}"
ls -1 "${WHEEL_OUT_DIR}"
