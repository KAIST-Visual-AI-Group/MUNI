#!/usr/bin/env bash
# Build the canonical 230K PolyMNIST-Quadrant-Labels uint8 tensor cache directly.
# No per-sample PNG files are created.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
: "${DATA_ROOT:?Set DATA_ROOT to the parent directory for PolyMNIST data.}"

PYTHON="${PYTHON:-python}"
OUT_DIR="${POLYMNIST_ROOT:-$DATA_ROOT/poly_mnist}"
SOURCE_ROOT="${POLYMNIST_SOURCES_ROOT:-$DATA_ROOT/poly_mnist_sources}"
MNIST_ROOT="${MNIST_ROOT:-$SOURCE_ROOT}"

ARGS=(
    --out-dir "$OUT_DIR"
    --source-dir "$SOURCE_ROOT"
    --mnist-root "$MNIST_ROOT"
)

append_value() {
    local option="$1"
    local value="$2"
    [[ -z "$value" ]] || ARGS+=("$option" "$value")
}

if [[ "${DOWNLOAD_MNIST:-1}" == "1" ]]; then
    ARGS+=(--download)
fi
append_value --num-samples "${NUM_SAMPLES:-}"
append_value --num-modalities "${NUM_MODALITIES:-}"
append_value --seed "${SEED:-}"
append_value --background-dir "${BACKGROUND_DIR:-}"
append_value --background-preset "${BACKGROUND_PRESET:-}"
if [[ -n "${CACHE_MODALITIES:-}" ]]; then
    read -r -a CACHE_MODALITY_ARGS <<< "$CACHE_MODALITIES"
    ARGS+=(--cache-modalities "${CACHE_MODALITY_ARGS[@]}")
fi
if [[ "${OVERWRITE:-0}" == "1" ]]; then
    ARGS+=(--overwrite)
fi

cd "$REPO_ROOT"
exec "$PYTHON" -m scripts.data_prep.build_polymnist_cache "${ARGS[@]}" "$@"
