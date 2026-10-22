#!/usr/bin/env bash
# Distributed training launcher.
# Usage:
#   bash scripts/train.sh configs/image_text_audio/muni_poe.yaml
#   bash scripts/train.sh configs/image_text_audio/muni_poe.yaml --port 29502
#   bash scripts/train.sh configs/image_text_audio/muni_poe.yaml --gpus 4 train.epochs=100
#   bash scripts/train.sh configs/image_text_audio/muni_poe.yaml --resume output/image_text_audio/muni_poe

set -euo pipefail

if (( $# == 0 )) || [[ "$1" == -* ]]; then
    echo "Usage: bash scripts/train.sh <config.yaml> [--gpus N] [--port PORT] [train.py arguments ...]"
    exit 1
fi

CONFIG="$1"
shift

NGPUS=1
MASTER_PORT=29500
TRAIN_ARGS=()

while (( $# > 0 )); do
    case "$1" in
        --gpus)
            [[ -n "${2:-}" ]] || { echo "Error: --gpus requires a value." >&2; exit 1; }
            NGPUS="$2"
            shift 2
            ;;
        --port)
            [[ -n "${2:-}" ]] || { echo "Error: --port requires a value." >&2; exit 1; }
            MASTER_PORT="$2"
            shift 2
            ;;
        *)
            TRAIN_ARGS+=("$1")
            shift
            ;;
    esac
done

echo "Config:      ${CONFIG}"
echo "GPUs:        ${NGPUS}"
echo "Master port: ${MASTER_PORT}"
(( ${#TRAIN_ARGS[@]} == 0 )) || echo "Train args:  ${TRAIN_ARGS[*]}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
PYTHON="${PYTHON:-python}"
cd "$REPO_ROOT"
exec "$PYTHON" -m torch.distributed.run \
    --nproc_per_node="$NGPUS" \
    --master_port="$MASTER_PORT" \
    --module scripts.train --config "$CONFIG" "${TRAIN_ARGS[@]}"
