#!/usr/bin/env bash
# Thin wrapper for the supplied-media cache builder; all options belong to Python.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
exec "${PYTHON:-python}" -m scripts.data_prep.build_image_text_audio_cache "$@"
