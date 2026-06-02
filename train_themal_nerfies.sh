#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 <nerfies_source_path> <stage1_output> [stage2_output]"
    exit 1
fi

SOURCE_PATH="$1"
STAGE1_OUTPUT="$2"
STAGE2_OUTPUT="${3:-${STAGE1_OUTPUT}_stage2}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -f "$SOURCE_PATH/dataset.json" ] && \
   [ ! -f "$SOURCE_PATH/thermal/dataset.json" ] && \
   [ ! -f "$SOURCE_PATH/rgb/dataset.json" ]; then
    echo "Error: Nerfies dataset.json was not found under:"
    echo "  $SOURCE_PATH"
    echo "  $SOURCE_PATH/thermal"
    echo "  $SOURCE_PATH/rgb"
    exit 1
fi

python "$ROOT_DIR/train_themal_stage1.py" -s "$SOURCE_PATH" -m "$STAGE1_OUTPUT"

if [ -e "$STAGE2_OUTPUT" ]; then
    echo "Error: stage-2 output already exists: $STAGE2_OUTPUT"
    echo "Choose a new output path or remove the existing directory explicitly."
    exit 1
fi

cp -r "$STAGE1_OUTPUT" "$STAGE2_OUTPUT"
python "$ROOT_DIR/train_themal_stage2.py" -s "$SOURCE_PATH" -m "$STAGE2_OUTPUT"
