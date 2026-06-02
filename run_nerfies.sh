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

"$ROOT_DIR/train_themal_nerfies.sh" "$SOURCE_PATH" "$STAGE1_OUTPUT" "$STAGE2_OUTPUT"

python "$ROOT_DIR/render_themal_stage2.py" -s "$SOURCE_PATH" -m "$STAGE2_OUTPUT" --skip_train
python "$ROOT_DIR/metrics_temp.py" -m "$STAGE2_OUTPUT"
