#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <nerfies_source_path> <output_root>"
    echo "Example: $0 /data/DynamicRGBT/HairDryer /data/outputs/HairDryer"
    exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_PATH="$(realpath -m "$1")"
OUTPUT_ROOT="$(realpath -m "$2")"
STAGE1_OUTPUT="${OUTPUT_ROOT}/stage1"
STAGE2_OUTPUT="${OUTPUT_ROOT}/stage2"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_METRICS="${RUN_METRICS:-1}"

export PYTHONPATH="${ROOT_DIR}/nvdiffrast-main:${PYTHONPATH:-}"

find_nerfies_root() {
    local candidate
    for candidate in "$SOURCE_PATH" "$SOURCE_PATH/thermal" "$SOURCE_PATH/rgb"; do
        if [ -f "$candidate/dataset.json" ] && [ -d "$candidate/camera" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

require_point_cloud() {
    local root filename
    for root in "$SOURCE_PATH" "$NERFIES_ROOT" "$SOURCE_PATH/rgb" "$SOURCE_PATH/thermal"; do
        for filename in mesh.ply points3D.ply points3d.ply points.npy; do
            if [ -f "$root/$filename" ]; then
                return 0
            fi
        done
    done
    echo "Error: Nerfies point cloud not found."
    echo "Provide mesh.ply, points3D.ply, points3d.ply, or points.npy."
    exit 1
}

if [ ! -d "$SOURCE_PATH" ]; then
    echo "Error: source directory does not exist: $SOURCE_PATH"
    exit 1
fi

if ! NERFIES_ROOT="$(find_nerfies_root)"; then
    echo "Error: Nerfies dataset.json and camera/ were not found under:"
    echo "  $SOURCE_PATH"
    echo "  $SOURCE_PATH/thermal"
    echo "  $SOURCE_PATH/rgb"
    exit 1
fi

require_point_cloud

NERFIES_ROOT="$NERFIES_ROOT" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["NERFIES_ROOT"])
with (root / "dataset.json").open("r", encoding="utf-8") as dataset_file:
    ids = list(json.load(dataset_file).get("ids", []))
if not ids:
    raise SystemExit(f"Error: no ids found in {root / 'dataset.json'}")

image_dir = None
for dirname in ("rgb", "images"):
    candidate = root / dirname
    if not candidate.is_dir():
        continue
    image_dir = next(
        (candidate / scale for scale in ("2x", "1x", "4x") if (candidate / scale).is_dir()),
        candidate,
    )
    break
if image_dir is None:
    raise SystemExit(f"Error: rgb/ or images/ was not found under {root}")

extensions = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")
for image_id in ids:
    camera_path = root / "camera" / f"{image_id}.json"
    if not camera_path.is_file():
        raise SystemExit(f"Error: missing camera file: {camera_path}")
    if not any((image_dir / f"{image_id}{extension}").is_file() for extension in extensions):
        raise SystemExit(f"Error: missing Thermal image for '{image_id}' under {image_dir}")

print(f"Nerfies preflight passed: {len(ids)} frames from {root}")
PY

if [ -e "$STAGE1_OUTPUT" ] || [ -e "$STAGE2_OUTPUT" ]; then
    echo "Error: output already exists. Choose a new output_root:"
    echo "  $STAGE1_OUTPUT"
    echo "  $STAGE2_OUTPUT"
    exit 1
fi

"$PYTHON_BIN" - <<'PY'
import sys
import torch

device_count = torch.cuda.device_count()
if device_count < 2:
    raise SystemExit(
        f"Error: NTR-Gaussian requires at least 2 CUDA GPUs, but found {device_count}."
    )
print("CUDA preflight passed:", ", ".join(
    f"cuda:{index}={torch.cuda.get_device_name(index)}" for index in range(device_count)
))
PY

mkdir -p "$OUTPUT_ROOT"

echo "[1/4] Training stage 1"
"$PYTHON_BIN" "$ROOT_DIR/train_themal_stage1.py" -s "$SOURCE_PATH" -m "$STAGE1_OUTPUT"

echo "[2/4] Preparing and training stage 2"
cp -r "$STAGE1_OUTPUT" "$STAGE2_OUTPUT"
"$PYTHON_BIN" "$ROOT_DIR/train_themal_stage2.py" -s "$SOURCE_PATH" -m "$STAGE2_OUTPUT"

echo "[3/4] Rendering grayscale and Ironbow test images"
"$PYTHON_BIN" "$ROOT_DIR/render_themal_stage2.py" -s "$SOURCE_PATH" -m "$STAGE2_OUTPUT" --skip_train

if [ "$RUN_METRICS" = "1" ]; then
    echo "[4/4] Evaluating Ironbow predictions against Thermal pseudocolor GT"
    "$PYTHON_BIN" "$ROOT_DIR/metrics_temp.py" -m "$STAGE2_OUTPUT"
else
    echo "[4/4] Skipping metrics because RUN_METRICS=$RUN_METRICS"
fi

echo "Done. Stage-2 output: $STAGE2_OUTPUT"
