#!/usr/bin/env bash
#
# run_parallel.sh — launch two SDXL FP32 pipeline-parallel jobs on 4 GPUs.
#
#   Pipeline A: GPU 0 (down) + GPU 1 (up)
#   Pipeline B: GPU 2 (down) + GPU 3 (up)
#
# Usage:
#   ./run_parallel.sh /path/to/model
#   ./run_parallel.sh /path/to/model "prompt A" "prompt B"
#
set -euo pipefail

MODEL="${1:-./models/sdxl-base-fp16}"
PROMPT_A="${2:-A cyberpunk cityscape at night, neon lights reflecting in wet streets, 8k}"
PROMPT_B="${3:-A peaceful mountain lake at dawn, mist rising, photorealistic}"
STEPS="${STEPS:-25}"
PY="${PY:-python}"

echo "[Pipeline A] GPU 0+1 -> out_0.png"
"$PY" generate_one.py \
    --model "$MODEL" \
    --gpu-down 0 --gpu-up 1 \
    --prompt "$PROMPT_A" \
    --steps "$STEPS" \
    --output out_0.png &

echo "[Pipeline B] GPU 2+3 -> out_1.png"
"$PY" generate_one.py \
    --model "$MODEL" \
    --gpu-down 2 --gpu-up 3 \
    --prompt "$PROMPT_B" \
    --steps "$STEPS" \
    --output out_1.png &

wait
echo "Both images ready: out_0.png, out_1.png"
