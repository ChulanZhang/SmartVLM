#!/bin/bash
# Run vision token prune demo for token_budget from 0.2 to 1.0 (step 0.1).
# Usage (from repo root):
#   bash scripts/demo_vision_token_prune_sweep.sh
#   MODEL_PATH=checkpoints/ada-llava-vision-token-scheduler-v1.5-7b IMAGE=docs/snowman.jpg bash scripts/demo_vision_token_prune_sweep.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

MODEL_PATH="${MODEL_PATH:-checkpoints/ada-llava-vision-token-scheduler-v1.5-7b}"
IMAGE="${IMAGE:-docs/snowman.jpg}"
QUERY="${QUERY:-What is in this image?}"
SEED="${SEED:-42}"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$REPO_ROOT/src"

echo "========== Vision token budget sweep 0.2 -> 1.0 =========="
echo "model=$MODEL_PATH image=$IMAGE query=$QUERY seed=$SEED"
echo ""

for budget in 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
  echo "--- token_budget=$budget ---"
  python3 scripts/demo_vision_token_prune.py \
    --model-path "$MODEL_PATH" \
    --image-file "$IMAGE" \
    --token_budget "$budget" \
    --query "$QUERY" \
    --seed "$SEED" \
    --brief
  echo ""
done

echo "Done. Sweep 0.2 to 1.0 finished."
