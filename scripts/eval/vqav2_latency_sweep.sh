#!/bin/bash
# Sweep latency budget from 0.5 to 1.0 (step 0.05) and run lmms-eval vqav2_val for each.
# Results go to ./results/logs_latency_sweep_vqav2_val/<latency>/ or similar.
# Usage: bash scripts/eval/vqav2_latency_sweep.sh
#
# Avoid PermissionError on shared cache: disable file locking so no lock file is created.
# Pass env inline so accelerate-launched subprocesses see it (don't change HF_DATASETS_CACHE
# so existing shared cache is reused and data is not copied again).
# Optional: DEBUG=1 to enable per-sample debug logs. Default off.
export HF_DATASETS_DISABLE_FILE_LOCKING=1
DEBUG="${DEBUG:-0}"
export DEBUG

set -e

MODEL="adallava"
MODEL_ARGS_BASE="pretrained=zhuoyanxu/ada-llava-L-v1.5-7b"
TASKS="vqav2_val"
BATCH_SIZE=1
BASE_OUTPUT="./results/logs_latency_sweep_vqav2_val"

for latency in 0.95; do
  echo "========== latency=${latency} =========="
  HF_DATASETS_DISABLE_FILE_LOCKING=1 python3 -m accelerate.commands.launch \
    -m adallava.eval.run_lmms_eval \
    --model "$MODEL" \
    --model_args "${MODEL_ARGS_BASE},latency=${latency}" \
    --tasks "$TASKS" \
    --batch_size "$BATCH_SIZE" \
    --log_samples \
    --log_samples_suffix "adallava_${latency}" \
    --output_path "${BASE_OUTPUT}/latency_${latency}/"
done

echo "Done. Results under ${BASE_OUTPUT}/"
