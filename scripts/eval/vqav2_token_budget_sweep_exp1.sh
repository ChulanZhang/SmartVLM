#!/bin/bash
# Exp1 vision token scheduler: sweep token_budget and run lmms-eval vqav2_val for each.
# Use for full profiling (accuracy + FLOPs / prefill_time / memory) across budgets.
# Run from repo root: bash scripts/eval/vqav2_token_budget_sweep_exp1.sh
#
# Uses accelerate default (all visible GPUs). Set CUDA_VISIBLE_DEVICES to limit GPUs.
# Optional: LIMIT=N to run only N samples per budget (for quick sweep). Omit for full eval.
set -e

export HF_DATASETS_DISABLE_FILE_LOCKING=1

MODEL="adallava"
MODEL_ARGS_BASE="pretrained=checkpoints/ada-llava-vision-token-scheduler-v1.5-7b"
TASKS="vqav2_val"
BATCH_SIZE=1
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29502}"
BASE_OUTPUT="./results/logs_token_budget_sweep_vqav2_val_exp1"

# Token budgets to sweep (vision token scheduler Exp1)
for token_budget in 0.25 0.5 0.75 1.0; do
  echo "========== token_budget=${token_budget} =========="
  LIMIT_ARGS=()
  if [[ -n "${LIMIT:-}" ]]; then
    LIMIT_ARGS=(--limit "$LIMIT")
  fi
  HF_DATASETS_DISABLE_FILE_LOCKING=1 python3 -m accelerate.commands.launch \
    --main_process_port "$MAIN_PROCESS_PORT" \
    -m adallava.eval.run_lmms_eval \
    --model "$MODEL" \
    --model_args "${MODEL_ARGS_BASE},token_budget=${token_budget},latency=1.0" \
    --tasks "$TASKS" \
    --batch_size "$BATCH_SIZE" \
    "${LIMIT_ARGS[@]}" \
    --log_samples \
    --log_samples_suffix "exp1_${token_budget}" \
    --output_path "${BASE_OUTPUT}/token_budget_${token_budget}/"
done

echo "Done. Results under ${BASE_OUTPUT}/"
