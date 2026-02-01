#!/bin/bash
# Exp1 quick test: one token_budget, single GPU, 10 samples only.
# Use for sanity-check before full sweep (vqav2_token_budget_sweep_exp1.sh).
#
# lmms_eval supports --limit N to run only N samples; change LIMIT below to adjust.
# Run from repo root (after sourcing activate_env.sh if needed): bash scripts/eval/vqav2_token_budget_test_exp1.sh

set -e

export HF_DATASETS_DISABLE_FILE_LOCKING=1

MODEL="adallava"
MODEL_ARGS_BASE="pretrained=checkpoints/ada-llava-vision-token-scheduler-v1.5-7b"
TASKS="vqav2_val"
BATCH_SIZE=1
# Single token_budget for test
TOKEN_BUDGET="${TOKEN_BUDGET:-0.5}"
# Number of samples (lmms_eval --limit)
LIMIT="${LIMIT:-10}"
# Single GPU: 1 process
NUM_PROCESSES=1
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29502}"
BASE_OUTPUT="./results/logs_token_budget_test_vqav2_val_exp1"
# Per-sample debug logs (Exp1/FLOPs): set DEBUG=1 to enable (default off)
DEBUG="${DEBUG:-0}"
export DEBUG

echo "========== token_budget=${TOKEN_BUDGET} limit=${LIMIT} samples (single GPU) =========="
HF_DATASETS_DISABLE_FILE_LOCKING=1 python3 -m accelerate.commands.launch \
  --main_process_port "$MAIN_PROCESS_PORT" \
  --num_processes "$NUM_PROCESSES" \
  -m adallava.eval.run_lmms_eval \
  --model "$MODEL" \
  --model_args "${MODEL_ARGS_BASE},token_budget=${TOKEN_BUDGET},latency=1.0" \
  --tasks "$TASKS" \
  --batch_size "$BATCH_SIZE" \
  --limit "$LIMIT" \
  --log_samples \
  --log_samples_suffix "exp1_test_${TOKEN_BUDGET}" \
  --output_path "${BASE_OUTPUT}/token_budget_${TOKEN_BUDGET}/"

echo "Done. Results under ${BASE_OUTPUT}/"
