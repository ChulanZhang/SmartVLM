#!/bin/bash
# Evaluate the vision token scheduler (Exp1) at multiple token budgets.
# Uses model_vqa_loader with --token_budget. Set model_path, question_file, etc. via env if needed.
# See docs/EXP1_ADAPTIVE_VISION_TOKEN_PRUNE_IMPLEMENTATION.md and README.md.
# Paths use lowercase by convention (see .cursor/rules/path-conventions.mdc).

model_path="${model_path:-checkpoints/ada-llava-vision-token-scheduler-v1.5-7b}"
question_file="${question_file:-data/eval/vqav2/llava_vqav2_mscoco_test-dev2015.jsonl}"
image_folder="${image_folder:-data/eval/vqav2/test2015}"
output_base="${output_base:-data/eval/vqav2/answers/llava_vqav2_mscoco_test-dev2015/vision-token-scheduler}"
split="llava_vqav2_mscoco_test-dev2015"

# Token budgets to sweep (ratio in [0,1])
budgets="${eval_budgets:-0.25 0.5 0.75 1.0}"

for tb in $budgets; do
  echo "Evaluating token_budget=$tb"
  out_dir="$output_base/token_budget_${tb}"
  mkdir -p "$out_dir"
  python -m src.adallava.eval.model_vqa_loader \
    --model-path "$model_path" \
    --question-file "$question_file" \
    --image-folder "$image_folder" \
    --answers-file "$out_dir/answers.jsonl" \
    --num-chunks 1 \
    --chunk-idx 0 \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --model-name ada_llava_llama \
    --latency 1.0 \
    --token_budget "$tb"
done

echo "Done. Results under $output_base/token_budget_*"
