#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

SPLIT="llava_vqav2_mscoco_test-dev2015"

#######################  checkpoint + output path + latency
CKPT="zhuoyanxu/ada-llava-L-v1.5-7b"
output_path="adallava-L-7b"
latency=0.6
#######################

echo "inside vqav2, CKPT: ${CKPT} | output_path: ${output_path} | latency: ${latency}"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m src.adallava.eval.model_vqa_loader \
        --model-path $CKPT \
        --question-file data/eval/vqav2/$SPLIT.jsonl \
        --image-folder data/eval/vqav2/test2015 \
        --answers-file data/eval/vqav2/answers/$SPLIT/$output_path/latency_${latency}/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --model-name ada_llava_llama \
        --latency ${latency} & 
        
done



wait


output_file=data/eval/vqav2/answers/$SPLIT/$output_path/latency_${latency}/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat data/eval/vqav2/answers/$SPLIT/$output_path/latency_${latency}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done


python scripts/convert/convert_vqav2_for_submission.py \
        --split $SPLIT \
        --ckpt $output_path/latency_${latency} \
        --dir "data/eval/vqav2"


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/eval/vqav2.sh

