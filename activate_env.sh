#!/bin/bash

module --force purge
module load modtree/gpu
module load gcc/11.2.0
module load cuda/12.0.1
module load cudnn/cuda-12.0_8.8
module load conda/2024.09
module use /anvil/projects/x-cis250705/modules
module load conda-env/smartvlm-py3.10

# Hugging Face cache/datasets (models, tokenizers, etc.)
export HF_HOME=/anvil/projects/x-cis250705/data/vlm/huggingface
