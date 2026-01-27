#!/bin/bash

module --force purge
module load modtree/gpu
module load gcc/11.2.0
module load cuda/12.8.0
module load cudnn/cuda-12.8_9.17
module load conda/2025.02
module use /anvil/projects/x-cis250705/modules
module load conda-env/smartvlm-py3.12.8


