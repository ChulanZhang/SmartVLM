#!/bin/bash
# Train the vision token scheduler (Exp1: adaptive vision token prune).
# Uses token_selecting="adaptive"; vision controller is trained with random token budget per batch.
# See docs/EXP1_ADAPTIVE_VISION_TOKEN_PRUNE_IMPLEMENTATION.md and README.md for data setup.
#
# Wandb: run once to log in: wandb login
# Optional: export WANDB_API_KEY=your_key      (if you cannot run wandb login)
export WANDB_PROJECT=adallava
# Uncomment one to choose where runs are logged:
# export WANDB_ENTITY=pengchengwang92                    # personal account
# export WANDB_ENTITY=pengchengwang92-purdue-university  # team (default if logged in via team)
RUN_NUM=""

original_model="liuhaotian/llava-v1.5-7b"

deepspeed ./src/adallava/train/train_mem.py \
    --deepspeed ./LLaVA/scripts/zero3.json \
    --model_name_or_path  ${original_model}\
    --version v1 \
    --data_path ./data/llava_v1_5_mix665k.json \
    --image_folder ./data/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --freeze_backbone False \
    --bf16 True \
    --num_prefix_layers 16 \
    --token_selecting "adaptive" \
    --scheduler_type "L" \
    --vision_controller_budget_min 0.2 \
    --vision_controller_budget_max 1.0 \
    --vision_controller_tau 5.0 \
    --num_vision_patches 576 \
    --mm_hidden_size 1024 \
    --output_dir checkpoints/ada-llava-vision-token-scheduler-v1.5-7b \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name ada-llava-vision-token-scheduler-v1.5-7b${RUN_NUM} \
