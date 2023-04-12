#!/usr/bin/env bash

set -xe

CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun \
    --nnodes 1 --nproc_per_node=4 --master_port=1234 train.py \
    --model_name_or_path roberta-large \
    --num_labels 3 \
    --train_data_path ./datasets/snli_train.csv \
    --eval_data_path datasets/snli_validation.csv \
    --test_data_path ./datasets/NLI_diagnostic.csv \
    --output_dir ../../data/DADC/NAME_OF_PATH \
    --num_train_epochs 15 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --lr_scheduler_type "linear" \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --logging_steps 1 \
    --load_best_model_at_end True \
    --fsdp "full_shard auto_wrap" \
    --early_stopping_patience 3 \
    --metric_for_best_model "accuracy"