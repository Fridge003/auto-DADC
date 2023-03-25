#!/usr/bin/env bash

set -xe

# result_path: A json file that stores the result of prediction
# device: set to -1 if using cpu, else set to the gpu device number
python predict.py \
    --model_name_or_path ./dadc_result/finetuned_model_6 \
    --test_data_path ./datasets/NLI_diagnostic.csv \
    --result_path ./dadc_result/prediction_result.json \
    --predict_batch_size 16 \
    --device 0 \
    --loop_cnt 6