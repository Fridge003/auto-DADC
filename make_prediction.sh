#!/usr/bin/env bash

set -xe

# result_path: A json file that stores the result of prediction
# device: set to -1 if using cpu, else set to the gpu device number
python predict.py \
    --model_name_or_path ../../data/dadc_results/finetuned-roberta-large \
    --num_labels 3 \
    --test_data_path ./datasets/NLI_diagnostic.csv \
    --result_path ./prediction_result.json \
    --predict_batch_size 96 \
    --device 2 \
    --loop_cnt 0
    # --model_name_or_path ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli