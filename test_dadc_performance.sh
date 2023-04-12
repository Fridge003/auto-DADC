#!/usr/bin/env bash

# set -xe

# ----------------------------1. Declare Variables-----------------------------------

# An integer indicating device for making prediction on test set.
# If using cpu, set it to -1, else set it to the id of gpu for using.
test_device=3

# Total number of loops for DADC.
total_loop=10

# This should be the main working directory that stores following files:
# 1. All the collected data through prompting GPT
# 2. All the models fine-tuned at each cycle.
# 3. The performance on a certain test dataset for the fine-tuned model at each loop.
working_dir="../../data/dadc_results/naive_dadc_result_0404"

# The file for recording performance
output_file="../../data/dadc_results/naive_dadc_result_0404/prediction_on_anli_test.json"

# A model name or path of a HuggingFace model, as the starting model of dadc loop.
# This model can be a pretrained model provided by HuggingFace,
# or a model fine-tuned on NLI datasets such as SNLI or MNLI.
base_model="../../data/dadc_results/finetuned-roberta-large"

# A path to dataset(a .csv file) for testing model performance.
test_data_path="./datasets/anli_test.csv"

# Batch Size for each cuda device during predicting.
predict_batch_size=128


# --------------------------------2. Run Script----------------------------------------


# First test the performance of starting model.
python predict.py \
    --model_name_or_path $base_model \
    --test_data_path $test_data_path \
    --result_path $output_file \
    --predict_batch_size $predict_batch_size \
    --device $test_device \
    --loop_cnt 0


loop_cnt=1
last_model_path="$base_model"


while [ $loop_cnt -le $total_loop ]
do

next_model_path="$working_dir/finetuned_model_$loop_cnt"

# Test performance.
python predict.py \
    --model_name_or_path $next_model_path \
    --test_data_path $test_data_path \
    --result_path $output_file \
    --predict_batch_size $predict_batch_size \
    --device $test_device \
    --loop_cnt $loop_cnt

# Update State
last_model_path="$next_model_path"
((loop_cnt++))

done