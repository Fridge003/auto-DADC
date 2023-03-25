#!/usr/bin/env bash

# set -xe


# ----------------------------1. Declare Variables-----------------------------------

# Openai Key.
openai_api_key=REPLACE_YOUR_KEY_HERE

# Visible Cuda Devices/number of visible cuda devices during fine-tuning. 
cuda_visible_devices="0,1,2,3"
num_cuda_devices=4

# Total number of loops for DADC.
total_loop=10

# This should be the main working directory that stores following files:
# 1. All the collected data through prompting GPT
# 2. All the models fine-tuned at each cycle.
# 3. The performance on a certain test dataset for the fine-tuned model at each loop.
working_dir="./dadc_result"

# A model name or path of a HuggingFace model, as the starting model of dadc loop.
# This model can be a pretrained model provided by HuggingFace,
# or a model fine-tuned on NLI datasets such as SNLI or MNLI.
base_model="../../data/DADC/finetuned-roberta-large"

# A path to dataset(a .csv file) for testing model performance.
test_data_path="./datasets/NLI_diagnostic.csv"

# An integer indicating device for making prediction on test set.
# If using cpu, set it to -1, else set it to the id of gpu for using.
test_device=0

# A path to seed dataset(a .csv file) for generating new examples.
seed_data_path="./datasets/snli_validation.csv"

# Number of examples for generation at each loop.
num_examples_generated_per_loop=600

# Number of epochs during fine-tuning.
num_epoch=15

# Patience of early stopping during fine-tuning.
early_stopping_patience=3

# Batch Size for each cuda device during training/evalutating/predicting.
per_device_train_batch_size=32
per_device_eval_batch_size=32
predict_batch_size=32


# --------------------------------2. Run Script----------------------------------------

mkdir -p $working_dir
generated_train_data="$working_dir/gen_train.csv"
generated_eval_data="$working_dir/gen_validation.csv"
result_record="$working_dir/prediction_result.json"


# First test the performance of starting model.
python predict.py \
    --model_name_or_path $base_model \
    --test_data_path $test_data_path \
    --result_path $result_record \
    --predict_batch_size $predict_batch_size \
    --device $test_device \
    --loop_cnt 0


# Start DADC Loop
loop_cnt=1
last_model_path="$base_model"
while [ $loop_cnt -le $total_loop ]
do

# Create the folder that stores model fine-tuned at this loop.
next_model_path="$working_dir/finetuned_model_$loop_cnt"
mkdir -p $next_model_path

# Prompt ChatGPT for examples.
OPENAI_API_KEY=$openai_api_key \
python -m data_generation generate_data_by_prompting \
    --output_dir $working_dir \
    --seed_dataset_path  $seed_data_path \
    --num_examples_to_generate $num_examples_generated_per_loop \
    --validation_ratio 0.1 \
    --model_name gpt-3.5-turbo \
    --num_prompt_examples 3 \
    --num_genetated_examples_per_prompt 5 \
    --temperature 1.0 \
    --num_cpus 8 \

# Finetune model on data collected.
CUDA_VISIBLE_DEVICES=$cuda_visible_devices torchrun \
    --nnodes=1 --nproc_per_node=$num_cuda_devices --master_port=1234 train.py \
    --model_name_or_path $last_model_path \
    --train_data_path $generated_train_data \
    --eval_data_path $generated_eval_data \
    --test_data_path $test_data_path \
    --output_dir $next_model_path \
    --num_train_epochs $num_epoch \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size $per_device_eval_batch_size \
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
    --metric_for_best_model "accuracy" \
    --early_stopping_patience $early_stopping_patience

# Test performance on newly-finetuned model.
python predict.py \
    --model_name_or_path $next_model_path \
    --test_data_path $test_data_path \
    --result_path $result_record \
    --predict_batch_size $predict_batch_size \
    --device $test_device \
    --loop_cnt $loop_cnt

# Update State
last_model_path="$next_model_path"
((loop_cnt++))

done

