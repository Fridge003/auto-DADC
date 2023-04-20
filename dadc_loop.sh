#!/usr/bin/env bash

# set -xe


# ----------------------------1. Declare Variables-----------------------------------

# Openai Key.
openai_api_key=REPLACE_YOUR_KEY_HERE

# Visible Cuda Devices/number of visible cuda devices during fine-tuning. 
cuda_visible_devices="0,1,2,3"
num_cuda_devices=4

# An integer indicating device for making prediction on test set.
# If using cpu, set it to -1, else set it to the id of gpu for using.
test_device=0

# Total number of loops for DADC.
total_loop=10

# Checkpointing. The dadc loop will start from starting_loop.
# Under default setting, starting_loop will be set to 0 to start from empty workspace.
starting_loop=0

# This should be the main working directory that stores following files:
# 1. All the collected data through prompting GPT
# 2. All the models fine-tuned at each cycle.
# 3. The performance on a certain test dataset for the fine-tuned model at each loop.
working_dir=WORKING_DIR

# A model name or path of a HuggingFace model, as the starting model of dadc loop.
# This model can be a pretrained model provided by HuggingFace,
# or a model fine-tuned on NLI datasets such as SNLI or MNLI.
base_model=BASE_MODEL

# A path to dataset(a .csv file) for testing model performance.
test_data_path=TEST_DATA_PATH

# A path to seed dataset(a .csv file) for generating new examples.
seed_data_path=SEED_DATA_PATH

# Number of examples for generation at each loop.
num_examples_generated_per_loop=600

# Number of epochs during fine-tuning.
num_epoch=15

# Patience of early stopping during fine-tuning.
early_stopping_patience=3

# 2:binary NLI/3:classical NLI
num_labels=3

# Batch Size for each cuda device during training/evalutating/predicting.
per_device_train_batch_size=32
per_device_eval_batch_size=32
predict_batch_size=32


# --------------------------------2. Run Script----------------------------------------


all_models_path="$working_dir/models"
all_collected_data_path="$working_dir/collected_data"
result_record="$working_dir/prediction_result.json"

# Default setting: starting from base model
loop_cnt=1
last_model_path="$base_model"
last_data_path="Init"

if  [[ $starting_loop -eq 0 ]]
then
    mkdir -p $working_dir
    mkdir -p $all_models_path
    mkdir -p $all_collected_data_path

    # First test the performance of starting model.
    python predict.py \
        --model_name_or_path $base_model \
        --num_labels $num_labels \
        --test_data_path $test_data_path \
        --result_path $result_record \
        --predict_batch_size $predict_batch_size \
        --device $test_device \
        --num_labels $num_labels \
        --loop_cnt 0

else 
    # Starting from Checkpoint
    loop_cnt=$starting_loop
    last_loop=$((starting_loop-1))
    last_model_path="$all_models_path/finetuned_model_$last_loop"
    last_data_path="$all_collected_data_path/collected_data_$last_loop"
fi



while [ $loop_cnt -le $total_loop ]
do

    # Create the folder that stores model fine-tuned at this loop.
    next_model_path="$all_models_path/finetuned_model_$loop_cnt"
    next_data_path="$all_collected_data_path/collected_data_$loop_cnt"
    mkdir -p $next_model_path
    mkdir -p $next_data_path

    # Prompt ChatGPT for examples.
    OPENAI_API_KEY=$openai_api_key \
    python -m data_generation generate_data_by_prompting \
        --input_dir $last_data_path \
        --output_dir $next_data_path \
        --seed_dataset_path  $seed_data_path \
        --num_examples_to_generate $num_examples_generated_per_loop \
        --validation_ratio 0.1 \
        --model_name gpt-3.5-turbo \
        --num_prompt_examples 3 \
        --num_genetated_examples_per_prompt 5 \
        --temperature 1.0 \
        --num_cpus 8

    # Finetune model on data collected.
    CUDA_VISIBLE_DEVICES=$cuda_visible_devices torchrun \
        --nnodes=1 --nproc_per_node=$num_cuda_devices --master_port=1234 train.py \
        --model_name_or_path $last_model_path \
        --num_labels $num_labels \
        --train_data_path "$next_data_path/gen_train.csv" \
        --eval_data_path "$next_data_path/gen_validation.csv" \
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
        --early_stopping_patience $early_stopping_patience \
        --metric_for_best_model "accuracy"


    # Test performance on newly-finetuned model.
    python predict.py \
        --model_name_or_path $next_model_path \
        --num_labels $num_labels \
        --test_data_path $test_data_path \
        --result_path $result_record \
        --predict_batch_size $predict_batch_size \
        --device $test_device \
        --loop_cnt $loop_cnt

    # Update State
    last_model_path="$next_model_path"
    last_data_path="$next_data_path"

    ((loop_cnt++))

done

