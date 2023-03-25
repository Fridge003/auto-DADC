#!/usr/bin/env bash

set -xe

# validtion_ratio: proportion of examples for validation set

OPENAI_API_KEY=REPLACE_YOUR_KEY_HERE \
python -m data_generation generate_data_by_prompting \
    --output_dir ./collected_data \
    --seed_dataset_path ./datasets/snli_validation.csv \
    --num_examples_to_generate 50 \
    --validation_ratio 0.05 \
    --model_name gpt-3.5-turbo \
    --num_prompt_examples 3 \
    --num_genetated_examples_per_prompt 5 \
    --temperature 1.0 \
    --num_cpus 8 \
