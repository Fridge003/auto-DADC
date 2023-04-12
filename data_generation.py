import os
import random
import pandas as pd
from prompt_utils import load_csv_file_as_list, save_list_as_csv_files
from prompt_utils import completions_with_backoff, parse_response, validate_example
from prompt_utils import generation_prompt
import tqdm
from rouge_score import rouge_scorer
import fire


def generate_data_by_prompting( 
    output_dir="./collected_data",
    seed_dataset_path="./datasets/snli_validation.csv",
    num_examples_to_generate=1000,
    validation_ratio=0.05,
    model_name="gpt-3.5-turbo",
    num_prompt_examples=3,
    num_genetated_examples_per_prompt=5,
    temperature=1.0,
    top_p=1.0,
    max_tokens=512,
    num_cpus=8
):

    # get api key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")

    # arguments for prompting
    prompt_args = {"api_key":api_key, "model":model_name, "messages":[], 
                   "top_p":top_p, "temperature":temperature, "max_tokens":max_tokens}

    os.makedirs(output_dir, exist_ok=True)
    output_path_train = os.path.join(output_dir, "gen_train.csv") # Generated Training Set
    output_path_eval = os.path.join(output_dir, "gen_validation.csv") # Generated Validation Set
    output_path_disagreed = os.path.join(output_dir, "gen_disagreed.csv")

    seed_examples = load_csv_file_as_list(seed_dataset_path)
    print(f"Loaded {len(seed_examples)} seed examples")
    
    # Divide seed examples according to label
    labels = ["Entailment", "Neutral", "Contradiction"]
    divided_seed_examples = {label:[] for label in labels}
    for example in seed_examples:
        # Here example['label'] is an integer in [0, 1, 2]
        divided_seed_examples[labels[example['label']]].append(example)

    # Load the prior LM-generated examples, including a training set and a validation set
    generated_train_data, generated_eval_data = [], []
    generated_train_data += load_csv_file_as_list(output_path_train)
    generated_eval_data += load_csv_file_as_list(output_path_eval)
    print(f"Loaded {len(generated_train_data)} examples from generated training set {output_path_train}")
    print(f"Loaded {len(generated_eval_data)} examples from generated validation set {output_path_eval}")
    
    # rouge for computing similarity
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    all_example_tokens = [scorer._tokenizer.tokenize(example["premise"] + example["hypothesis"]) for example in generated_train_data] 
    all_example_tokens += [scorer._tokenizer.tokenize(example["premise"] + example["hypothesis"]) for example in generated_eval_data]


    new_examples = []  # Newly generated valid examples
    disagreed_examples = [] # Examples that doesn't reach argreement between generator and discriminator
    label_cnts = {label: 0 for label in labels}

    request_idx = 0

    progress_bar = tqdm.tqdm(total=num_examples_to_generate, desc="Collected Examples")

    while len(new_examples) < num_examples_to_generate:

        # to achieve uniform distribution among three labels, pick the label with smallest number of examples at each prompt
        picked_label = "Entailment"

        if (label_cnts["Neutral"] <= label_cnts["Entailment"]) and \
            (label_cnts["Neutral"] <= label_cnts["Contradiction"]):
            picked_label = "Neutral"

        if (label_cnts["Contradiction"] <= label_cnts["Entailment"]) and \
            (label_cnts["Contradiction"] <= label_cnts["Neutral"]):
            picked_label = "Contradiction"
        
        # only sampling from the seed examples with given label
        prompt_examples = random.sample(divided_seed_examples[picked_label],
                                            num_prompt_examples)
        prompt = generation_prompt(prompt_examples, num_genetated_examples_per_prompt, label=picked_label)
        # print("Prompt:\n" + prompt + "\n")
        
        prompt_args["temperature"] = 1.0
        prompt_args["messages"] = [{"role":"user", "content": prompt}]
        response = completions_with_backoff(**prompt_args)
        # print("Response:\n" + response + "\n")

        collected_examples = parse_response(response)
        # Only keep validated examples (those have valid premise/hypothesis/label, and not similar to any preceded examples)
        for example in collected_examples:
            
            if validate_example(example, scorer, all_example_tokens, prompt_args, disagreed_examples, num_cpus):
                new_examples += [example]
                label_cnts[picked_label] += 1
                all_example_tokens.append(scorer._tokenizer.tokenize(example["premise"] + example["hypothesis"]))
                progress_bar.update(1)

        request_idx += 1

    # Split training set and validation set with given train_validation_ratio
    eval_size = int(validation_ratio * len(new_examples))
    generated_train_data += new_examples[:-eval_size]
    generated_eval_data += new_examples[-eval_size:]

    #  Write back collected examples
    save_list_as_csv_files(output_path_train, generated_train_data)
    save_list_as_csv_files(output_path_eval, generated_eval_data)
    save_list_as_csv_files(output_path_disagreed, disagreed_examples)
    print(f"Saving {len(generated_train_data)} examples to training set {output_path_train}")
    print(f"Saving {len(generated_eval_data)} examples to validation set {output_path_eval}")
    

def main(task, **kwargs):
    globals()[task](**kwargs)

if __name__ == "__main__":
    fire.Fire(main)