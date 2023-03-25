import os
import random
import pandas as pd
from prompt_utils import load_csv_file_as_list, save_list_as_csv_files
from prompt_utils import completions_with_backoff, parse_response, validate_example

import tqdm
from rouge_score import rouge_scorer
import fire

#TODO: similartity check using Rouge_Score; pick seed examples with low confidence

def generate_prompt(examples, num_genetated_examples, label):
    """
    :param examples: A list of (premise, hypothesis, label) tuples
    :return: prompt: A string as prompt
    """
    num_prompt_examples = len(examples)
    instruction = "In an NLI task, you are given two sentences. The first sentence is called \'Premise\', while" \
                  " the second sentence is called \'Hypothesis\'. The label determines whether “Hypothesis” is " \
                  " true, false, or undetermined under the condition of “premise”. If the answer is true, label should be \'Entailment\';" \
                  "If the answer is false, label should be \'Contradiction\'; If the answer is undetermined, label should be \'Neutral\'."
    
    instruction += f"Now you are going to generate {num_prompt_examples + num_genetated_examples} example of NLI task with {label} as its label." \
                  "Please generate in the following way:\n"

    prompt = str(instruction)
    str_labels = ['Entailment', 'Neutral', 'Contradiction']
    for i, example in enumerate(examples):
        prompt += f"{i+1}.\n" \
                  f"Premise:{example['premise']}\n" \
                  f"Hypothesis:{example['hypothesis']}\n" \
                  f"Label:{str_labels[example['label']]}\n"
    return prompt



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

    os.makedirs(output_dir, exist_ok=True)
    output_path_train = os.path.join(output_dir, "gen_train.csv") # Generated Training Set
    output_path_eval = os.path.join(output_dir, "gen_validation.csv") # Generated Validation Set

    seed_examples = load_csv_file_as_list(seed_dataset_path)
    print(f"Loaded {len(seed_examples)} seed examples")
    
    # Divide seed examples according to label
    labels = ['Entailment', 'Neutral', 'Contradiction']
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


    # How many examples have been newly generated
    new_examples = []

    # The counting of requests to openai
    request_idx = 0

    progress_bar = tqdm.tqdm(total=num_examples_to_generate, desc="Collected Examples")

    while len(new_examples) < num_examples_to_generate:

        # only sampling from the seed examples
        # to achieve uniform distribution among three labels, pick label[request_idx % 3] at each prompt
        picked_label = labels[request_idx % 3]
        prompt_examples = random.sample(divided_seed_examples[picked_label],
                                            num_prompt_examples)
        prompt = generate_prompt(prompt_examples, num_genetated_examples_per_prompt, label=picked_label)
        # print("Prompt:\n" + prompt + "\n")
    
        messages = [{"role":"user", "content": prompt}]
        response = completions_with_backoff(api_key=api_key, model=model_name, messages=messages, top_p=top_p, temperature=temperature, max_tokens=max_tokens)
        # print("Response:\n" + response + "\n")

        collected_examples = parse_response(response)

        # Only keep validated examples (those have valid premise/hypothesis/label, and not similar to any preceded examples)
        for example in collected_examples:

            if validate_example(example, scorer, all_example_tokens, num_cpus):
                new_examples += [example]
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
    print(f"Saving {len(generated_train_data)} examples to training set {output_path_train}")
    print(f"Saving {len(generated_eval_data)} examples to validation set {output_path_eval}")
    

def main(task, **kwargs):
    globals()[task](**kwargs)

if __name__ == "__main__":
    fire.Fire(main)



