import logging
import os
from typing import Optional, Sequence, Union
from functools import partial
from multiprocessing import Pool
from rouge_score import rouge_scorer
import openai
import pandas as pd
import backoff


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def completions_with_backoff(api_key: str, **kwargs):
    openai.api_key = api_key
    while True:
        try:
            response = openai.ChatCompletion.create(**kwargs)
            break
        except openai.error.OpenAIError as e:
            logging.warning(f"OpenAIError: {e}.")
    
    return response['choices'][0]['message']['content']


def generation_prompt(examples, num_genetated_examples, label):
    """
    :param examples: A list of (premise, hypothesis, label) tuples
    :return: prompt: A string as prompt
    """
    num_prompt_examples = len(examples)
    prompt = "In an NLI task, you are given two sentences. The first sentence is called \'Premise\', while" \
                  " the second sentence is called \'Hypothesis\'. The label determines whether “Hypothesis” is " \
                  " true, false, or undetermined under the condition of “premise”. If the answer is true, label should be \'Entailment\';" \
                  "If the answer is false, label should be \'Contradiction\'; If the answer is undetermined, label should be \'Neutral\'."
    
    prompt += f"Now you are going to generate {num_prompt_examples + num_genetated_examples} example of NLI task with {label} as its label." \
                    "Each example should contain three lines, with the first line being a sentence as 'Premise', " \
                    "the second line being a sentence as 'Hypothesis', and the last line being a sentence as 'Label'." 

    str_labels = ['Entailment', 'Neutral', 'Contradiction']

    for i, example in enumerate(examples):
        prompt += f"{i+1}.\n" \
                  f"Premise:{example['premise']}\n" \
                  f"Hypothesis:{example['hypothesis']}\n" \
                  f"Label:{str_labels[example['label']]}\n"
    return prompt


def critique_prompt(example):
    prompt = "In an NLI task, you are given two sentences. The first sentence is called \'Premise\', while" \
                " the second sentence is called \'Hypothesis\'. The label determines whether “Hypothesis” is " \
                " true, false, or undetermined under the condition of “premise”. If the answer is true, label should be \'Entailment\';" \
                "If the answer is false, label should be \'Contradiction\'; If the answer is undetermined, label should be \'Neutral\'."
    prompt += f"Now you are given an NLI task example, with the \'Premise\' being \'{example['premise']}\', " \
                    f"and the \'Hypothesis\' being \'{example['hypothesis']}\'. "\
                     "Please give your prediction of label at the first line, and then briefly explain your answer at the second line. The length of explanation should not be exceeding 100 words." 
    return prompt


def parse_response(response: str) -> Sequence[dict]:
    """
    :param response: a string of response from gpt3/chatgpt
    :return: a list of examples int the form of {'premise':.., 'hypothesis':.., 'label':..}
             where label should be 0, 1 or 2
    """

    split_sentences = response.split('\n')
    label2id = {'Entailment': 0, 'Neutral': 1, 'Contradiction': 2}
    collected_examples = []

    i = 0
    while i < len(split_sentences):

        # Searching for the next example
        if (split_sentences[i].find('Premise') == -1) and \
            (split_sentences[i].find('premise') == -1):
            i += 1
            continue

        if (i + 2 >= len(split_sentences)):
            break

        # Assume the next three lines is in the form of 
        # Premise:...
        # Hypothesis:...
        # Label:...
        premise = split_sentences[i][split_sentences[i].find(':')+1:].strip('"')
        hypothesis = split_sentences[i+1][split_sentences[i+1].find(':')+1:].strip('"')
        label = split_sentences[i+2][split_sentences[i+2].find(':')+1:]
        label = label.strip(' .')
        i += 3   
        if label not in label2id.keys():
            continue
        collected_examples.append({"premise": premise, 
                                   "hypothesis": hypothesis,
                                   "label": label2id[label]})
        
    return collected_examples    


def validate_example(example: dict, scorer: rouge_scorer.RougeScorer, all_example_tokens: Sequence, 
                     prompt_args: dict, disagreed_examples: Sequence, num_cpus: int=4) -> bool:
    
    id2label = {0: 'Entailment', 1: 'Neutral', 2: 'Contradiction'}

    premise, hypothesis = example["premise"], example["hypothesis"]
    if (len(premise) == 0 or len(hypothesis) == 0):
        return False

    # computing similarity with the pre-tokenzied examples
    if (len(all_example_tokens) > 0):
        new_instruction_token = scorer._tokenizer.tokenize(premise + hypothesis)
        with Pool(num_cpus) as p:
            rouge_scores = p.map(
                partial(rouge_scorer._score_lcs, new_instruction_token),
                all_example_tokens,
            )
        rouge_scores = [score.fmeasure for score in rouge_scores]
        if max(rouge_scores) > 0.7: # There exists some simliar examples
            return False
    
    # Check correctness of example by prompting ChatGPT.
    # If ChatGPT doesn't return the same label as example provides, invalidate this example.
    prompt_for_checking_correctness = critique_prompt(example)
    prompt_args["temperature"] = 0.2
    prompt_args["messages"] = [{"role":"user", "content": prompt_for_checking_correctness}]
    response = completions_with_backoff(**prompt_args)
    answer_sentence = response.split('.')[0].split('\n')[0]
    if answer_sentence.find(id2label[example["label"]]) == -1:
        example["label"] = f"G:{id2label[example['label']]}/D:" + answer_sentence
        disagreed_examples.append(example)
        return False
    
    return True


# In this function, dataset is stored as a list of dict, 
# where each dict represents one example in the form of {"premise":.., "hypothesis":.., "label":..}.
def load_csv_file_as_list(file_path: str) -> Sequence[dict]:
    list_of_data = []
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        list_of_data += [
            {"premise": df.loc[id, "premise"], 
             "hypothesis": df.loc[id, "hypothesis"], 
             "label": df.loc[id, "label"]}
            for id in range(len(df))
        ]
    return list_of_data



def save_list_as_csv_files(file_path: str, list_of_data: Sequence[dict]):
    df = pd.DataFrame({"premise": [ex["premise"] for ex in list_of_data],
                       "hypothesis": [ex["hypothesis"] for ex in list_of_data],
                       "label": [ex["label"] for ex in list_of_data]})  
                       
    with open(file_path, 'w') as f_out:
        f_out.write(df.to_csv(index=False))
    