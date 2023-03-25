import logging
from typing import Dict
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
import evaluate

class NLIDataset(Dataset):
    """Create an NLI Dataset for training."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(NLIDataset, self).__init__()
        
        logging.warning(f"Processing dataset from {data_path}")
        df = pd.read_csv(data_path)
        # df.dropna()

        # tokenize all the examples and pad them to the same length
        self.tokenized_examples = tokenizer(list(df['premise']), 
                                            list(df['hypothesis']), 
                                            padding="longest",
                                            max_length=tokenizer.model_max_length,
                                            truncation=True)
        
        # collect labels
        self.labels = list(df['label'])
        print(len(self.labels))
        print(len(self.tokenized_examples['input_ids'][0]))
        logging.warning(f"Finish processing dataset from {data_path}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        item = {key: torch.tensor(val[i]) for key, val in self.tokenized_examples.items()}
        item["labels"] = torch.tensor(self.labels[i])
        return item
    

class NLIDatasetForPrediction(Dataset):
    """Create an NLI Dataset for predicting."""

    def __init__(self, data_path: str):
        super(NLIDatasetForPrediction, self).__init__()

        logging.info(f"Processing testing dataset from {data_path}")
        df = pd.read_csv(data_path)
        # df.dropna()

        # Concatenate "premise" and "hypothesis" string for examples
        self.input_sentences = [df.loc[id, "premise"] + df.loc[id, "hypothesis"] for id in range(len(df))]
        
        # collect labels
        self.labels = list(df["label"])

        logging.info(f"Finish processing testing dataset from {data_path}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return self.input_sentences[i]

    def get_labels(self):
        return np.array(self.labels)



def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def prepare_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Prepare the training/evalutation datasets for finetuning"""
    train_dataset = NLIDataset(tokenizer=tokenizer, data_path=data_args.train_data_path)
    eval_dataset = NLIDataset(tokenizer=tokenizer, data_path=data_args.eval_data_path)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, compute_metrics=compute_metrics)

