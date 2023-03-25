import os, logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import transformers
from transformers import Trainer, EarlyStoppingCallback
from dataset import prepare_data_module
from args import ModelArguments, DataArguments, TrainingArguments

def train():

    local_rank = int(os.environ["LOCAL_RANK"])
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    id2label = {"0": "Entailment", "1": "Neutral", "2": "Contradiction"}
    label2id = {"Entailment": "0", "Neutral": "1", "Contradiction": "2"}
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        num_labels=3,
        id2label=id2label,
        label2id=label2id
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    training_data_module = prepare_data_module(tokenizer=tokenizer, data_args=data_args)

    trainer = Trainer(model=model, 
                    tokenizer=tokenizer, 
                    args=training_args, 
                    callbacks=[EarlyStoppingCallback(training_args.early_stopping_patience)],
                    **training_data_module)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()
