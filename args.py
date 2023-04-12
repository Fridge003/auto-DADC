from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import transformers


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="roberta-large")
    num_labels: int = field(default=3)

@dataclass
class DataArguments:
    train_data_path: str = field(default="SNLI/train.csv")
    test_data_path: str = field(default="SNLI/test.csv")
    eval_data_path: str = field(default="SNLI/validation.csv")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512)
    early_stopping_patience: int = field(default=3)


@dataclass 
class PredictingArguments:
    test_data_path: str = field(default="SNLI/test.csv")
    predict_batch_size: int = field(default=64)
    result_path: str = field(default="./prediction_performance.json")
    device: int = field(default=-1)
    loop_cnt: int = field(default=0)
  