import os, logging
import numpy as np
import transformers
from args import ModelArguments, PredictingArguments
from dataset import NLIDatasetForPrediction
from transformers import TextClassificationPipeline
from tqdm import trange
import json


def predict():

    parser = transformers.HfArgumentParser((ModelArguments, PredictingArguments))
    model_args, predict_args = parser.parse_args_into_dataclasses()


    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        num_labels=3
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="right",
        use_fast=False,
    )

    label2id = model.config.label2id
    dataset_pred = NLIDatasetForPrediction(data_path=predict_args.test_data_path)

    # Use Transformers pipeline to do prediction on test dataset
    classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=predict_args.device)

    num_test_examples, bs = len(dataset_pred), predict_args.predict_batch_size
    num_batches = (num_test_examples + bs - 1) // bs
    labels = dataset_pred.get_labels()
    correct_count = 0

    for i in trange(num_batches, desc = "prediction"):
        batch_start, batch_end = i * bs, min((i + 1) * bs, num_test_examples) # The starting & ending index of batch

        input_sentences = [dataset_pred[id] for id in range(batch_start, batch_end)]
        batch_pred = classifier(input_sentences)
        predicted_labels = np.array([int(label2id[example['label']]) for example in batch_pred])
        gold_labels = labels[batch_start: batch_end]
        # print(predicted_labels, gold_labels)
        correct_count += (predicted_labels == gold_labels).sum()


    logging.info(f"Accuracy on test set {predict_args.test_data_path}: {correct_count / num_test_examples}")
    

    logging.info(f"Storing result to {predict_args.result_path}")

    # If the file for storing result doesn't exist, create and initialize it with an empty list
    if not os.path.exists(predict_args.result_path):

        with open (predict_args.result_path, 'w') as f:
            initialize_file_data = {'results':[]}
            json.dump(initialize_file_data, f, indent = 4)

    # Write new result to the json file 
    new_result = {'model':model_args.model_name_or_path, 
                     'test_dataset':predict_args.test_data_path,
                     'accuracy':str(correct_count / num_test_examples),
                     'loop':predict_args.loop_cnt}
    
    with open (predict_args.result_path, 'r+') as f:
        file_data = json.load(f)
        file_data["results"].append(new_result)
        f.seek(0)
        json.dump(file_data, f, indent = 4)

    logging.info(f"Finish Storing Prediction Result for Loop {predict_args.loop_cnt}.")




if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    predict()
