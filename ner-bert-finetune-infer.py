"""
Use this file to fine-tune nad use a NER model. Specify the dataset, tokenizer, and model.
Usage example is given at the end.

Don't forget to include a HuggingFace token that has write permissions if you wish to upload the tokenizer and model to the HugginFace Hub.
"""

import os
import torch
import evaluate

from dotenv import load_dotenv
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

from torch.optim import AdamW
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    AutoModelForTokenClassification,
    get_scheduler
)

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
MODELS_SAVE_FOLDER = "models"
MODEL_CHECKPOINT = "bert-base-cased"

assert HF_TOKEN, "Please add a HuggingFace token that has repository write permissions to the .env file."

NUM_EPOCHS = 3
AGG_EVERY_N_BATCHES = 100

raw_dataset = load_dataset("conll2003", trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)


def align_labels_with_tokens(word_label_ids, word_ids):
    token_label_ids = []
    previous_word_id = -1

    for word_id in word_ids:
        if word_id is None:
            token_label_ids.append(-100)
        else:
            if previous_word_id == word_id and word_label_ids[word_id] not in (0, 8):
                token_label_ids.append(word_label_ids[word_id] + 1)
            else:
                token_label_ids.append(word_label_ids[word_id])

        previous_word_id = word_id

    return token_label_ids


def preprocess(examples):
    encoding = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    pertoken_ner_tags = []
    for i in range(len(examples["ner_tags"])):
        label_ids = examples["ner_tags"][i]
        word_ids = encoding.word_ids(i)
        pertoken_label_ids = align_labels_with_tokens(label_ids, word_ids)
        pertoken_ner_tags.append(pertoken_label_ids)

    encoding["labels"] = pertoken_ner_tags
    return encoding


tokenized_dataset = raw_dataset.map(
    preprocess,
    batched=True,
    remove_columns=raw_dataset["train"].column_names,
)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
label_names = raw_dataset["train"].features["ner_tags"].feature.names

train_dataloader = DataLoader(
    tokenized_dataset["train"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=8,
)

val_dataloader = DataLoader(
    tokenized_dataset["validation"],
    collate_fn=data_collator,
    batch_size=8,
)

test_dataloader = DataLoader(
    tokenized_dataset["test"],
    collate_fn=data_collator,
    batch_size=8,
)

id2label = {i: label for i, label in enumerate(label_names)}
label2id = {label: i for i, label in enumerate(label_names)}

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_CHECKPOINT,
    id2label=id2label,
    label2id=label2id,
)

NUM_TRAINING_STEPS = NUM_EPOCHS * len(train_dataloader)

optimizer = AdamW(model.parameters(), lr=2e-5)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=NUM_TRAINING_STEPS,
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model.to(device)


def seqeval_postprocess(predictions, labels):
    labels_batch = labels.detach().cpu().clone().numpy()
    predictions_batch = predictions.detach().cpu().clone().numpy()

    new_labels_batch = []
    for labels_sample in labels_batch:
        new_labels_sample = [label_names[l] for l in labels_sample if l != -100]
        new_labels_batch.append(new_labels_sample)

    new_predictions_batch = []
    for preds_sample, labels_sample in zip(predictions_batch, labels_batch):
        new_preds_sample = [label_names[p] for p, l in zip(preds_sample, labels_sample) if l != -100]
        new_predictions_batch.append(new_preds_sample)

    # Both of these are lists of lists of strings (the inner list is one sample)
    return new_predictions_batch, new_labels_batch


def train_loop():
    """
    We usually calculate the mean training loss and some metric (e.g. accuracy) for every N number of batches.
    """

    model.train()
    running_loss = 0.
    seqeval = evaluate.load("seqeval")
    progress_bar = tqdm(total=len(train_dataloader))

    for i, batch in enumerate(train_dataloader, start=1):
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()

        outputs = model(**batch)
        loss = outputs.loss
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        progress_bar.update(1)

        running_loss += loss.item()
        clean_predictions, clean_labels = seqeval_postprocess(predictions, batch["labels"])
        seqeval.add_batch(
            predictions=clean_predictions,
            references=clean_labels
        )

        if not i % AGG_EVERY_N_BATCHES:
            avg_loss = running_loss / AGG_EVERY_N_BATCHES
            running_loss = 0.

            metric_results = seqeval.compute()
            seqeval = evaluate.load("seqeval")

            print(f"Batch {i} | Training Metrics")
            print(
                "Loss: {} | Overall Precision: {} | Overall Recall: {} | Overall F1: {} | Overall Accuracy: {}".format(
                    avg_loss, metric_results["overall_precision"], metric_results["overall_recall"],
                    metric_results["overall_f1"], metric_results["overall_accuracy"]))

    # Computing loss and metrics for the remainder batches
    remaining_n_batches = (len(train_dataloader) % AGG_EVERY_N_BATCHES)
    avg_loss = running_loss / remaining_n_batches
    metric_results = seqeval.compute()

    return avg_loss, metric_results


def eval_loop(dataloader):
    """
    We only calculate the mean validation loss and some metric (e.g. accuracy) over the whole epoch for the validation and test sets.
    No mean calculations are made for every N number of batches.

    Args:
        dataloader: This should be either a validation or a test dataloader.
    """

    model.eval()
    running_loss = 0.
    seqeval = evaluate.load("seqeval")

    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            running_loss += outputs.loss.item()
            clean_predictions, clean_labels = seqeval_postprocess(predictions, batch["labels"])
            seqeval.add_batch(
                predictions=clean_predictions,
                references=clean_labels
            )

    avg_loss = running_loss / len(dataloader)
    metric_results = seqeval.compute()

    return avg_loss, metric_results


Path(MODELS_SAVE_FOLDER).mkdir(exist_ok=True)

timestamp = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
best_val_loss = 1_000_000.

for epoch in range(1, NUM_EPOCHS + 1):
    print(f"Epoch: {epoch}")

    avg_train_loss, train_metrics = train_loop()
    print("Training Metrics")
    print("Loss: {} | Overall Precision: {} | Overall Recall: {} | Overall F1: {} | Overall Accuracy: {}".format(
        avg_train_loss, train_metrics["overall_precision"], train_metrics["overall_recall"],
        train_metrics["overall_f1"], train_metrics["overall_accuracy"]))

    avg_val_loss, val_metrics = eval_loop(val_dataloader)
    print("Validation Metrics")
    print("Loss: {} | Overall Precision: {} | Overall Recall: {} | Overall F1: {} | Overall Accuracy: {}".format(
        avg_val_loss, val_metrics["overall_precision"], val_metrics["overall_recall"], val_metrics["overall_f1"],
        val_metrics["overall_accuracy"]))

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        model_path = f"{MODELS_SAVE_FOLDER}/model_bert_ner_epoch{epoch}_{timestamp}"
        torch.save(model.state_dict(), model_path)

        # You may choose to push the model to hub during training
        # tokenizer.push_to_hub("bert-finetuned-ner", use_auth_token=HF_TOKEN)
        # model.push_to_hub("bert-finetuned-ner", use_auth_token=HF_TOKEN)

        # You may choose to save as safetensors
        # save_model(model, f"{model_path}.safetensors")

avg_test_loss, test_metrics = eval_loop(test_dataloader)
print("Test Metrics")
print("Loss: {} | Overall Precision: {} | Overall Recall: {} | Overall F1: {} | Overall Accuracy: {}".format(
    avg_test_loss, test_metrics["overall_precision"], test_metrics["overall_recall"], test_metrics["overall_f1"],
    test_metrics["overall_accuracy"]))

# Usage
# from transformers import pipeline
#
# model_checkpoint = "odil111/bert-finetuned-ner"
# token_classifier = pipeline("token-classification", checkpoint=model_checkpoint, aggregation_strategy="simple")
#
# token_classifier("Hi, my name is Mike and I work at Microsoft in Washington.")