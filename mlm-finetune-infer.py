"""
You can choose to either use a whole word masking data collator (wwm_data_collator), or the token masking data collator (DataCollatorForLanguageModeling).
"""

import os
import math
import torch
import numpy as np

from dotenv import load_dotenv
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from torch.optim import AdamW
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    default_data_collator,
    DataCollatorForLanguageModeling,
    AutoModelForMaskedLM,
    get_scheduler
)


load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
MODELS_SAVE_FOLDER = "models"
MODEL_CHECKPOINT = "distilbert-base-uncased"

assert HF_TOKEN, "Please add a HuggingFace token that has repository write permissions to the .env file."

NUM_EPOCHS = 3
CHUNK_SIZE = 128
BATCH_SIZE = 32
AGG_EVERY_N_BATCHES = 50
MLM_PROBABILITY = 0.15

imdb_dataset = load_dataset("imdb")

tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)


def preprocess(examples):
    inputs = tokenizer(examples["text"])
    if tokenizer.is_fast:
        inputs["word_ids"] = [inputs.word_ids(i) for i in range(len(inputs.input_ids))]
    return inputs


def create_chunks(examples):
    concatenated_examples = {
        k: sum(v, []) for k, v in examples.items()
    }

    total_sample_length = len(concatenated_examples[list(examples.keys())[0]])
    total_sample_length = (total_sample_length // CHUNK_SIZE) * CHUNK_SIZE

    chunks = {
        k: [v[i:i + CHUNK_SIZE] for i in range(0, total_sample_length, CHUNK_SIZE)] for k, v in concatenated_examples.items()
    }

    chunks["labels"] = chunks["input_ids"].copy()

    return chunks


tokenized_dataset = imdb_dataset.map(preprocess, batched=True, remove_columns=["text", "label"])
chunked_dataset = tokenized_dataset.map(create_chunks, batched=True)
downsampled_dataset = chunked_dataset["train"].train_test_split(
    train_size=10000, test_size=1000, seed=42
)

token_masking_data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm_probability=MLM_PROBABILITY
)


def wwm_data_collator(features):
    """
    A data collator inputs the features in the form of a list with dictionaries. Each dictionary corresponds to one sample.
    In this case, one sample represents one chunk.

    A data collator should output a batch in the form of a dictionary containing tensors of shape (batch_size, chunk_size).
    """

    batch = {k: [] for k in features[0].keys() if k != "word_ids"}

    for sample in features:
        word2token_mapping = defaultdict(list)
        word_ids = sample.pop("word_ids")

        current_word_idx = -1
        current_word_id = None

        for token_idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if current_word_id != word_id:
                    current_word_idx += 1
                    current_word_id = word_id
                word2token_mapping[current_word_idx].append(token_idx)

        word_ids_mask = np.random.binomial(1, MLM_PROBABILITY, len(word2token_mapping))
        new_labels = [-100] * len(sample["labels"])

        for word_id in np.where(word_ids_mask)[0]:
            for token_idx in word2token_mapping[word_id]:
                new_labels[token_idx] = sample["labels"][token_idx]
                sample["input_ids"][token_idx] = tokenizer.mask_token_id

        sample["labels"] = new_labels

        for k in sample:
            batch[k].append(sample[k])

    batch = {k: torch.tensor(v) for k, v in batch.items()}

    return batch


def insert_random_mask(batch):
    features = [{k: batch[k][i] for k in batch.keys()} for i in range(len(batch["input_ids"]))]
    masked_batch = wwm_data_collator(features)
    return {f"masked_{k}": v for k, v in masked_batch.items()}


remove_columns = downsampled_dataset["test"].column_names
remove_columns.remove("word_ids")

train_dataset = downsampled_dataset["train"].map(
    insert_random_mask,
    batched=True,
    remove_columns=remove_columns
)

val_dataset = downsampled_dataset["test"].map(
    insert_random_mask,
    batched=True,
    remove_columns=remove_columns
)

train_dataset = train_dataset.rename_columns(
    {
        "masked_input_ids": "input_ids",
        "masked_attention_mask": "attention_mask",
        "masked_labels": "labels",
    }
)

val_dataset = val_dataset.rename_columns(
    {
        "masked_input_ids": "input_ids",
        "masked_attention_mask": "attention_mask",
        "masked_labels": "labels",
    }
)

train_dataloader = DataLoader(
    train_dataset,
    shuffle=True,
    batch_size=BATCH_SIZE,
    collate_fn=default_data_collator,
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=default_data_collator,
)

model = AutoModelForMaskedLM.from_pretrained(MODEL_CHECKPOINT)

NUM_TRAINING_STEPS = NUM_EPOCHS * len(train_dataloader)

optimizer = AdamW(model.parameters(), lr=5e-5)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=NUM_TRAINING_STEPS,
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model.to(device)


def calculate_perplexity(mean_loss):
    try:
        perplexity = math.exp(mean_loss)
    except OverflowError:
        perplexity = float("inf")
    return perplexity


def train_loop():
    """
    We usually calculate the mean training loss and some metric (e.g. accuracy) for every N number of batches.
    """

    model.train()
    running_loss = 0.
    progress_bar = tqdm(total=len(train_dataloader))

    for i, batch in enumerate(train_dataloader, start=1):
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()

        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        progress_bar.update(1)

        running_loss += loss.item()

        if not i % AGG_EVERY_N_BATCHES:
            avg_loss = running_loss / AGG_EVERY_N_BATCHES
            running_loss = 0.

            perplexity = calculate_perplexity(avg_loss)

            print(f"Batch {i} | Training Metrics")
            print("Perplexity: {}".format(perplexity))

    # Computing loss and metrics for the remainder batches
    remaining_n_batches = (len(train_dataloader) % AGG_EVERY_N_BATCHES)
    avg_loss = running_loss / remaining_n_batches
    perplexity = calculate_perplexity(avg_loss)

    return perplexity


def eval_loop(dataloader):
    """
    We only calculate the mean validation loss and some metric (e.g. accuracy) over the whole epoch for the validation and test sets.
    No mean calculations are made for every N number of batches.

    Args:
        dataloader: This should be either a validation or a test dataloader.
    """

    model.eval()
    running_loss = 0.

    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)

            running_loss += outputs.loss.item()

    avg_loss = running_loss / len(dataloader)
    perplexity = calculate_perplexity(avg_loss)

    return perplexity


Path(MODELS_SAVE_FOLDER).mkdir(exist_ok=True)

timestamp = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
best_val_perplexity = 1_000_000_000_000.

for epoch in range(1, NUM_EPOCHS + 1):
    print(f"Epoch: {epoch}")

    train_perplexity = train_loop()
    print("Training Metrics")
    print("Perplexity: {}".format(train_perplexity))

    val_perplexity = eval_loop(val_dataloader)
    print("Validation Metrics")
    print("Perplexity: {}".format(val_perplexity))

    if val_perplexity < best_val_perplexity:
        best_val_perplexity = val_perplexity
        model_path = f"{MODELS_SAVE_FOLDER}/model_distilbert_mlm_epoch{epoch}_{timestamp}"
        model.save_pretrained(model_path)

        # You may choose to push the model to hub during training
        # tokenizer.push_to_hub("distilbert-finetuned-mlm", use_auth_token=HF_TOKEN)
        # model.push_to_hub("distilbert-finetuned-mlm", use_auth_token=HF_TOKEN)

        # You may choose to save the model as a PyTorch state dictionary. Note that the model architecture is not saved, just the weights.
        # torch.save(model.state_dict(), model_path)


# USAGE -------------
#
# from transformers import pipeline
#
# FOR SAVED MODEL AND TOKENIZER STORED IN HF HUB
# model_checkpoint = "odil111/distilbert-finetuned-mlm"
# mask_filler = pipeline("fill-mask", checkpoint=model_checkpoint)
#
# FOR SAVED MODEL AND TOKENIZER STORED LOCALLY (the MODEL_PATH can be the model saved with `model.save_pretrained(MODEL_PATH)`)
# mask_filler = pipeline("fill-mask", model=MODEL_PATH, tokenizer="distilbert-base-uncased")
# mask_filler("Hi, my name is Mike and I work at Microsoft in [MASK].")
#
# You can load the model after saving with `model.save_pretrained(PATH)` as:
# model = AutoModelForMaskedLM.from_pretrained(PATH)