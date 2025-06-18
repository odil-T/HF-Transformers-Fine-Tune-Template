import datasets
import torch
from pathlib import Path
from datetime import datetime
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, get_scheduler
from safetensors.torch import save_model


MODELS_SAVE_FOLDER = "models"
CHECKPOINT = "bert-base-uncased"

NUM_EPOCHS = 3
BATCH_SIZE = 8
AGG_EVERY_N_BATCHES = 100

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
data_collator = DataCollatorWithPadding(tokenizer)
raw_datasets = datasets.load_dataset("glue", "mrpc")


def preprocess(sample):
    return tokenizer(sample["sentence1"], sample["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(preprocess, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_collator)
val_dataloader = DataLoader(tokenized_datasets["validation"], batch_size=BATCH_SIZE, collate_fn=data_collator)
test_dataloader = DataLoader(tokenized_datasets["test"], batch_size=BATCH_SIZE, collate_fn=data_collator)

NUM_TRAINING_STEPS = NUM_EPOCHS * len(train_dataloader)

model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT)
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=NUM_TRAINING_STEPS
)


def train_loop():
    """
    We usually calculate the mean training loss and some metric (e.g. accuracy) for every N number of batches.
    """

    model.train()
    progress_bar = tqdm(total=len(train_dataloader))

    running_loss = 0.
    running_n_correct_predictions = 0


    for i, batch in enumerate(train_dataloader, start=1):
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()

        outputs = model(**batch)

        loss = outputs.loss
        running_loss += loss.item()

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        running_n_correct_predictions += (predictions == batch["labels"]).type(torch.int).sum().item()

        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        progress_bar.update(1)

        if not i % AGG_EVERY_N_BATCHES:
            avg_loss = running_loss / AGG_EVERY_N_BATCHES
            running_loss = 0.

            accuracy = (running_n_correct_predictions / (AGG_EVERY_N_BATCHES * BATCH_SIZE)) * 100
            running_n_correct_predictions = 0

            print(f"Batch {i} | Training Loss: {avg_loss} | Accuracy: {accuracy:.2f}%")

    # Necessary to calculate average loss for any remainder batches (e.g. 63 batches left from 1463 when SHOW_EVERY_N_BATCHES = 100)
    remaining_n_batches = (len(train_dataloader) % AGG_EVERY_N_BATCHES)
    avg_loss = running_loss / remaining_n_batches
    accuracy = (running_n_correct_predictions / (remaining_n_batches * BATCH_SIZE)) * 100

    return avg_loss, accuracy


def eval_loop(dataloader):
    """
    We only calculate the mean validation loss and some metric (e.g. accuracy) over the whole epoch for the validation and test sets.
    No mean calculations are made for every N number of batches.

    Args:
        dataloader: This should be either a validation or a test dataloader.
    """

    model.eval()
    running_loss = 0.
    running_n_correct_predictions = 0

    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            running_loss += outputs.loss.item()
            running_n_correct_predictions += (predictions == batch["labels"]).type(torch.int).sum().item()


    avg_loss = running_loss / len(dataloader)
    accuracy = (running_n_correct_predictions / (len(dataloader) * BATCH_SIZE)) * 100

    return avg_loss, accuracy


Path(MODELS_SAVE_FOLDER).mkdir(exist_ok=True)

timestamp = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
best_val_loss = 1_000_000.

for epoch in range(1, NUM_EPOCHS+1):
    print(f"Epoch: {epoch}")

    avg_train_loss, train_accuracy = train_loop()
    print(f"Training Loss: {avg_train_loss} | Training Accuracy: {train_accuracy:.2f}%")

    avg_val_loss, val_accuracy = eval_loop(val_dataloader)
    print(f"Validation Loss: {avg_val_loss} | Validation Accuracy: {val_accuracy:.2f}%", end='\n\n')

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        model_path = f"{MODELS_SAVE_FOLDER}/model_bert_epoch{epoch}_{timestamp}"
        torch.save(model.state_dict(), model_path)

        # You may choose to save as safetensors
        # save_model(model, f"{model_path}.safetensors")

avg_test_loss, test_accuracy = eval_loop(test_dataloader)
print(f"Test Loss: {avg_test_loss} | Test Accuracy: {test_accuracy:.2f}%")
