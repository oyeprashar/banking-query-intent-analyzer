"""
    Input text
       ↓
    Tokenizer
       ↓
    DistilBERT encoder <------- We freeze the weights here so zero learning
       ↓
    Contextual representation of the text
       ↓
    Classification head <---- This still learns how to map the output of the transformer into the 77 intents
       ↓
    77 intent scores
"""

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, f1_score

from data import prepare_data


MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 77
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
NUM_EPOCHS = 5


# returns objects of dataloader for both training and validation. These are helpful for feeding batches
# when training the model
def create_dataloaders(train_dataset, validation_dataset, tokenizer,batch_size=BATCH_SIZE):


    # the tokenized data can differ in dimension, which we do not want.
    # Padding makes them of same dimension after getting padded with zeroes
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        return_tensors="pt"
    )

    # the dataset still contains the raw human redable text which we do not need
    train_dataset = train_dataset.remove_columns(["text"])
    validation_dataset = validation_dataset.remove_columns(["text"])


    # hugging face's model expects labels not not label
    train_dataset = train_dataset.rename_column("label", "labels")
    validation_dataset = validation_dataset.rename_column("label", "labels")


    # dataloaders are basically used to smartly feed data in batches to the model we are training
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)

    return train_loader, validation_loader


"""
Frozen == weights are not updated
"""
def load_frozen_model():

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS
    )

    # Freeze all DistilBERT encoder parameters
    for parameter in model.distilbert.parameters():
        parameter.requires_grad = False # mark it as no gradient descent is needed

    return model


def train_one_epoch(model, dataloader, optimizer, device):

    model.train()
    total_loss = 0

    for batch in dataloader:
        batch = {key: value.to(device)
            for key, value in batch.items()
        }

        optimizer.zero_grad() # clear the old weights
        outputs = model(**batch) # forward pass
        loss = outputs.loss
        loss.backward() # backpropogation
        optimizer.step() # update the weights
        total_loss += loss.item() # add the loss to the total lostt

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            outputs = model(**batch)

            predictions = torch.argmax(
                outputs.logits,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                batch["labels"].cpu().tolist()
            )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro"
    )

    return accuracy, macro_f1


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def train():

    train_dataset, validation_dataset, _, tokenizer = prepare_data()
    train_loader, validation_loader = create_dataloaders(train_dataset,validation_dataset,tokenizer)

    # returns a model that will not learn over the data as froze the weights for the encoder
    model = load_frozen_model()

    # Pytorch needs to understand the GPU it is training over
    device = get_device()
    model.to(device) # import the model's weight to the device

    # Only train parameters where requires_grad=True <--- frozen encoder weights are never trained
    # so DistilBERT is not learning how to be better at predictions, rather it learns how to convert its output into
    # the 77 expected intent. That
    # s it!

    # optimizer is the thing that works on the weights
    optimizer = torch.optim.AdamW(
        filter(
            lambda parameter: parameter.requires_grad,
            model.parameters()
        ),
        lr=LEARNING_RATE
    )


    for epoch in range(NUM_EPOCHS):

        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        validation_accuracy, validation_f1 = evaluate(model, validation_loader, device)

        print(f"Epoch: {epoch + 1}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Accuracy: {validation_accuracy:.4f}")
        print(f"Validation Macro F1: {validation_f1:.4f}")


if __name__ == "__main__":
    train()