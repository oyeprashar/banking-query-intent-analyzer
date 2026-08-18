import torch
from transformers import AutoModelForSequenceClassification

from data import prepare_data
from transformer_utils import (create_dataloaders, train_one_epoch, evaluate, get_device)


MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 77
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
NUM_EPOCHS = 5


def load_frozen_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS
    )

    for parameter in model.distilbert.parameters():
        parameter.requires_grad = False

    return model


def train():
    train_dataset, validation_dataset, _, tokenizer = prepare_data()

    train_loader, validation_loader = create_dataloaders(train_dataset, validation_dataset, tokenizer, BATCH_SIZE)
    model = load_frozen_model() # encoder weights are frozen
    device = get_device()
    model.to(device)
    optimizer = torch.optim.AdamW(filter(lambda parameter: parameter.requires_grad, model.parameters()),lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):

        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        validation_accuracy, validation_f1 = evaluate(model, validation_loader, device)

        print(f"Epoch: {epoch + 1}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Accuracy: {validation_accuracy:.4f}")
        print(f"Validation Macro F1: {validation_f1:.4f}")


if __name__ == "__main__":
    train()