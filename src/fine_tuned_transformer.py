import torch
from transformers import AutoModelForSequenceClassification
from pathlib import Path

from data import prepare_data
from transformer_utils import (create_dataloaders, train_one_epoch, evaluate, get_device)

MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 77
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "best_distilbert"

# the encoder weights are not frozen so that learning can be
def load_model():
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS
    )


def train():
    train_dataset, validation_dataset, _, tokenizer = prepare_data()

    train_loader, validation_loader = create_dataloaders(
        train_dataset,
        validation_dataset,
        tokenizer,
        BATCH_SIZE
    )

    model = load_model()
    device = get_device()
    model.to(device)

    # this will fine tune the encoder as well, not just the last classification layer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_validation_f1 = 0

    for epoch in range(NUM_EPOCHS):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device
        )

        validation_accuracy, validation_f1 = evaluate(
            model,
            validation_loader,
            device
        )

        print(f"Epoch: {epoch + 1}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Accuracy: {validation_accuracy:.4f}")
        print(f"Validation Macro F1: {validation_f1:.4f}")

        # save the model only if this epoch gives the best validation F1 so far
        if validation_f1 > best_validation_f1:
            best_validation_f1 = validation_f1

            model.save_pretrained(CHECKPOINT_DIR)
            tokenizer.save_pretrained(CHECKPOINT_DIR)

            print(
                f"Saved best checkpoint with Validation Macro F1: "
                f"{best_validation_f1:.4f}"
            )


if __name__ == "__main__":
    train()