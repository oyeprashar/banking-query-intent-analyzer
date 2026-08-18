"""
We will evaluate the fine-tuned transformer over the unseen test data


Goals:

    1. Load test dataset <-- we still need to prepare this by tokenization
    2. Load best fine-tuned checkpoint
    3. Run inference
    4. Compute Accuracy + Macro F1
    5. Compute per-class metrics
    6. Build confusion matrix
    7. Capture prediction confidence
"""

import torch

from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from data import load_banking77, tokenize_dataset
from transformer_utils import get_device
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "best_distilbert"

BATCH_SIZE = 16


def prepare_test_dataloader():
    _, test_dataset = load_banking77()
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
    test_dataset = tokenize_dataset(test_dataset,tokenizer)
    raw_texts = test_dataset["text"]
    test_dataset = test_dataset.remove_columns(["text"])
    test_dataset = test_dataset.rename_column("label","labels")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=data_collator)
    return test_loader, raw_texts, tokenizer


# use the weights we learned during fine-tuning our model
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_DIR)
    return model



# We use the test dataset to evaluate the predictions made by our fine-tuned transformer
def predict(model, dataloader, device):
    model.eval()
    all_predictions = []
    all_labels = []
    all_confidences = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits

            # Convert 77 logits into probabilities
            probabilities = torch.softmax(
                logits,
                dim=1
            )

            # confidence means the highest probability out of 77 labels
            # prediction is the index of highest probability label
            confidences, predictions = torch.max(probabilities, dim=1)
            all_predictions.extend(predictions.cpu().tolist())
            all_confidences.extend(confidences.cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())

    return all_predictions,all_labels, all_confidences

def evaluate_predictions(
    labels,
    predictions,
    label_names
):
    """
    Compute overall and per-class metrics.
    """

    accuracy = accuracy_score(
        labels,
        predictions
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro"
    )

    print("\nFinal Test Results")
    print("------------------")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Macro F1: {macro_f1:.4f}")

    print("\nPer-class Metrics")
    print("-----------------")

    print(
        classification_report(
            labels,
            predictions,
            target_names=label_names,
            digits=4
        )
    )

    matrix = confusion_matrix(
        labels,
        predictions
    )

    return accuracy, macro_f1, matrix


def find_misclassified_examples(
    texts,
    labels,
    predictions,
    confidences,
    label_names,
    limit=20
):
    """
    Return highest-confidence incorrect predictions.

    These are especially useful because the model was confidently wrong.
    """

    mistakes = []

    for text, true_label, predicted_label, confidence in zip(
        texts,
        labels,
        predictions,
        confidences
    ):

        if true_label != predicted_label:

            mistakes.append(
                {
                    "text": text,
                    "true_intent": label_names[true_label],
                    "predicted_intent": label_names[predicted_label],
                    "confidence": confidence,
                }
            )

    mistakes.sort(
        key=lambda mistake: mistake["confidence"],
        reverse=True
    )

    return mistakes[:limit]


def find_most_confused_pairs(
    confusion_matrix_values,
    label_names,
    limit=10
):
    """
    Find intent pairs that the model confuses most often.
    """

    confused_pairs = []

    num_classes = len(label_names)

    for true_index in range(num_classes):

        for predicted_index in range(num_classes):

            if true_index == predicted_index:
                continue

            count = confusion_matrix_values[
                true_index
            ][
                predicted_index
            ]

            if count > 0:

                confused_pairs.append(
                    {
                        "true_intent": label_names[true_index],
                        "predicted_intent": label_names[predicted_index],
                        "count": int(count),
                    }
                )

    confused_pairs.sort(
        key=lambda pair: pair["count"],
        reverse=True
    )

    return confused_pairs[:limit]


def evaluate_confidence_thresholds(
    labels,
    predictions,
    confidences
):
    """
    Evaluate different confidence thresholds.

    Coverage:
        Percentage of total queries that would be automatically routed.

    Auto-route accuracy:
        Accuracy only on queries whose confidence is greater than or
        equal to the threshold.
    """

    thresholds = [
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        0.95
    ]

    print("\nConfidence Threshold Analysis")
    print("-----------------------------")

    for threshold in thresholds:

        selected_indices = [
            index
            for index, confidence in enumerate(confidences)
            if confidence >= threshold
        ]

        if len(selected_indices) == 0:
            coverage = 0
            auto_route_accuracy = 0

        else:
            selected_predictions = [
                predictions[index]
                for index in selected_indices
            ]

            selected_labels = [
                labels[index]
                for index in selected_indices
            ]

            coverage = (
                len(selected_indices)
                / len(labels)
            )

            auto_route_accuracy = accuracy_score(
                selected_labels,
                selected_predictions
            )

        print(
            f"Threshold: {threshold:.2f} | "
            f"Coverage: {coverage:.4f} | "
            f"Auto-route Accuracy: {auto_route_accuracy:.4f}"
        )


def main():
    print("CHECKPOINT_DIR :::: :",CHECKPOINT_DIR )

    # --------------------------------
    # Prepare official test split
    # --------------------------------

    test_loader, raw_texts, _ = prepare_test_dataloader()

    # --------------------------------
    # Load fine-tuned model
    # --------------------------------

    model = load_model()

    device = get_device()

    model.to(device)

    print("Using device:", device)

    # --------------------------------
    # Run inference
    # --------------------------------

    predictions, labels, confidences = predict(
        model,
        test_loader,
        device
    )

    # --------------------------------
    # Get human-readable intent names
    # --------------------------------

    _, test_dataset = load_banking77()

    label_names = test_dataset.features["label"].names

    # --------------------------------
    # Evaluate
    # --------------------------------

    accuracy, macro_f1, matrix = evaluate_predictions(
        labels,
        predictions,
        label_names
    )

    # --------------------------------
    # Confidence threshold analysis
    # --------------------------------

    evaluate_confidence_thresholds(
        labels,
        predictions,
        confidences
    )

    # --------------------------------
    # Most confused intent pairs
    # --------------------------------

    confused_pairs = find_most_confused_pairs(
        matrix,
        label_names
    )

    print("\nMost Confused Intent Pairs")
    print("--------------------------")

    for pair in confused_pairs:

        print(
            f"{pair['true_intent']} "
            f"→ {pair['predicted_intent']} "
            f"({pair['count']} times)"
        )

    # --------------------------------
    # High-confidence mistakes
    # --------------------------------

    mistakes = find_misclassified_examples(
        raw_texts,
        labels,
        predictions,
        confidences,
        label_names
    )

    print("\nHigh-confidence Mistakes")
    print("------------------------")

    for mistake in mistakes:

        print("Query:", mistake["text"])
        print("True:", mistake["true_intent"])
        print("Predicted:", mistake["predicted_intent"])
        print(
            "Confidence:",
            f"{mistake['confidence']:.4f}"
        )
        print("-" * 60)


if __name__ == "__main__":
    main()