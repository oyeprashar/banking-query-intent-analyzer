# Banking Query Intent Analyzer

An end-to-end intent classification system for banking customer queries using the banking77 dataset.

The project compares a classical TF-IDF baseline, a frozen pretrained transformer, and a fully fine-tuned DistilBERT model across 77 banking intents. The final model is exposed through a FastAPI inference service with confidence-based routing for uncertain predictions.

## Architecture

```text
Customer Query
      ↓
DistilBERT Tokenizer
      ↓
Fine-tuned DistilBERT
      ↓
77 Intent Logits
      ↓
Softmax
      ↓
Intent + Confidence
      ↓
Confidence >= 0.90?
   ↙              ↘
Auto Route      Human Review
```

## Dataset

This project uses banking77, an intent classification dataset containing banking customer queries across 77 intent categories.
The official training split is further divided into training and validation sets using a stratified split. The official test set remains untouched until final evaluation.

## Models

### 1. TF-IDF + Logistic Regression 

A classical NLP baseline using TF-IDF features and multiclass Logistic Regression.

### 2. Frozen DistilBERT

A pretrained DistilBERT encoder with frozen transformer weights. Only the classification head is trained to map DistilBERT representations to the 77 banking intents.

### 3. Fine-tuned DistilBERT

The full DistilBERT model and classification head are trained on BANKING77 using PyTorch and AdamW.

## Results

| Model                       | Accuracy | Macro F1 |
|:----------------------------|:--------:|:--------:|
| TF-IDF + Logistic Regression | 85.3%   | 84.6%    |
| Frozen DistilBERT            | 72.6%   | 71.1%    |
| Fine-tuned DistilBERT        | **90.19%** | **90.04%** |

Fine-tuning the pretrained encoder improved performance substantially over the frozen transformer baseline and outperformed the classical TF-IDF model.

## Error Analysis

The fine-tuned model was evaluated using:

- Per-class precision, recall, and F1
- Confusion matrix
- Most frequently confused intent pairs
- High-confidence incorrect predictions

A major source of error is semantic overlap between closely related banking intents.

Examples include:

```text
verify_my_identity → why_verify_identity
balance_not_updated_after_bank_transfer → pending_transfer
card_arrival → card_delivery_estimate
fiat_currency_support → exchange_via_app
```

## Confidence-Based Routing

Instead of automatically routing every prediction, the system evaluates the maximum softmax confidence.

| Threshold | Auto-route Coverage | Auto-route Accuracy |
|:---------:|:-------------------:|:-------------------:|
| 0.50 | 92.89% | 93.99% |
| 0.60 | 87.99% | 95.61% |
| 0.70 | 81.20% | 97.24% |
| 0.80 | 72.92% | 98.09% |
| 0.90 | 54.71% | **99.41%** |
| 0.95 | 28.57% | 99.77% |

The system uses a **0.90 confidence threshold**, automatically routing approximately **54.7% of queries at 99.4% accuracy** while sending lower-confidence predictions for human review.

## Inference API

The fine-tuned model is exposed through FastAPI.

### Endpoint

```text
POST /predict
```

Example request:

```json
{
  "query": "Why hasn't my card arrived?"
}
```

Example response:

```json
{
  "intent": "card_arrival",
  "confidence": 0.97,
  "route": "automatic",
  "latency_ms": 35.2
}
```


## Running the API

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the inference server:

```bash
python -m uvicorn src.api:app --reload
```

Swagger documentation is available at `/docs`.