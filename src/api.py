from pathlib import Path
from time import perf_counter
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.data import load_banking77
from src.transformer_utils import get_device

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "best_distilbert"
CONFIDENCE_THRESHOLD = 0.90

app = FastAPI(
    title="Banking Query Intent Analyzer"
)

class PredictionRequest(BaseModel):
    query: str

class PredictionResponse(BaseModel):
    intent: str
    confidence: float
    route: str
    latency_ms: float


# Load once when the API starts
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_DIR)
device = get_device()
model.to(device)
model.eval()

_, test_dataset = load_banking77()
label_names = test_dataset.features["label"].names


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(request: PredictionRequest):

    start_time = perf_counter()
    inputs = tokenizer(request.query, truncation=True, return_tensors="pt")
    inputs = { key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

        confidence, prediction = torch.max(probabilities, dim=1)

    predicted_label = prediction.item()
    confidence_score = confidence.item()
    intent = label_names[predicted_label]

    if confidence_score >= CONFIDENCE_THRESHOLD:
        route = "automatic"
    else:
        route = "human_review"

    latency_ms = (
        perf_counter() - start_time
    ) * 1000

    return PredictionResponse(
        intent=intent,
        confidence=confidence_score,
        route=route,
        latency_ms=latency_ms
    )
