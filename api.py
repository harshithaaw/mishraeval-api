"""
api.py — FastAPI REST API for MishraEval (Day 11)

Imports predict() from pipeline.py rather than redefining any model
loading / inference logic here — same reuse principle as app.py (Day 10).

Run locally with:
    uvicorn api:app --reload

Then test at:
    http://127.0.0.1:8000/docs   (interactive Swagger UI, auto-generated)
"""

from fastapi import FastAPI
from pydantic import BaseModel

from pipeline import predict

app = FastAPI(
    title="MishraEval API",
    description="Detects intent mismatches between chatbot responses and user messages.",
)


class PredictRequest(BaseModel):
    """
    Defines the expected shape of incoming JSON. FastAPI uses this to
    validate requests automatically — a missing field or wrong type
    gets rejected with a clear 422 error before predict() ever runs.
    """
    user_message: str
    bot_response: str


@app.post("/predict")
def predict_endpoint(request: PredictRequest):
    """
    POST /predict
    Body: {"user_message": "...", "bot_response": "..."}
    Returns: predict()'s full result dict as JSON — FastAPI serializes
    a returned dict to JSON automatically, no manual conversion needed.
    """
    result = predict(request.user_message, request.bot_response)
    return result


@app.get("/")
def health_check():
    """Simple root endpoint to confirm the API is running at all —
    useful for a quick browser check or uptime monitoring later."""
    return {"status": "MishraEval API is running"}