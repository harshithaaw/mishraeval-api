"""
api.py — FastAPI REST API for MishraEval (Day 11)

THIN WRAPPER: does not load any model locally. Calls the already-running
HF Spaces Gradio app (which holds the actual model, embedder, and
calibration files) over the network via gradio_client. This keeps this
service's memory footprint small enough for Render's free 512MB tier —
loading sentence-transformers/torch/model weights directly here caused
an out-of-memory deploy failure.

Run locally with:
    uvicorn api:app --reload

Then test at:
    http://127.0.0.1:8000/docs   (interactive Swagger UI, auto-generated)
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from gradio_client import Client

app = FastAPI(
    title="MishraEval API",
    description="Detects intent mismatches between chatbot responses and user messages.",
)
HF_SPACE_ID = "harshithaaa06/mishraeval"
client = Client(HF_SPACE_ID)


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

    Calls the HF Space's /predict endpoint via gradio_client and
    reshapes its tuple response (matching app.py's outputs= order:
    label, confidence, language, risk_level, needs_handoff,
    user_topic, bot_topic) into a named JSON dict.
    """
    try:
        result = client.predict(
            request.user_message,
            request.bot_response,
            api_name="/run_prediction",
        )
    except Exception as e:
        # Surface HF Space errors (e.g. Space asleep/cold-starting, or
        # a mismatched api_name/signature) as a clear 502, not a silent
        # crash or a misleading 500 with no context.
        raise HTTPException(status_code=502, detail=f"HF Space call failed: {e}")

    return {
        "label": result[0],
        "confidence": result[1],
        "language": result[2],
        "risk_level": result[3],
        "needs_handoff": result[4],
        "user_topic": result[5],
        "bot_topic": result[6],
    }


@app.get("/")
def health_check():
    """Simple root endpoint to confirm the API is running at all —
    useful for a quick browser check or uptime monitoring later."""
    return {"status": "MishraEval API is running"}
