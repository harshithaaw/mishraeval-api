import re
from fastapi import FastAPI
from pydantic import BaseModel
from gradio_client import Client

app = FastAPI(title="MishraEval API (HF Spaces wrapper)")
client = Client("harshithaaa06/mishraeval")

class PredictRequest(BaseModel):
    user_message: str
    bot_response: str

def extract_risk_text(html):
    match = re.search(r">([A-Z]+)<", html)
    return match.group(1) if match else html

@app.post("/predict")
def predict_endpoint(request: PredictRequest):
    result = client.predict(
        user_message=request.user_message,
        bot_response=request.bot_response,
        api_name="/run_prediction",
    )
    label, confidence_dict, language, risk_html, needs_handoff, user_topic, bot_topic = result
    top_confidence = confidence_dict["confidences"][0]["confidence"]
    return {
        "label": label,
        "confidence": top_confidence,
        "language": language,
        "risk_level": extract_risk_text(risk_html),
        "needs_handoff": needs_handoff,
        "user_topic": user_topic,
        "bot_topic": bot_topic,
    }

@app.get("/")
def health_check():
    return {"status": "MishraEval API is running (via HF Spaces backend)"}
