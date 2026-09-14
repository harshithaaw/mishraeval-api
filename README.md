# MishraEval API - FastAPI on Render

FastAPI REST API for MishraEval multilingual intent mismatch classifier.

## Deployment on Render

This repo is configured for deployment on Render.com as a FastAPI web service.

### Environment Setup

Render will automatically install dependencies from `requirements.txt`:
- fastapi
- uvicorn  
- torch
- sentence-transformers
- pandas
- scikit-learn

### Startup Command

Render should use:
```
uvicorn api:app --host 0.0.0.0 --port $PORT
```

### API Endpoints

- `GET /` - Health check
- `POST /predict` - Main prediction endpoint

### Example Request

```bash
curl -X POST "https://your-app.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "user_message": "Mera refund abhi tak nahi aaya",
    "bot_response": "Aapka refund 3-5 din mein process hoga"
  }'
```

### Response

```json
{
  "label": "INTENT_MISMATCH",
  "confidence": 0.9234,
  "language": "hinglish",
  "topic_match": 1,
  "user_topic": "refund",
  "bot_topic": "payment",
  "risk_level": "MEDIUM",
  "needs_handoff": false
}
```

## v3b Model Updates

This deployment uses the v3b model with:
- Dataset: dataset_v3b.csv (570 rows)
- Temperature: 2.60
- Handoff threshold: 0.93
- Overall F1: 0.9210
- Calibration: 34.8% ECE reduction