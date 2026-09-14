"""
pipeline.py — MishraEval inference pipeline
Extracted from day9_inference_pipeline notebook (Day 10).

Self-contained: running `from pipeline import predict` loads everything
this needs (embedder, models, calibration values) at import time, in a
fixed order — no dependency on notebook cell execution order.
"""

import os
import pickle

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from model_defs import TopicClassifier, MishraEvalMLP, assign_risk, needs_handoff
from lang_detect import detect_language

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── topic_label_map ─────────────────────────────────────────────
# Load if saved on disk; otherwise reconstruct the same way LabelEncoder
# would have at training time (alphabetical .classes_ order).
if os.path.exists(os.path.join(PROJECT_ROOT, "models", "topic_label_map.pkl")):
    with open(os.path.join(PROJECT_ROOT, "models", "topic_label_map.pkl"), "rb") as f:
        topic_label_map = pickle.load(f)
else:
    from sklearn.preprocessing import LabelEncoder

    df_topics = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "dataset_v3b.csv"))
    le = LabelEncoder()
    le.fit(df_topics["user_topic"])
    topic_label_map = {i: cls for i, cls in enumerate(le.classes_)}
    print("Reconstructed topic_label_map:", topic_label_map)

# ── embedder ─────────────────────────────────────────────────────
embedder = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

# ── models ───────────────────────────────────────────────────────
user_topic_model = TopicClassifier()
user_topic_model.load_state_dict(torch.load(os.path.join(PROJECT_ROOT, "models", "user_topic_classifier.pth")))
user_topic_model.eval()

bot_topic_model = TopicClassifier()
bot_topic_model.load_state_dict(torch.load(os.path.join(PROJECT_ROOT, "models", "bot_topic_classifier.pth")))
bot_topic_model.eval()

main_model = MishraEvalMLP()
main_model.load_state_dict(torch.load(os.path.join(PROJECT_ROOT, "models", "mishraeval_best.pth")))
main_model.eval()

# ── calibration / label maps ────────────────────────────────────
with open(os.path.join(PROJECT_ROOT, "models", "label_map.pkl"), "rb") as f:
    label_map = pickle.load(f)

with open(os.path.join(PROJECT_ROOT, "models", "temperature.pkl"), "rb") as f:
    T = pickle.load(f)
# T = 2.60 confirmed correct (v3b calibration). 

with open(os.path.join(PROJECT_ROOT, "models", "handoff_threshold.pkl"), "rb") as f:
    handoff_threshold = pickle.load(f)

inv_label_map = {v: k for k, v in label_map.items()}


def predict(user_message, bot_response):
    # Step 1: detect language
    language = detect_language(user_message)

    # Step 2: embed both texts separately
    user_emb_raw = embedder.encode(user_message)
    bot_emb_raw = embedder.encode(bot_response)

    # Step 3: tensors + batch dim
    user_emb = torch.tensor(user_emb_raw, dtype=torch.float32).unsqueeze(0)
    bot_emb = torch.tensor(bot_emb_raw, dtype=torch.float32).unsqueeze(0)

    # Step 4: topic classifiers
    with torch.no_grad():
        user_topic_logits = user_topic_model(user_emb)
        bot_topic_logits = bot_topic_model(bot_emb)

    user_topic_pred = torch.argmax(user_topic_logits, dim=1).item()
    bot_topic_pred = torch.argmax(bot_topic_logits, dim=1).item()

    user_topic_label = topic_label_map[user_topic_pred]
    bot_topic_label = topic_label_map[bot_topic_pred]

    # Step 5: topic_match feature
    topic_match = int(user_topic_pred == bot_topic_pred)

    # Step 6: build the 1537-dim combined feature vector
    # order must match training: user_emb (768) + bot_emb (768) + topic_match (1)
    topic_match_tensor = torch.tensor([[topic_match]], dtype=torch.float32)
    combined = torch.cat([user_emb, bot_emb, topic_match_tensor], dim=1)

    # Step 7: run the main model -> raw logits
    with torch.no_grad():
        logits = main_model(combined)

    # Step 8: temperature scaling -- divide logits by T BEFORE softmax
    calibrated_logits = logits / T
    probs = torch.softmax(calibrated_logits, dim=1)

    predicted_class = torch.argmax(probs, dim=1).item()
    calibrated_confidence = probs[0, predicted_class].item()

    # Pass handoff_threshold explicitly rather than relying on model_defs.py's
    # hardcoded default (0.80) — this way handoff_threshold.pkl is the single
    # source of truth. If it's ever re-tuned, this call picks up the new value
    # automatically with no other code to update.
    risk_level = assign_risk(predicted_class, calibrated_confidence, handoff_threshold)
    handoff = needs_handoff(predicted_class, calibrated_confidence, handoff_threshold)

    predicted_label = inv_label_map[predicted_class]

    return {
        "label": predicted_label,
        "confidence": calibrated_confidence,
        "language": language,
        "topic_match": topic_match,
        "user_topic": user_topic_label,
        "bot_topic": bot_topic_label,
        "risk_level": risk_level,
        "needs_handoff": handoff,
    }