"""
model_defs.py
Shared PyTorch model class definitions for MishraEval.

Why this file exists:
Loading a .pth file only restores learned WEIGHTS, not the class
definition itself — the nn.Module subclass must be defined in code
before load_state_dict() can work. Keeping the definitions here,
in one place, means every notebook/script (Day 9 inference, Day 10
Gradio, Day 11 FastAPI, Day 12 Streamlit) imports the SAME class,
instead of copy-pasting it repeatedly and risking silent drift if
one copy gets edited and another doesn't.
"""

import torch.nn as nn


class TopicClassifier(nn.Module):
    """
    Used for BOTH user_topic_classifier.pth and bot_topic_classifier.pth.
    Same architecture, two different sets of trained weights —
    load this class twice, once per .pth file.
    Input: 768-dim sentence embedding. Output: 6 topic classes.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 6)   # 6 topic classes
        )

    def forward(self, x):
        return self.net(x)


class MishraEvalMLP(nn.Module):
    """
    The main classifier: predicts CORRECT vs INTENT_MISMATCH.
    Input: 1537-dim = 1536-dim combined embedding (user+bot) + 1
    topic_match feature. Output: 2 classes.

    NOTE: this was called MainClassifier at training/save time
    (mishraeval_best.pth), and re-defined ad hoc as MLP(1537) in a
    later notebook cell to reload it. Same architecture both times —
    consolidated here under one clear name.
    """
    def __init__(self, input_dim=1537):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.net(x)

def assign_risk(predicted_label, calibrated_confidence, low_conf_threshold=0.80):
    is_low_conf = calibrated_confidence < low_conf_threshold
    if predicted_label == 0 and is_low_conf:
        return "HIGH"
    elif predicted_label == 1:
        return "MEDIUM"
    else:
        return "LOW"

def needs_handoff(predicted_label, calibrated_confidence, low_conf_threshold=0.80):
    return calibrated_confidence < low_conf_threshold