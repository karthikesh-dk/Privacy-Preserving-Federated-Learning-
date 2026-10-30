"""
Model Inference Service for EchoSafe.
Loads pre-trained federated CommentClassifierMLP and vocabulary vectorizer.
Zero C++ DLL dependencies (Pure Python / NumPy).
"""

import os
import sys
from typing import Dict, Any, List, Tuple
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.config import GLOBAL_MODEL_PATH
from modules.data_prep import CommentDataPreprocessor, LABEL_MAP, SEVERITY_MAP
from modules.local_model import CommentClassifierMLP
from modules.moderation_engine import mask_text_display

REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

class ModelInferenceService:
    """Singleton service wrapping the trained CommentClassifierMLP."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelInferenceService, cls).__new__(cls)
            cls._instance._init_model()
        return cls._instance

    def _init_model(self):
        print(" [ModelService] Initializing TF-IDF vectorizer and loading trained Federated model...")
        self.preprocessor = CommentDataPreprocessor()
        self.preprocessor.load_and_preprocess(test_user_ratio=0.20, test_template_ratio=0.20, seed=42)
        
        vocab_size = len(self.preprocessor.vectorizer.vocabulary_)
        self.model = CommentClassifierMLP(input_dim=vocab_size, hidden1=64, hidden2=32, num_classes=4, seed=42)
        
        if os.path.exists(GLOBAL_MODEL_PATH):
            data = np.load(GLOBAL_MODEL_PATH)
            weights = [data[f"arr_{i}"] for i in range(len(data.files))]
            self.model.set_weights(weights)
            print(f" [ModelService] Successfully loaded weights from: {GLOBAL_MODEL_PATH}")
        else:
            print(" [ModelService] Warning: Global model file not found, running with initial weights.")

        self.vocabulary = self.preprocessor.vectorizer.vocabulary_
        # Invert vocabulary for keyword contribution analysis
        self.inv_vocab = {idx: word for word, idx in self.vocabulary.items()}

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Classifies incoming comment text.
        Returns label, confidence, calibrated Softmax probabilities,
        abuse severity, masked text, and top contributing words.
        """
        # 1. Feature extraction via TF-IDF
        X = self.preprocessor.vectorizer.transform([text])
        
        # 2. Forward pass through neural network
        probs = self.model.forward(X)[0]  # [pos, neu, crit, abu]
        pred_idx = int(np.argmax(probs))
        pred_label = REVERSE_LABEL_MAP.get(pred_idx, "neutral")
        confidence = float(probs[pred_idx])
        abuse_prob = float(probs[3])

        # 3. Abuse severity estimation
        if pred_label == "abusive-vulgar":
            if abuse_prob > 0.85:
                severity = "severe"
            elif abuse_prob > 0.60:
                severity = "moderate"
            else:
                severity = "mild"
        else:
            severity = "none"

        # 4. Explainability: Top contributing words in text
        tokens = text.lower().split()
        matched_words = [w.strip(".,!?\"'()") for w in tokens if w.strip(".,!?\"'()") in self.vocabulary]
        top_words = list(dict.fromkeys(matched_words))[:4]

        return {
            "label": pred_label,
            "confidence": round(confidence, 4),
            "abuse_probability": round(abuse_prob, 4),
            "abuse_severity": severity,
            "probabilities": {
                "positive": round(float(probs[0]), 4),
                "neutral": round(float(probs[1]), 4),
                "criticism": round(float(probs[2]), 4),
                "abusive": round(float(probs[3]), 4)
            },
            "masked_text": mask_text_display(text),
            "top_keywords": ", ".join(top_words) if top_words else "general"
        }

model_service = ModelInferenceService()
