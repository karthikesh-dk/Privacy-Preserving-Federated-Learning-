"""
Module 8: Automated Community Moderation Engine.

Integrates:
1. Federated CommentClassifierMLP inference (Positive, Neutral, Criticism, Abusive).
2. Historical User Behaviour Risk Profiling.
3. Multi-tiered enforcement decision rules (Approve, Review, Slow-Mode, Quarantine, Mute, Block).
4. Masked text display to safeguard operator and creator visual wellbeing.
"""

from typing import Dict, List, Any, Optional
import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.data_prep import CommentDataPreprocessor, LABEL_MAP
from modules.local_model import CommentClassifierMLP

REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

def mask_text_display(text: str) -> str:
    """Mask vulgar words for safe UI rendering."""
    masked = text
    for w in ["fuck", "shit", "bitch", "bastard", "asshole", "idiot", "loser", "dumb"]:
        masked = masked.replace(w, w[0] + "***")
        masked = masked.replace(w.capitalize(), w[0].capitalize() + "***")
    return masked

class ModerationEngine:
    """
    Evaluates incoming comments in real-time by combining local federated model
    inference with author behavioral history to recommend proportionate moderation actions.
    """
    def __init__(
        self,
        model_weights_path: str = "experiments/results/global_fedavg_model.npz",
        profiles_path: str = "experiments/results/user_behaviour_profiles.csv"
    ):
        self.model_weights_path = model_weights_path
        self.profiles_path = profiles_path
        self.preprocessor = CommentDataPreprocessor()
        self.model: Optional[CommentClassifierMLP] = None
        self.profiles_df: Optional[pd.DataFrame] = None
        self._initialize()

    def _initialize(self):
        """Load vocabulary vectorizer, global model parameters, and user profiles."""
        if os.path.exists(self.model_weights_path):
            self.preprocessor.load_and_preprocess()
            vocab_dim = len(self.preprocessor.vectorizer.vocabulary_)
            self.model = CommentClassifierMLP(input_dim=vocab_dim, hidden1=64, hidden2=32, num_classes=4)
            
            data = np.load(self.model_weights_path)
            weights = [data[f"arr_{i}"] for i in range(len(data.files))]
            self.model.set_weights(weights)

        if os.path.exists(self.profiles_path):
            self.profiles_df = pd.read_csv(self.profiles_path).set_index("user_id")

    def evaluate_comment(
        self,
        text: str,
        user_id: str,
        influencer_id: str = "inf_0"
    ) -> Dict[str, Any]:
        """
        Evaluate an incoming comment and return automated moderation verdict.
        """
        if self.model is None or self.preprocessor.vectorizer is None:
            self._initialize()

        # 1. Feature extraction
        X = self.preprocessor.vectorizer.transform([text])
        probs = self.model.forward(X)[0]  # [pos, neu, crit, abu]
        pred_idx = int(np.argmax(probs))
        pred_label = REVERSE_LABEL_MAP.get(pred_idx, "unknown")
        abuse_prob = float(probs[3])

        # 2. Lookup author behavioral risk profile
        user_risk = 0.0
        repeat_offences = 0
        archetype = "unknown"

        if self.profiles_df is not None and user_id in self.profiles_df.index:
            user_row = self.profiles_df.loc[user_id]
            user_risk = float(user_row.get("user_risk_score", 0.0))
            repeat_offences = int(user_row.get("repeat_offences", 0))
            archetype = str(user_row.get("predicted_archetype", "normal_positive"))

        # 3. Determine Tiered Moderation Decision
        if abuse_prob >= 0.70 or pred_label == "abusive-vulgar":
            if user_risk >= 0.75 or repeat_offences >= 4:
                verdict = "BLOCKED"
                action = "permanent_block_referral"
                rationale = f"Severe toxicity ({abuse_prob*100:.1f}%) from repeat chronic abuser ({repeat_offences} prior offences)."
            elif user_risk >= 0.50 or repeat_offences >= 2:
                verdict = "MUTED"
                action = "temporary_mute_48h"
                rationale = f"Abusive content detected ({abuse_prob*100:.1f}%). Author has history of repeated infractions."
            elif archetype == "burst_spammer":
                verdict = "RATE_LIMITED"
                action = "rate_limit_cooldown_15m"
                rationale = "Rapid burst spam detected. Comment hidden and user placed on slow mode cooldown."
            else:
                verdict = "AUTO_QUARANTINED"
                action = "shadow_filter_pending_review"
                rationale = f"Hostile language detected ({abuse_prob*100:.1f}%). Placed in creator quarantine queue."
        elif pred_label == "negative-but-not-abusive":
            if user_risk >= 0.60:
                verdict = "FLAGGED_FOR_REVIEW"
                action = "flag_for_creator_review"
                rationale = "Constructive criticism from a high-risk account. Held for optional creator inspection."
            else:
                verdict = "APPROVED"
                action = "allow"
                rationale = "Non-abusive criticism or community debate. Allowed to protect open discussion."
        else:
            verdict = "APPROVED"
            action = "allow"
            rationale = "Benign positive or neutral community engagement."

        return {
            "user_id": user_id,
            "influencer_id": influencer_id,
            "original_text_masked": mask_text_display(text),
            "predicted_label": pred_label,
            "class_probabilities": {
                "positive": round(float(probs[0]), 4),
                "neutral": round(float(probs[1]), 4),
                "criticism": round(float(probs[2]), 4),
                "abusive": round(float(probs[3]), 4)
            },
            "author_risk_score": round(user_risk, 3),
            "author_repeat_offences": repeat_offences,
            "author_archetype": archetype,
            "moderation_verdict": verdict,
            "recommended_action": action,
            "decision_rationale": rationale
        }

def test_moderation_engine():
    """Verify moderation decisions on test cases."""
    engine = ModerationEngine()
    test_cases = [
        ("Your editing in this tutorial is amazing, learned so much!", "user_0001"),
        ("I disagree with your point about the pricing, seems quite high.", "user_0002"),
        ("You are an absolute clown and your content is garbage.", "user_1922"),
        ("This is the worst fucking video, go delete your channel asshole.", "user_0686")
    ]

    print("\n" + "=" * 80)
    print(" AUTOMATED MODERATION ENGINE TEST RUN")
    print("=" * 80)
    for text, uid in test_cases:
        res = engine.evaluate_comment(text, uid)
        print(f" Comment   : {res['original_text_masked']}")
        print(f" Class     : {res['predicted_label']} (Abuse Prob: {res['class_probabilities']['abusive']*100:.1f}%)")
        print(f" Verdict   : {res['moderation_verdict']} -> Action: {res['recommended_action']}")
        print(f" Rationale : {res['decision_rationale']}")
        print("-" * 80)

if __name__ == "__main__":
    test_moderation_engine()
