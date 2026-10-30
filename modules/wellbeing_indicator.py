"""
Module 7: Influencer Community Wellbeing & Toxicity Exposure Indicator.

Implements:
1. Community Health Index (0-100 scale) based on comment sentiment and abuse severity.
2. Toxicity exposure velocity tracking to detect coordinated harassment raids / pile-ons.
3. Positive-to-Toxic ratio and severe threat exposure metrics.
4. Creator mental wellbeing alert states (HEALTHY, ELEVATED_STRESS, CRITICAL_TOXICITY).
5. Automated protective recommendations (slow mode, cooldown, auto-quarantine).
"""

from typing import Dict, List, Any, Optional
import os
import json
import numpy as np
import pandas as pd

class InfluencerWellbeingIndicator:
    """
    Computes real-time community health and toxicity exposure indicators
    to safeguard social media content creators' mental wellbeing.
    """
    def __init__(self, comments_df: Optional[pd.DataFrame] = None):
        self.comments_df = comments_df

    def load_comments(self, comments_path: str = "data/synthetic_comments.csv"):
        if os.path.exists(comments_path):
            self.comments_df = pd.read_csv(comments_path)
            self.comments_df["timestamp"] = pd.to_datetime(self.comments_df["timestamp"])
        else:
            raise FileNotFoundError(f"Missing {comments_path}")

    def compute_influencer_wellbeing(self, influencer_id: str) -> Dict[str, Any]:
        """Compute community health index and safety metrics for a specific influencer."""
        if self.comments_df is None:
            self.load_comments()

        inf_comments = self.comments_df[self.comments_df["influencer_id"] == influencer_id].copy()
        if inf_comments.empty:
            return {
                "influencer_id": influencer_id,
                "community_health_score": 100.0,
                "status": "HEALTHY",
                "message": "No comments recorded for this creator."
            }

        total_comments = len(inf_comments)
        inf_comments = inf_comments.sort_values("timestamp")

        # Class counts
        pos_count = int((inf_comments["label"] == "positive").sum())
        neu_count = int((inf_comments["label"] == "neutral").sum())
        crit_count = int((inf_comments["label"] == "negative-but-not-abusive").sum())
        abu_count = int((inf_comments["label"] == "abusive-vulgar").sum())

        # Severity breakdown
        sev_counts = inf_comments["abuse_severity"].value_counts().to_dict()
        mild_abu = int(sev_counts.get("mild", 0))
        mod_abu = int(sev_counts.get("moderate", 0))
        severe_abu = int(sev_counts.get("severe", 0))

        # Toxicity penalty formulation
        abuse_rate = abu_count / total_comments
        severe_rate = severe_abu / total_comments
        pos_rate = pos_count / total_comments

        # Health score: 100 base, penalized by abuse and rewarded by positive support
        raw_score = 100.0 - (abuse_rate * 120.0) - (severe_rate * 80.0) - (crit_count / total_comments * 20.0) + (pos_rate * 25.0)
        health_score = round(float(np.clip(raw_score, 0.0, 100.0)), 1)

        # Toxicity Velocity (compare first half of timeline vs second half)
        mid_idx = total_comments // 2
        first_half = inf_comments.iloc[:mid_idx]
        second_half = inf_comments.iloc[mid_idx:]

        rate_first = (first_half["label"] == "abusive-vulgar").mean() if len(first_half) > 0 else 0.0
        rate_second = (second_half["label"] == "abusive-vulgar").mean() if len(second_half) > 0 else 0.0
        velocity = round(float(rate_second - rate_first), 3)

        # Status determination
        if health_score >= 75.0:
            status = "HEALTHY"
            alert_color = "#16a34a"  # Green
            protective_action = "Standard moderation active. Community environment is positive and constructive."
        elif health_score >= 50.0:
            status = "ELEVATED_STRESS"
            alert_color = "#f59e0b"  # Amber
            protective_action = "Enable Slow-Mode (15m delay) on controversial posts and flag repeat critics for review."
        else:
            status = "CRITICAL_TOXICITY"
            alert_color = "#dc2626"  # Red
            protective_action = "Harassment surge detected! Auto-quarantine comments from unverified accounts and pause push notifications."

        # Unique toxic authors
        toxic_authors = int(inf_comments[inf_comments["label"] == "abusive-vulgar"]["user_id"].nunique())

        return {
            "influencer_id": influencer_id,
            "total_comments": total_comments,
            "community_health_score": health_score,
            "status": status,
            "alert_color": alert_color,
            "toxicity_velocity": velocity,
            "is_raid_detected": bool(velocity > 0.10 and abu_count > 10),
            "positive_ratio": round(pos_rate, 3),
            "abuse_ratio": round(abuse_rate, 3),
            "severe_threat_count": severe_abu,
            "unique_toxic_users": toxic_authors,
            "protective_recommendation": protective_action
        }

    def compute_all_influencers(
        self,
        influencers_json_path: str = "data/synthetic_influencers.json"
    ) -> List[Dict[str, Any]]:
        """Compute wellbeing indicators across all registered influencers."""
        if not os.path.exists(influencers_json_path):
            raise FileNotFoundError(f"Missing {influencers_json_path}")

        with open(influencers_json_path, "r", encoding="utf-8") as f:
            influencers = json.load(f)

        results = []
        for inf in influencers:
            res = self.compute_influencer_wellbeing(inf["influencer_id"])
            res["creator_name"] = inf["name"]
            res["niche"] = inf["niche"]
            res["platform"] = inf["platform"]
            results.append(res)

        return results

def test_wellbeing_indicator():
    """Verify wellbeing calculation across synthetic influencers."""
    indicator = InfluencerWellbeingIndicator()
    indicator.load_comments()
    all_metrics = indicator.compute_all_influencers()

    print("\n" + "=" * 80)
    print(" INFLUENCER COMMUNITY MENTAL WELLBEING & HEALTH REPORT")
    print("=" * 80)
    for m in all_metrics:
        print(f" Creator: {m['creator_name']:<22} | Platform: {m['platform']:<10} | Health: {m['community_health_score']:>5.1f}/100 | Status: {m['status']}")
        if m["is_raid_detected"]:
            print(f"   [!] HARASSMENT SURGE ALERT: Toxicity velocity = +{m['toxicity_velocity']*100:.1f}%")
        print(f"   Recommendation: {m['protective_recommendation']}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    test_wellbeing_indicator()
