"""
Module 4: User Behaviour Analyzer & Risk Profiler.

Implements:
1. Interaction history aggregation per user across posts and influencer communities.
2. Temporal burst detection (< 5 min clusters), repeat offence tracking, and severity escalation trend.
3. Cross-modal engagement preferences (likes/shares vs abusive commenting).
4. Composite risk scoring (0.0 to 1.0) and automated tiered moderation recommendations.
5. Evaluation benchmark against ground-truth synthetic archetypes.
"""

import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd

# Severity weight mapping for trend and risk calculation
SEVERITY_MAP = {
    "none": 0.0,
    "mild": 1.0,
    "moderate": 2.0,
    "severe": 3.0
}

class UserBehaviourAnalyzer:
    """
    Analyzes historical user interactions to profile behavioral patterns,
    detect repeat and escalating abusers, and recommend tiered moderation actions.
    """
    def __init__(
        self,
        comments_df: Optional[pd.DataFrame] = None,
        likes_shares_df: Optional[pd.DataFrame] = None,
        burst_threshold_seconds: int = 300  # 5 minutes
    ):
        self.comments_df = comments_df
        self.likes_shares_df = likes_shares_df
        self.burst_threshold_seconds = burst_threshold_seconds
        self.profiles: Dict[str, Dict[str, Any]] = {}

    def load_data(
        self,
        comments_path: str = "data/synthetic_comments.csv",
        likes_shares_path: str = "data/synthetic_likes_shares.csv"
    ):
        """Load interaction datasets from CSV files."""
        if os.path.exists(comments_path):
            self.comments_df = pd.read_csv(comments_path)
            self.comments_df["timestamp"] = pd.to_datetime(self.comments_df["timestamp"])
        else:
            raise FileNotFoundError(f"Comments data file not found at: {comments_path}")

        if os.path.exists(likes_shares_path):
            self.likes_shares_df = pd.read_csv(likes_shares_path)
            self.likes_shares_df["timestamp"] = pd.to_datetime(self.likes_shares_df["timestamp"])
        else:
            self.likes_shares_df = pd.DataFrame(columns=["user_id", "post_type", "event_type"])

    def _calculate_escalation_slope(self, severities: List[float]) -> float:
        """
        Calculate linear regression slope of abuse severity over comment sequence.
        Positive slope indicates escalating hostility over time.
        """
        n = len(severities)
        if n < 2:
            return 0.0
        x = np.arange(n, dtype=np.float32)
        y = np.array(severities, dtype=np.float32)
        # Normalize x to [0, 1] range to make slopes comparable across different comment counts
        x_norm = x / (n - 1)
        x_mean = np.mean(x_norm)
        y_mean = np.mean(y)
        denom = np.sum((x_norm - x_mean) ** 2)
        if denom < 1e-6:
            return 0.0
        slope = float(np.sum((x_norm - x_mean) * (y - y_mean)) / denom)
        # Normalize slope so 3.0 shift corresponds to 1.0
        return float(np.clip(slope / 3.0, -1.0, 1.0))

    def _detect_bursts(self, timestamps: List[pd.Timestamp]) -> Tuple[int, float, Optional[float]]:
        """
        Detect rapid-fire commenting bursts where delta-t between consecutive comments < threshold.
        Returns: (burst_comment_count, burst_ratio, min_gap_seconds)
        """
        n = len(timestamps)
        if n < 2:
            return 0, 0.0, None

        # Ensure timestamps are sorted
        sorted_times = sorted(timestamps)
        gaps = [(sorted_times[i] - sorted_times[i-1]).total_seconds() for i in range(1, n)]
        min_gap = min(gaps) if gaps else None

        burst_indices = set()
        for i, gap in enumerate(gaps):
            if gap <= self.burst_threshold_seconds:
                burst_indices.add(i)      # previous comment
                burst_indices.add(i + 1)  # current comment

        burst_count = len(burst_indices)
        burst_ratio = burst_count / float(n)
        return burst_count, burst_ratio, min_gap

    def _compute_risk_score(
        self,
        abuse_ratio: float,
        repeat_offence_count: int,
        burst_ratio: float,
        escalation_slope: float,
        severe_ratio: float
    ) -> float:
        """
        Composite risk score between 0.0 and 1.0.
        Weights:
        - 35% Abuse ratio
        - 25% Repeat offence count penalty
        - 15% Burst frequency
        - 15% Escalation trend slope
        - 10% Severe abuse ratio
        """
        repeat_penalty = min(1.0, repeat_offence_count / 4.0)
        escalation_component = max(0.0, escalation_slope)

        score = (
            0.35 * abuse_ratio +
            0.25 * repeat_penalty +
            0.15 * burst_ratio +
            0.15 * escalation_component +
            0.10 * severe_ratio
        )
        return round(float(np.clip(score, 0.0, 1.0)), 4)

    def _predict_archetype(
        self,
        total_comments: int,
        total_likes_shares: int,
        abuse_ratio: float,
        criticism_ratio: float,
        positive_ratio: float,
        repeat_offences: int,
        burst_ratio: float,
        burst_count: int,
        escalation_slope: float
    ) -> str:
        """Heuristically infer behavioral archetype from interaction signatures."""
        if total_comments == 0 and total_likes_shares > 0:
            return "lurker"
        if burst_ratio >= 0.40 and burst_count >= 2:
            return "burst_spammer"
        if escalation_slope > 0.15 and repeat_offences >= 1:
            return "escalating_abuser"
        if abuse_ratio >= 0.40 or repeat_offences >= 2:
            return "repeat_abuser"
        if criticism_ratio >= 0.30 and abuse_ratio == 0:
            return "occasional_critic"
        if total_comments >= 5 and abuse_ratio < 0.15:
            return "frequent_commenter"
        return "normal_positive"

    def _recommend_action(self, risk_score: float, archetype: str, repeat_offences: int) -> str:
        """Tiered moderation action recommendation."""
        if risk_score >= 0.80 or repeat_offences >= 5:
            return "permanent_block_referral"
        if risk_score >= 0.60 or archetype == "repeat_abuser":
            return "temporary_mute"
        if archetype == "burst_spammer":
            return "rate_limit_cooldown"
        if archetype == "escalating_abuser":
            return "shadow_filter"
        if risk_score >= 0.30 or archetype == "occasional_critic":
            return "flag_for_review"
        return "allow"

    def analyze_all_users(self) -> pd.DataFrame:
        """
        Run user behavior analysis across all users in the dataset.
        Returns a DataFrame of rich user profiles.
        """
        if self.comments_df is None:
            self.load_data()

        # Group comments and events by user_id
        comments_by_user = dict(tuple(self.comments_df.groupby("user_id")))
        
        events_by_user = {}
        if self.likes_shares_df is not None and not self.likes_shares_df.empty:
            events_by_user = dict(tuple(self.likes_shares_df.groupby("user_id")))

        all_user_ids = set(comments_by_user.keys()).union(set(events_by_user.keys()))
        profiles_list = []

        for u_id in sorted(all_user_ids):
            u_comments = comments_by_user.get(u_id, pd.DataFrame())
            u_events = events_by_user.get(u_id, pd.DataFrame())

            total_comments = len(u_comments)
            total_events = len(u_events)
            if total_events > 0 and "event_type" in u_events.columns:
                total_likes = int((u_events["event_type"] == "like").sum())
                total_shares = int((u_events["event_type"] == "share").sum())
                like_post_types = list(u_events[u_events["event_type"] == "like"]["post_type"].value_counts().to_dict().keys())[:2]
            else:
                total_likes = 0
                total_shares = 0
                like_post_types = []
            comment_post_types = []
            
            if total_comments > 0:
                # Sort comments chronologically
                u_comments_sorted = u_comments.sort_values("timestamp")
                timestamps = list(u_comments_sorted["timestamp"])
                labels = list(u_comments_sorted["label"])
                severities_raw = list(u_comments_sorted["abuse_severity"])

                # Label counts & proportions
                pos_count = labels.count("positive")
                neu_count = labels.count("neutral")
                crit_count = labels.count("negative-but-not-abusive")
                abu_count = labels.count("abusive-vulgar")

                pos_ratio = pos_count / total_comments
                neu_ratio = neu_count / total_comments
                crit_ratio = crit_count / total_comments
                abu_ratio = abu_count / total_comments

                # Severity breakdown
                sev_counts = {k: severities_raw.count(k) for k in ["none", "mild", "moderate", "severe"]}
                severe_ratio = sev_counts["severe"] / total_comments

                # Escalation slope
                sev_values = [SEVERITY_MAP.get(str(s).lower(), 0.0) for s in severities_raw]
                escalation_slope = self._calculate_escalation_slope(sev_values)

                # Bursts
                burst_count, burst_ratio, min_gap = self._detect_bursts(timestamps)

                # Distinct influencers targeted
                influencers_targeted = int(u_comments["influencer_id"].nunique())
            else:
                pos_count = neu_count = crit_count = abu_count = 0
                pos_ratio = neu_ratio = crit_ratio = abu_ratio = 0.0
                severe_ratio = 0.0
                escalation_slope = 0.0
                burst_count = 0
                burst_ratio = 0.0
                min_gap = None
                influencers_targeted = 0

            # Composite Risk Score
            risk_score = self._compute_risk_score(
                abuse_ratio=abu_ratio,
                repeat_offence_count=abu_count,
                burst_ratio=burst_ratio,
                escalation_slope=escalation_slope,
                severe_ratio=severe_ratio
            )

            # Categorize Risk
            if risk_score >= 0.65:
                risk_level = "HIGH"
            elif risk_score >= 0.30:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"

            # Predicted Archetype
            pred_arch = self._predict_archetype(
                total_comments=total_comments,
                total_likes_shares=total_events,
                abuse_ratio=abu_ratio,
                criticism_ratio=crit_ratio,
                positive_ratio=pos_ratio,
                repeat_offences=abu_count,
                burst_ratio=burst_ratio,
                burst_count=burst_count,
                escalation_slope=escalation_slope
            )

            # Recommended Action
            action = self._recommend_action(risk_score, pred_arch, abu_count)

            profile = {
                "user_id": u_id,
                "total_comments": total_comments,
                "total_likes": total_likes,
                "total_shares": total_shares,
                "positive_ratio": round(pos_ratio, 3),
                "neutral_ratio": round(neu_ratio, 3),
                "criticism_ratio": round(crit_ratio, 3),
                "abuse_ratio": round(abu_ratio, 3),
                "repeat_offences": abu_count,
                "severe_abuse_count": sev_counts["severe"] if total_comments > 0 else 0,
                "burst_comments": burst_count,
                "burst_ratio": round(burst_ratio, 3),
                "min_time_gap_sec": round(min_gap, 1) if min_gap is not None else -1.0,
                "escalation_slope": round(escalation_slope, 3),
                "influencers_targeted": influencers_targeted,
                "user_risk_score": risk_score,
                "risk_level": risk_level,
                "predicted_archetype": pred_arch,
                "recommended_action": action
            }
            profiles_list.append(profile)

        df_profiles = pd.DataFrame(profiles_list)
        return df_profiles

    def evaluate_against_ground_truth(
        self,
        profiles_df: pd.DataFrame,
        users_meta_path: str = "data/synthetic_users.json"
    ) -> Dict[str, Any]:
        """
        Benchmark behavioral analyzer against ground-truth user archetypes.
        Evaluates precision, recall, and detection accuracy for abusive archetypes.
        """
        if not os.path.exists(users_meta_path):
            return {"error": f"Metadata file not found: {users_meta_path}"}

        with open(users_meta_path, "r", encoding="utf-8") as f:
            users_meta = json.load(f)

        gt_map = {u["user_id"]: u["archetype"] for u in users_meta}
        profiles_df["ground_truth_archetype"] = profiles_df["user_id"].map(gt_map)

        # Drop users not in ground truth
        eval_df = profiles_df.dropna(subset=["ground_truth_archetype"]).copy()

        # Abuser Detection Specifics (Repeat Abuser, Escalating Abuser, Burst Spammer)
        abuser_archetypes = {"repeat_abuser", "escalating_abuser", "burst_spammer"}
        eval_df["is_true_abuser"] = eval_df["ground_truth_archetype"].isin(abuser_archetypes)
        eval_df["is_detected_abuser"] = eval_df["predicted_archetype"].isin(abuser_archetypes) | (eval_df["user_risk_score"] >= 0.45)

        tp = int(((eval_df["is_true_abuser"] == True) & (eval_df["is_detected_abuser"] == True)).sum())
        fp = int(((eval_df["is_true_abuser"] == False) & (eval_df["is_detected_abuser"] == True)).sum())
        fn = int(((eval_df["is_true_abuser"] == True) & (eval_df["is_detected_abuser"] == False)).sum())
        tn = int(((eval_df["is_true_abuser"] == False) & (eval_df["is_detected_abuser"] == False)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / len(eval_df) if len(eval_df) > 0 else 0.0
        fpr_normal = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        # Archetype breakdown
        archetype_metrics = {}
        for arch in sorted(eval_df["ground_truth_archetype"].unique()):
            arch_mask = eval_df["ground_truth_archetype"] == arch
            arch_pred_mask = eval_df["predicted_archetype"] == arch
            arch_tp = int((arch_mask & arch_pred_mask).sum())
            arch_total_gt = int(arch_mask.sum())
            arch_recall = arch_tp / arch_total_gt if arch_total_gt > 0 else 0.0
            archetype_metrics[arch] = {
                "ground_truth_count": arch_total_gt,
                "correctly_identified": arch_tp,
                "recall": round(arch_recall, 4)
            }

        benchmark_summary = {
            "total_users_evaluated": len(eval_df),
            "abuser_detection": {
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "true_negatives": tn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "overall_accuracy": round(accuracy, 4),
                "fpr_on_benign_users": round(fpr_normal, 4)
            },
            "per_archetype_recall": archetype_metrics,
            "risk_distribution": eval_df["risk_level"].value_counts().to_dict(),
            "action_distribution": eval_df["recommended_action"].value_counts().to_dict()
        }
        return benchmark_summary

def run_behaviour_analysis(
    output_profiles_csv: str = "experiments/results/user_behaviour_profiles.csv",
    output_benchmark_json: str = "experiments/results/behaviour_benchmark.json"
) -> Dict[str, Any]:
    """Execute complete behavioral profiling and benchmark evaluation."""
    os.makedirs(os.path.dirname(output_profiles_csv), exist_ok=True)

    analyzer = UserBehaviourAnalyzer()
    analyzer.load_data()
    
    print("\n" + "=" * 80)
    print(" EXECUTING USER BEHAVIOUR ANALYZER & RISK PROFILER")
    print("=" * 80)

    profiles_df = analyzer.analyze_all_users()
    profiles_df.to_csv(output_profiles_csv, index=False)
    print(f" [OK] Generated {len(profiles_df)} behavioral profiles -> {output_profiles_csv}")

    benchmark = analyzer.evaluate_against_ground_truth(profiles_df)
    with open(output_benchmark_json, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, indent=2)
    print(f" [OK] Saved benchmark metrics -> {output_benchmark_json}")

    print("\n--- Abuser Identification Benchmark ---")
    det = benchmark["abuser_detection"]
    print(f" Precision           : {det['precision']:.4f}")
    print(f" Recall              : {det['recall']:.4f}")
    print(f" F1-Score            : {det['f1_score']:.4f}")
    print(f" Overall Accuracy    : {det['overall_accuracy']:.4f}")
    print(f" Benign False Alarm  : {det['fpr_on_benign_users'] * 100:.2f}%")

    print("\n--- Recommended Moderation Actions ---")
    for act, cnt in benchmark["action_distribution"].items():
        print(f"   {act:<28} : {cnt:>4} users")

    print("\n--- High-Risk Repeat Abuser Profiles (Top 5) ---")
    high_risk = profiles_df.sort_values("user_risk_score", ascending=False).head(5)
    cols = ["user_id", "total_comments", "abuse_ratio", "repeat_offences", "burst_comments", "escalation_slope", "user_risk_score", "recommended_action"]
    print(high_risk[cols].to_string(index=False))
    print("=" * 80 + "\n")

    return benchmark

if __name__ == "__main__":
    run_behaviour_analysis()
