"""
Live Comment Moderation Pipeline & Policy Enforcement Service.

Executes:
1. User restriction / cooldown status check.
2. Federated Classifier inference (positive, neutral, criticism, abusive).
3. Author behavioral risk profiling update (bursts, repeat offences, escalation slope).
4. Multi-tiered graded moderation enforcement (publish, warn, hold, mute, block).
5. Audit logging, notifications, and real-time WebSocket dispatch.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, Optional
import numpy as np
from sqlalchemy.orm import Session

from backend.models import User, Post, Comment, Influencer, Notification, ModerationAuditLog
from backend.services.model_service import model_service
from modules.behaviour_analyzer import SEVERITY_MAP

class PolicyService:
    """Orchestrates real-time comment moderation and graded policy actions."""

    def check_user_restriction(self, user: User, db: Session) -> Tuple[bool, Optional[str], int]:
        """
        Check if user is currently under a timed restriction (mute, slow mode, or suspension).
        Returns: (is_restricted, restriction_type, remaining_minutes)
        """
        if not user.is_restricted:
            return False, None, 0

        now = datetime.utcnow()
        if user.restriction_until and user.restriction_until > now:
            remaining_mins = int((user.restriction_until - now).total_seconds() / 60) + 1
            return True, user.restriction_type, remaining_mins

        # Restriction expired -> auto-lift
        user.is_restricted = False
        user.restriction_type = None
        user.restriction_until = None
        db.commit()
        return False, None, 0

    def compute_author_behavior(self, user_id: str, db: Session) -> Dict[str, Any]:
        """
        Aggregate author's historical comments from SQLite to compute
        repeat offence count, burst activity, escalation slope, and risk score.
        """
        user_comments = db.query(Comment).filter(Comment.user_id == user_id).order_by(Comment.created_at.asc()).all()
        total_comments = len(user_comments)
        if total_comments == 0:
            return {
                "total_comments": 0,
                "repeat_offences": 0,
                "abuse_ratio": 0.0,
                "burst_comments": 0,
                "burst_ratio": 0.0,
                "escalation_slope": 0.0,
                "author_risk_score": 0.0,
                "archetype": "normal_positive",
                "is_recent_burst": False
            }

        abu_count = sum(1 for c in user_comments if c.label == "abusive-vulgar")
        crit_count = sum(1 for c in user_comments if c.label == "negative-but-not-abusive")
        abuse_ratio = abu_count / float(total_comments)

        # Burst detection (< 300 seconds between consecutive comments)
        burst_count = 0
        is_recent_burst = False
        now = datetime.utcnow()

        if total_comments >= 2:
            for i in range(1, total_comments):
                delta_sec = (user_comments[i].created_at - user_comments[i-1].created_at).total_seconds()
                if delta_sec <= 300:
                    burst_count += 1
            
            # Check if last comment was in the last 5 minutes
            last_delta = (now - user_comments[-1].created_at).total_seconds()
            if last_delta <= 300:
                is_recent_burst = True

        burst_ratio = burst_count / float(total_comments)

        # Escalation slope
        severities = [SEVERITY_MAP.get(str(c.abuse_severity).lower(), 0.0) for c in user_comments]
        n = len(severities)
        if n >= 2:
            x = np.arange(n, dtype=np.float32) / (n - 1)
            x_mean = np.mean(x)
            y_mean = np.mean(severities)
            denom = np.sum((x - x_mean) ** 2)
            slope = float(np.sum((x - x_mean) * (severities - y_mean)) / denom) if denom > 1e-6 else 0.0
            escalation_slope = float(np.clip(slope / 3.0, -1.0, 1.0))
        else:
            escalation_slope = 0.0

        # Composite Risk Score
        repeat_penalty = min(1.0, abu_count / 4.0)
        score = (
            0.35 * abuse_ratio +
            0.25 * repeat_penalty +
            0.15 * burst_ratio +
            0.15 * max(0.0, escalation_slope) +
            0.10 * (abu_count / float(total_comments))
        )
        risk_score = round(float(np.clip(score, 0.0, 1.0)), 4)

        # Inferred archetype
        if burst_ratio >= 0.40 and burst_count >= 2:
            archetype = "burst_spammer"
        elif escalation_slope > 0.15 and abu_count >= 1:
            archetype = "escalating_abuser"
        elif abuse_ratio >= 0.40 or abu_count >= 2:
            archetype = "repeat_abuser"
        elif crit_count / total_comments >= 0.35 and abu_count == 0:
            archetype = "occasional_critic"
        elif total_comments >= 5 and abuse_ratio < 0.10:
            archetype = "frequent_commenter"
        else:
            archetype = "normal_positive"

        return {
            "total_comments": total_comments,
            "repeat_offences": abu_count,
            "abuse_ratio": round(abuse_ratio, 3),
            "burst_comments": burst_count,
            "burst_ratio": round(burst_ratio, 3),
            "escalation_slope": round(escalation_slope, 3),
            "author_risk_score": risk_score,
            "archetype": archetype,
            "is_recent_burst": is_recent_burst
        }

    def process_and_moderate_comment(
        self,
        user: User,
        post: Post,
        text: str,
        db: Session
    ) -> Dict[str, Any]:
        """
        Executes end-to-end Live Comment Moderation Pipeline.
        """
        # 1. Restriction Check
        is_restricted, res_type, remaining_mins = self.check_user_restriction(user, db)
        if is_restricted:
            return {
                "allowed": False,
                "status": "blocked_by_restriction",
                "message": f"You are currently restricted ({res_type.replace('_', ' ')}). Please wait {remaining_mins} minutes before commenting again.",
                "remaining_minutes": remaining_mins
            }

        # 2. Classifier Prediction via Model Service
        pred = model_service.predict(text)
        label = pred["label"]
        abuse_severity = pred["abuse_severity"]
        confidence = pred["confidence"]
        abuse_prob = pred["abuse_probability"]
        masked_text = pred["masked_text"]
        top_keywords = pred["top_keywords"]

        # 3. Behavioral Risk Profiling
        b_ctx = self.compute_author_behavior(user.id, db)
        author_risk = b_ctx["author_risk_score"]
        repeat_offences = b_ctx["repeat_offences"]
        is_burst = b_ctx["is_recent_burst"]

        # 4. Multi-Tiered Graded Moderation Policy Decision
        warning_issued = False
        warning_msg = None
        action_taken = "allow"
        status = "published"
        reason = "Clean community contribution."

        if label == "abusive-vulgar":
            if author_risk >= 0.70 or repeat_offences >= 4 or abuse_severity == "severe":
                status = "rejected"
                action_taken = "mute_48h"
                verdict = "BLOCKED"
                reason = f"Severe toxicity ({abuse_prob*100:.1f}%) from chronic repeat abuser ({repeat_offences} prior offences)."
                
                # Apply 48-hour mute to user
                user.is_restricted = True
                user.restriction_type = "mute_48h"
                user.restriction_until = datetime.utcnow() + timedelta(hours=48)
                
                # Notification to user
                notif = Notification(
                    user_id=user.id,
                    title="Account Muted (48 Hours)",
                    message=f"Your comment was rejected and your account muted for 48h due to persistent abusive language. Reason: {reason}",
                    type="mute"
                )
                db.add(notif)

            elif is_burst or b_ctx["archetype"] == "burst_spammer":
                status = "held_for_review"
                action_taken = "slow_mode_15m"
                verdict = "RATE_LIMITED"
                reason = "Rapid burst commenting detected. Comment placed in review and 15-minute slow mode cooldown enforced."
                
                user.is_restricted = True
                user.restriction_type = "slow_mode_15m"
                user.restriction_until = datetime.utcnow() + timedelta(minutes=15)
                
                notif = Notification(
                    user_id=user.id,
                    title="Slow Mode Cooldown (15 Min)",
                    message="You are commenting too rapidly. Your comment has been held for review and a 15-minute cooldown applied.",
                    type="cooldown"
                )
                db.add(notif)

            else:
                status = "held_for_review"
                action_taken = "quarantine_for_review"
                verdict = "AUTO_QUARANTINED"
                reason = f"Hostile language detected ({abuse_prob*100:.1f}%, Severity: {abuse_severity}). Placed in moderator quarantine queue."
                
                notif = Notification(
                    user_id=user.id,
                    title="Comment Held for Review",
                    message="Your comment has been held for human moderation inspection. You may file an appeal from your profile.",
                    type="review_hold"
                )
                db.add(notif)

        elif label == "negative-but-not-abusive":
            if author_risk >= 0.60:
                status = "held_for_review"
                action_taken = "shadow_filter"
                verdict = "FLAGGED_FOR_REVIEW"
                reason = "Constructive criticism from a high-risk account held for creator inspection."
            elif abuse_prob >= 0.25:
                status = "flagged_warning"
                action_taken = "warn"
                verdict = "PUBLISHED_WITH_WARNING"
                warning_issued = True
                warning_msg = "Your comment was published, but please avoid aggressive terminology to maintain positive community standards."
                reason = "Borderline criticism published with warning notification to author."
                
                notif = Notification(
                    user_id=user.id,
                    title="Community Standard Reminder",
                    message=warning_msg,
                    type="warning"
                )
                db.add(notif)
            else:
                status = "published"
                action_taken = "allow"
                verdict = "APPROVED"
                reason = "Constructive feedback and debate allowed under Free Speech Guidelines."
        else:
            status = "published"
            action_taken = "allow"
            verdict = "APPROVED"
            reason = "Benign positive or informational interaction."

        # 5. Create Comment Record in SQLite
        new_comment = Comment(
            post_id=post.id,
            user_id=user.id,
            influencer_id=post.influencer_id,
            text=text,
            masked_text=masked_text,
            status=status,
            label=label,
            abuse_severity=abuse_severity,
            confidence=confidence,
            author_risk_score=author_risk,
            moderation_reason=reason,
            top_keywords=top_keywords,
            created_at=datetime.utcnow()
        )
        db.add(new_comment)

        # Update Post comment count
        if status in ["published", "flagged_warning"]:
            post.comments_count += 1

        # 6. Create Audit Log for non-trivial decisions
        if action_taken != "allow":
            audit = ModerationAuditLog(
                comment_id=new_comment.id,
                target_user_id=user.id,
                moderator_id="system_policy_engine",
                action_taken=action_taken,
                reason=reason,
                previous_status="pending",
                new_status=status
            )
            db.add(audit)

        db.commit()
        db.refresh(new_comment)

        return {
            "allowed": True,
            "comment_id": new_comment.id,
            "status": status,
            "verdict": verdict,
            "action_taken": action_taken,
            "label": label,
            "abuse_severity": abuse_severity,
            "confidence": confidence,
            "author_risk_score": author_risk,
            "masked_text": masked_text,
            "moderation_reason": reason,
            "warning_issued": warning_issued,
            "warning_message": warning_msg,
            "top_keywords": top_keywords,
            "created_at": new_comment.created_at
        }

policy_service = PolicyService()
