"""
Moderator & Admin Console API Routes for EchoSafe.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import User, Comment, Appeal, ModerationAuditLog, Notification, Like, Post
from backend.auth import get_current_user, require_roles
from backend.schemas import (
    CommentResponse, ModerationActionRequest, AppealResponse, ModerationAuditLogResponse
)
from backend.services.policy_service import policy_service

router = APIRouter(prefix="/api/moderation", tags=["Moderation Console"])

# Require moderator or admin role
mod_auth = require_roles(["moderator", "admin"])

@router.get("/queue", response_model=List[CommentResponse])
def get_moderation_queue(
    status_filter: Optional[str] = "held_for_review",
    limit: int = 50,
    db: Session = Depends(get_db),
    current_user: User = Depends(mod_auth)
):
    """
    Retrieve flagged comments for moderator review.
    Abusive text is blurred in UI with click-to-reveal.
    """
    query = db.query(Comment).order_by(Comment.created_at.desc())
    if status_filter and status_filter != "all":
        query = query.filter(Comment.status == status_filter)
    else:
        query = query.filter(Comment.status.in_(["held_for_review", "flagged_warning", "rejected"]))

    comments = query.limit(limit).all()
    results = []
    for c in comments:
        author = c.author
        results.append({
            "id": c.id,
            "post_id": c.post_id,
            "user_id": c.user_id,
            "username": author.username if author else "user",
            "user_avatar": author.avatar_url if author else None,
            "text": c.text,  # Full text provided to moderators
            "masked_text": c.masked_text,
            "status": c.status,
            "label": c.label,
            "abuse_severity": c.abuse_severity,
            "confidence": c.confidence,
            "author_risk_score": c.author_risk_score,
            "moderation_reason": c.moderation_reason,
            "created_at": c.created_at
        })
    return results

@router.post("/comments/{comment_id}/action")
def execute_moderation_action(
    comment_id: str,
    payload: ModerationActionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(mod_auth)
):
    """
    Execute human moderator action:
    Actions: approve, warn, slow_mode_15m, shadow_filter, mute_48h, block.
    """
    comment = db.query(Comment).filter(Comment.id == comment_id).first()
    if not comment:
        raise HTTPException(status_code=404, detail="Comment not found.")

    target_user = comment.author or (db.query(User).filter(User.id == comment.user_id).first() if comment.user_id else None)
    target_user_id = target_user.id if target_user else (comment.user_id or "anonymous")
    prev_status = comment.status
    action = payload.action.lower()
    now = datetime.utcnow()

    if action == "approve":
        comment.status = "published"
        new_status = "published"
        # Notify user of approval
        if target_user_id:
            notif = Notification(
                user_id=target_user_id,
                title="Comment Approved",
                message="Your held comment was reviewed and approved by a moderator.",
                type="approval"
            )
            db.add(notif)

    elif action == "warn":
        comment.status = "published"
        new_status = "flagged_warning"
        if target_user_id:
            notif = Notification(
                user_id=target_user_id,
                title="Moderator Warning",
                message=f"Official Warning: {payload.reason}",
                type="warning"
            )
            db.add(notif)

    elif action == "slow_mode_15m":
        comment.status = "held_for_review"
        new_status = "held_for_review"
        if target_user:
            target_user.is_restricted = True
            target_user.restriction_type = "slow_mode_15m"
            target_user.restriction_until = now + timedelta(minutes=15)
        if target_user_id:
            notif = Notification(
                user_id=target_user_id,
                title="Slow-Mode Cooldown Applied",
                message=f"A 15-minute cooldown has been applied to your account. Reason: {payload.reason}",
                type="cooldown"
            )
            db.add(notif)

    elif action == "mute_48h":
        comment.status = "rejected"
        new_status = "rejected"
        if target_user:
            target_user.is_restricted = True
            target_user.restriction_type = "mute_48h"
            target_user.restriction_until = now + timedelta(hours=48)
        if target_user_id:
            notif = Notification(
                user_id=target_user_id,
                title="Account Muted (48 Hours)",
                message=f"Your comment was rejected and your account was muted for 48h. Reason: {payload.reason}",
                type="mute"
            )
            db.add(notif)

    elif action == "block":
        comment.status = "rejected"
        new_status = "rejected"
        if target_user:
            target_user.is_restricted = True
            target_user.restriction_type = "block"
            target_user.restriction_until = now + timedelta(days=30)
        if target_user_id:
            notif = Notification(
                user_id=target_user_id,
                title="Account Suspended (30 Days)",
                message=f"Severe repeat harassment violation: account suspended for 30 days. Reason: {payload.reason}",
                type="mute"
            )
            db.add(notif)

    elif action == "shadow_filter":
        comment.status = "rejected"
        new_status = "shadow_filtered"

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported action: {payload.action}")

    # Record Audit Log
    audit = ModerationAuditLog(
        comment_id=comment.id,
        target_user_id=target_user_id,
        moderator_id=current_user.id,
        action_taken=action,
        reason=payload.reason,
        previous_status=prev_status,
        new_status=new_status
    )
    db.add(audit)
    db.commit()

    return {
        "success": True,
        "comment_id": comment.id,
        "action_taken": action,
        "new_status": new_status,
        "moderator": current_user.username,
        "reason": payload.reason
    }

@router.get("/users/{user_id}/profile")
def get_user_behavior_profile(
    user_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(mod_auth)
):
    """
    Returns deep behavioral profile for an account:
    - Comment frequency over time
    - Share of abusive/criticism comments
    - Escalation slope and repeat offence count
    - Wellbeing signal (LOW, MEDIUM, HIGH) with MANDATORY non-clinical disclaimer.
    """
    target_user = db.query(User).filter(User.id == user_id).first()
    if not target_user:
        raise HTTPException(status_code=404, detail="User not found.")

    now = datetime.utcnow()
    b_ctx = policy_service.compute_author_behavior(user_id, db)

    # Derive Wellbeing Signal (Non-clinical indicator)
    risk = b_ctx["author_risk_score"]
    if risk >= 0.65:
        wellbeing_signal = "HIGH"
        wellbeing_color = "#dc2626"
    elif risk >= 0.35:
        wellbeing_signal = "MEDIUM"
        wellbeing_color = "#f59e0b"
    else:
        wellbeing_signal = "LOW"
        wellbeing_color = "#16a34a"

    # Preferred post types for likes vs comments
    user_comments = db.query(Comment).filter(Comment.user_id == user_id).all()
    comment_post_ids = [c.post_id for c in user_comments]
    comment_posts = db.query(Post).filter(Post.id.in_(comment_post_ids)).all() if comment_post_ids else []
    comment_post_types = {}
    for p in comment_posts:
        comment_post_types[p.post_type] = comment_post_types.get(p.post_type, 0) + 1

    user_likes = db.query(Like).filter(Like.user_id == user_id).all()
    like_post_ids = [l.post_id for l in user_likes]
    like_posts = db.query(Post).filter(Post.id.in_(like_post_ids)).all() if like_post_ids else []
    like_post_types = {}
    for p in like_posts:
        like_post_types[p.post_type] = like_post_types.get(p.post_type, 0) + 1

    return {
        "user_id": target_user.id,
        "username": target_user.username,
        "email": target_user.email,
        "role": target_user.role,
        "avatar_url": target_user.avatar_url,
        "bio": target_user.bio,
        "is_restricted": target_user.is_restricted,
        "restriction_type": target_user.restriction_type,
        "restriction_until": target_user.restriction_until,
        "cooldown_seconds_remaining": max(0, int((target_user.restriction_until - now).total_seconds())) if target_user.restriction_until and target_user.restriction_until > now else 0,
        "total_comments": b_ctx.get("total_comments", 0),
        "negative_ratio": b_ctx.get("abuse_ratio", 0.0),
        "abuse_ratio": b_ctx.get("abuse_ratio", 0.0),
        "repeat_offence_count": b_ctx.get("repeat_offences", 0),
        "repeat_offences": b_ctx.get("repeat_offences", 0),
        "escalation_slope": b_ctx.get("escalation_slope", 0.0),
        "burst_count": b_ctx.get("burst_comments", 0),
        "burst_comments": b_ctx.get("burst_comments", 0),
        "wellbeing_signal": wellbeing_signal.lower(),
        "wellbeing_color": wellbeing_color,
        "wellbeing_disclaimer": "NOT A CLINICAL DIAGNOSIS. Provided solely as a community safety indicator for human moderator review.",
        "preferred_post_types_comments": list(comment_post_types.keys()),
        "preferred_post_types_likes": list(like_post_types.keys()),
        "behavior": b_ctx,
        "post_type_preferences": {
            "comment_distribution": comment_post_types,
            "like_distribution": like_post_types
        }
    }

@router.get("/appeals", response_model=List[AppealResponse])
def get_appeals(
    status_filter: Optional[str] = "pending",
    db: Session = Depends(get_db),
    current_user: User = Depends(mod_auth)
):
    """Retrieve user appeals."""
    query = db.query(Appeal).order_by(Appeal.created_at.desc())
    if status_filter and status_filter != "all":
        query = query.filter(Appeal.status == status_filter)

    appeals = query.all()
    results = []
    for a in appeals:
        results.append({
            "id": a.id,
            "comment_id": a.comment_id,
            "user_id": a.user_id,
            "username": a.user.username if a.user else "user",
            "comment_text": a.comment.masked_text if a.comment else "N/A",
            "reason": a.reason,
            "status": a.status,
            "moderator_notes": a.moderator_notes,
            "created_at": a.created_at
        })
    return results

@router.post("/appeals/{appeal_id}/resolve")
def resolve_appeal(
    appeal_id: str,
    action: str,  # approve or reject
    notes: Optional[str] = "Reviewed by moderation team.",
    db: Session = Depends(get_db),
    current_user: User = Depends(mod_auth)
):
    """Approve or reject a user appeal."""
    appeal = db.query(Appeal).filter(Appeal.id == appeal_id).first()
    if not appeal:
        raise HTTPException(status_code=404, detail="Appeal not found.")

    if action.lower() == "approve":
        appeal.status = "approved"
        appeal.moderator_notes = notes
        # Overturn comment status to published
        if appeal.comment:
            appeal.comment.status = "published"
            appeal.comment.moderation_reason = "Overturned on appeal by moderator."
        # Lift user restriction if any
        if appeal.user:
            appeal.user.is_restricted = False
            appeal.user.restriction_type = None
            appeal.user.restriction_until = None

        notif = Notification(
            user_id=appeal.user_id,
            title="Appeal Approved",
            message=f"Your appeal has been approved! Your comment has been restored. Notes: {notes}",
            type="approval"
        )
        db.add(notif)
    else:
        appeal.status = "rejected"
        appeal.moderator_notes = notes
        notif = Notification(
            user_id=appeal.user_id,
            title="Appeal Rejected",
            message=f"Your appeal was reviewed and rejected. Notes: {notes}",
            type="warning"
        )
        db.add(notif)

    db.commit()
    return {"success": True, "appeal_id": appeal.id, "status": appeal.status}

@router.get("/audit-logs", response_model=List[ModerationAuditLogResponse])
def get_audit_logs(
    limit: int = 50,
    db: Session = Depends(get_db),
    current_user: User = Depends(mod_auth)
):
    """Retrieve decision audit log trail."""
    return db.query(ModerationAuditLog).order_by(ModerationAuditLog.created_at.desc()).limit(limit).all()
