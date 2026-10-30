"""
Comment and Live Moderation Pipeline API Routes for EchoSafe.
"""

from typing import List
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import User, Post, Comment, Appeal, Influencer
from backend.auth import get_current_user
from backend.schemas import CommentCreate, CommentResponse, CommentModerationResult, AppealCreate, AppealResponse
from backend.services.policy_service import policy_service
from backend.services.websocket_manager import ws_manager

router = APIRouter(tags=["Comments & Moderation Pipeline"])

@router.get("/api/posts/{post_id}/comments", response_model=List[CommentResponse])
def get_post_comments(
    post_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Retrieve comments for a post.
    Normal users see only published comments and their own held comments.
    Moderators and the post's Influencer see all comments including held ones.
    """
    post = db.query(Post).filter(Post.id == post_id).first()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found.")

    is_privileged = (
        current_user.role in ["moderator", "admin"] or
        (post.influencer and post.influencer.user_id == current_user.id)
    )

    query = db.query(Comment).filter(Comment.post_id == post_id).order_by(Comment.created_at.asc())
    if not is_privileged:
        # Regular users see published or their own held comments
        query = query.filter((Comment.status.in_(["published", "flagged_warning"])) | (Comment.user_id == current_user.id))

    comments = query.all()
    results = []
    for c in comments:
        author = c.author
        results.append({
            "id": c.id,
            "post_id": c.post_id,
            "user_id": c.user_id,
            "username": author.username if author else "user",
            "user_avatar": author.avatar_url if author else None,
            "text": c.masked_text if not is_privileged else c.text,
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

@router.post("/api/comments", response_model=CommentModerationResult)
async def post_comment_with_pipeline(
    payload: CommentCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Core Live Comment Pipeline:
    1. Checks author restriction status.
    2. Runs federated neural comment classifier.
    3. Profiles author's historical behavior (bursts, escalation, repeat offences).
    4. Executes graded policy engine (publish, warn, hold, mute).
    5. Dispatches real-time WebSocket notifications.
    """
    post = db.query(Post).filter(Post.id == payload.post_id).first()
    if not post:
        raise HTTPException(status_code=404, detail="Target post not found.")

    # Process comment through Live Moderation Pipeline
    result = policy_service.process_and_moderate_comment(
        user=current_user,
        post=post,
        text=payload.text,
        db=db
    )

    if not result["allowed"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=result["message"]
        )

    # Broadcast event via WebSocket
    if result["status"] in ["published", "flagged_warning"]:
        await ws_manager.broadcast({
            "event": "new_comment",
            "data": {
                "post_id": post.id,
                "comment_id": result["comment_id"],
                "username": current_user.username,
                "user_avatar": current_user.avatar_url,
                "masked_text": result["masked_text"],
                "status": result["status"]
            }
        }, channel="feed")
    elif result["status"] == "held_for_review":
        await ws_manager.broadcast({
            "event": "comment_held",
            "data": {
                "comment_id": result["comment_id"],
                "author": current_user.username,
                "label": result["label"],
                "severity": result["abuse_severity"],
                "reason": result["moderation_reason"]
            }
        }, channel="moderation")

    return {
        "comment_id": result["comment_id"],
        "status": result["status"],
        "verdict": result["verdict"],
        "action_taken": result["action_taken"],
        "label": result["label"],
        "abuse_severity": result["abuse_severity"],
        "confidence": result["confidence"],
        "author_risk_score": result["author_risk_score"],
        "masked_text": result["masked_text"],
        "moderation_reason": result["moderation_reason"],
        "warning_issued": result["warning_issued"],
        "warning_message": result["warning_message"]
    }

@router.post("/api/comments/{comment_id}/appeal", response_model=AppealResponse)
def submit_appeal(
    comment_id: str,
    payload: AppealCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Submit an appeal for a held or rejected comment."""
    comment = db.query(Comment).filter(Comment.id == comment_id, Comment.user_id == current_user.id).first()
    if not comment:
        raise HTTPException(status_code=404, detail="Comment not found or does not belong to you.")

    # Check if appeal already exists
    existing = db.query(Appeal).filter(Appeal.comment_id == comment_id).first()
    if existing:
        raise HTTPException(status_code=400, detail="An appeal is already pending for this comment.")

    appeal = Appeal(
        comment_id=comment_id,
        user_id=current_user.id,
        reason=payload.reason,
        status="pending"
    )
    db.add(appeal)
    db.commit()
    db.refresh(appeal)

    return {
        "id": appeal.id,
        "comment_id": appeal.comment_id,
        "user_id": appeal.user_id,
        "username": current_user.username,
        "comment_text": comment.masked_text,
        "reason": appeal.reason,
        "status": appeal.status,
        "moderator_notes": appeal.moderator_notes,
        "created_at": appeal.created_at
    }
