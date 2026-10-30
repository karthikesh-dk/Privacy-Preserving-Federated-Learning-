"""
Post and Feed API Routes for EchoSafe.
"""

from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import User, Post, Influencer, Like, Follow
from backend.auth import get_current_user
from backend.schemas import PostCreate, PostResponse
from backend.services.websocket_manager import ws_manager

router = APIRouter(prefix="/api/posts", tags=["Posts & Feed"])

@router.get("", response_model=List[PostResponse])
def get_home_feed(
    post_type: Optional[str] = None,
    limit: int = 50,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Returns home feed of posts across creators with like status.
    Filterable by post_type (reel, image, video, text, opinion, live).
    """
    query = db.query(Post).order_by(Post.created_at.desc())
    if post_type:
        query = query.filter(Post.post_type == post_type.lower())

    posts = query.limit(limit).all()

    # User likes cache
    user_liked_post_ids = set(
        pid[0] for pid in db.query(Like.post_id).filter(Like.user_id == current_user.id).all()
    )

    feed = []
    for p in posts:
        inf = p.influencer
        feed.append({
            "id": p.id,
            "influencer_id": p.influencer_id,
            "influencer_name": inf.name if inf else "Community Creator",
            "influencer_avatar": inf.user.avatar_url if inf and inf.user else None,
            "niche": inf.niche if inf else "General",
            "platform": inf.platform if inf else "Instagram",
            "post_type": p.post_type,
            "caption": p.caption,
            "media_url": p.media_url,
            "likes_count": p.likes_count,
            "comments_count": p.comments_count,
            "created_at": p.created_at,
            "is_liked_by_me": p.id in user_liked_post_ids
        })

    return feed

@router.post("", response_model=PostResponse)
async def create_post(
    payload: PostCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Influencers create and publish new content.
    Supported types: image, video, reel, text, opinion, live.
    """
    # Check if user is an influencer
    influencer = db.query(Influencer).filter(Influencer.user_id == current_user.id).first()
    if not influencer and current_user.role != "admin":
        # Auto-create influencer profile for demo purposes if creator posted
        influencer = Influencer(
            user_id=current_user.id,
            name=current_user.username.replace("_", " ").title(),
            niche="General Content",
            platform="Instagram",
            follower_count=12000
        )
        db.add(influencer)
        db.commit()
        db.refresh(influencer)

    sample_media = {
        "reel": "https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?w=600",
        "image": "https://images.unsplash.com/photo-1518770660439-4636190af475?w=600",
        "video": "https://images.unsplash.com/photo-1536240478700-b869070f9279?w=600",
        "opinion": "https://images.unsplash.com/photo-1457369804613-52c61a468e7d?w=600",
        "text": "https://images.unsplash.com/photo-1455390582262-044cdead277a?w=600",
        "live": "https://images.unsplash.com/photo-1516251193007-45ef944ab0c6?w=600"
    }

    ptype = payload.post_type.lower()
    media_url = payload.media_url or sample_media.get(ptype, sample_media["image"])

    new_post = Post(
        influencer_id=influencer.id,
        post_type=ptype,
        caption=payload.caption,
        media_url=media_url,
        created_at=datetime.utcnow()
    )
    db.add(new_post)
    db.commit()
    db.refresh(new_post)

    resp = {
        "id": new_post.id,
        "influencer_id": new_post.influencer_id,
        "influencer_name": influencer.name,
        "influencer_avatar": current_user.avatar_url,
        "niche": influencer.niche,
        "platform": influencer.platform,
        "post_type": new_post.post_type,
        "caption": new_post.caption,
        "media_url": new_post.media_url,
        "likes_count": 0,
        "comments_count": 0,
        "created_at": new_post.created_at,
        "is_liked_by_me": False
    }

    # Broadcast new post to live feed WebSocket
    await ws_manager.broadcast({"event": "new_post", "data": resp}, channel="feed")
    return resp

@router.post("/{post_id}/like")
def toggle_like(
    post_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Toggle like on post."""
    post = db.query(Post).filter(Post.id == post_id).first()
    if not post:
        raise HTTPException(status_code=404, detail="Post not found.")

    existing_like = db.query(Like).filter(Like.user_id == current_user.id, Like.post_id == post_id).first()
    if existing_like:
        db.delete(existing_like)
        post.likes_count = max(0, post.likes_count - 1)
        liked = False
    else:
        new_like = Like(user_id=current_user.id, post_id=post_id)
        db.add(new_like)
        post.likes_count += 1
        liked = True

    db.commit()
    return {"liked": liked, "likes_count": post.likes_count}
