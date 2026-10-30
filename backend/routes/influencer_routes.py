"""
Influencer Creator Studio and Analytics API Routes for EchoSafe.
"""

from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func

from backend.database import get_db
from backend.models import Influencer, Post, Comment, User
from backend.schemas import InfluencerResponse, InfluencerAnalytics

router = APIRouter(prefix="/api/influencers", tags=["Influencers & Creator Analytics"])

@router.get("", response_model=List[InfluencerResponse])
def get_all_influencers(db: Session = Depends(get_db)):
    """Return all registered influencer client nodes."""
    return db.query(Influencer).all()

@router.get("/{influencer_id}", response_model=InfluencerResponse)
def get_influencer_profile(influencer_id: str, db: Session = Depends(get_db)):
    """Return single creator profile with live wellbeing score."""
    inf = db.query(Influencer).filter(Influencer.id == influencer_id).first()
    if not inf:
        raise HTTPException(status_code=404, detail="Influencer not found.")
    return inf

@router.get("/{influencer_id}/analytics", response_model=InfluencerAnalytics)
def get_influencer_analytics(influencer_id: str, db: Session = Depends(get_db)):
    """
    Return comprehensive Creator Studio analytics:
    - Abuse rate per post
    - Sentiment breakdown (Positive, Neutral, Criticism, Abusive)
    - Post type abuse rates (Reels vs Opinions vs Images)
    - Top repeat offenders targeting this creator
    """
    inf = db.query(Influencer).filter(Influencer.id == influencer_id).first()
    if not inf:
        raise HTTPException(status_code=404, detail="Influencer not found.")

    total_posts = db.query(Post).filter(Post.influencer_id == influencer_id).count()
    comments = db.query(Comment).filter(Comment.influencer_id == influencer_id).all()
    total_comments = len(comments)

    # Sentiment Breakdown
    sentiment_counts = {"positive": 0, "neutral": 0, "criticism": 0, "abusive": 0}
    user_offence_map = {}
    post_type_stats = {}

    # Query posts by type
    posts = db.query(Post).filter(Post.influencer_id == influencer_id).all()
    post_type_map = {p.id: p.post_type for p in posts}

    for c in comments:
        lbl = c.label
        if lbl in sentiment_counts:
            sentiment_counts[lbl] += 1
        elif "abuse" in lbl:
            sentiment_counts["abusive"] += 1
        elif "crit" in lbl or "neg" in lbl:
            sentiment_counts["criticism"] += 1
        else:
            sentiment_counts["positive"] += 1

        if c.label == "abusive-vulgar":
            user_offence_map[c.user_id] = user_offence_map.get(c.user_id, 0) + 1

        ptype = post_type_map.get(c.post_id, "general")
        if ptype not in post_type_stats:
            post_type_stats[ptype] = {"total": 0, "abusive": 0}
        post_type_stats[ptype]["total"] += 1
        if c.label == "abusive-vulgar":
            post_type_stats[ptype]["abusive"] += 1

    # Post type abuse percentages
    post_type_rates = {}
    for ptype, s in post_type_stats.items():
        rate = (s["abusive"] / float(s["total"])) * 100 if s["total"] > 0 else 0.0
        post_type_rates[ptype] = round(rate, 1)

    # Top repeat offenders targeting this creator
    sorted_offenders = sorted(user_offence_map.items(), key=lambda x: x[1], reverse=True)[:5]
    top_offenders = []
    for uid, count in sorted_offenders:
        user_obj = db.query(User).filter(User.id == uid).first()
        top_offenders.append({
            "user_id": uid,
            "username": user_obj.username if user_obj else uid,
            "repeat_offences_here": count,
            "is_restricted": user_obj.is_restricted if user_obj else False,
            "restriction_type": user_obj.restriction_type if user_obj else None
        })

    return {
        "influencer_id": inf.id,
        "name": inf.name,
        "niche": inf.niche,
        "total_posts": total_posts,
        "total_comments": total_comments,
        "community_health_score": inf.community_health_score,
        "toxicity_velocity": inf.toxicity_velocity,
        "is_raid_detected": inf.is_raid_detected,
        "protective_recommendation": inf.protective_recommendation,
        "sentiment_breakdown": sentiment_counts,
        "post_type_abuse_rates": post_type_rates,
        "top_repeat_offenders": top_offenders
    }
