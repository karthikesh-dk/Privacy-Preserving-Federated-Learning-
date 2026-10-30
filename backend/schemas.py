"""
Pydantic Validation Schemas for EchoSafe REST API.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

# ----------------- Auth Schemas -----------------
class UserRegister(BaseModel):
    username: str = Field(..., min_length=3, max_length=32)
    email: str = Field(..., min_length=5)
    password: str = Field(..., min_length=6)
    role: Optional[str] = "follower"  # follower, influencer, moderator, admin
    bio: Optional[str] = ""

class UserLogin(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    username: str
    role: str
    avatar_url: Optional[str] = None

class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    role: str
    avatar_url: Optional[str] = None
    bio: Optional[str] = ""
    archetype: Optional[str] = "normal_positive"
    is_restricted: bool = False
    restriction_type: Optional[str] = None
    restriction_until: Optional[datetime] = None
    created_at: datetime

    class Config:
        from_attributes = True

# ----------------- Influencer Schemas -----------------
class InfluencerResponse(BaseModel):
    id: str
    user_id: str
    name: str
    niche: str
    platform: str
    follower_count: int
    community_health_score: float
    toxicity_velocity: float
    is_raid_detected: bool
    protective_recommendation: str

    class Config:
        from_attributes = True

class InfluencerAnalytics(BaseModel):
    influencer_id: str
    name: str
    niche: str
    total_posts: int
    total_comments: int
    community_health_score: float
    toxicity_velocity: float
    is_raid_detected: bool
    protective_recommendation: str
    sentiment_breakdown: Dict[str, int]
    post_type_abuse_rates: Dict[str, float]
    top_repeat_offenders: List[Dict[str, Any]]

# ----------------- Post Schemas -----------------
class PostCreate(BaseModel):
    post_type: str = Field("image", description="image, video, reel, text, opinion, live")
    caption: str = Field(..., min_length=1)
    media_url: Optional[str] = None

class PostResponse(BaseModel):
    id: str
    influencer_id: str
    influencer_name: str
    influencer_avatar: Optional[str] = None
    niche: str
    platform: str
    post_type: str
    caption: str
    media_url: Optional[str] = None
    likes_count: int
    comments_count: int
    created_at: datetime
    is_liked_by_me: bool = False

    class Config:
        from_attributes = True

# ----------------- Comment Schemas -----------------
class CommentCreate(BaseModel):
    post_id: str
    text: str = Field(..., min_length=1)

class CommentResponse(BaseModel):
    id: str
    post_id: str
    user_id: str
    username: str
    user_avatar: Optional[str] = None
    text: str
    masked_text: str
    status: str
    label: str
    abuse_severity: str
    confidence: float
    author_risk_score: float
    moderation_reason: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True

class CommentModerationResult(BaseModel):
    comment_id: str
    status: str
    verdict: str
    action_taken: str
    label: str
    abuse_severity: str
    confidence: float
    author_risk_score: float
    masked_text: str
    moderation_reason: str
    warning_issued: bool = False
    warning_message: Optional[str] = None

# ----------------- Moderation & Appeals -----------------
class ModerationActionRequest(BaseModel):
    action: str = Field(..., description="approve, warn, slow_mode_15m, shadow_filter, mute_48h, block")
    reason: str = Field(..., min_length=3)

class AppealCreate(BaseModel):
    comment_id: str
    reason: str = Field(..., min_length=5)

class AppealResponse(BaseModel):
    id: str
    comment_id: str
    user_id: str
    username: str
    comment_text: str
    reason: str
    status: str
    moderator_notes: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True

class ModerationAuditLogResponse(BaseModel):
    id: str
    comment_id: Optional[str] = None
    target_user_id: str
    moderator_id: str
    action_taken: str
    reason: str
    created_at: datetime

    class Config:
        from_attributes = True

# ----------------- Notification Schemas -----------------
class NotificationResponse(BaseModel):
    id: str
    title: str
    message: str
    type: str
    is_read: bool
    created_at: datetime

    class Config:
        from_attributes = True

# ----------------- Federated Learning Console -----------------
class RunFederatedRoundRequest(BaseModel):
    strategy: str = "FedAvg"  # FedAvg, FedProx
    proximal_mu: float = 0.01
    enable_secagg: bool = False
    enable_dp: bool = False
    dp_noise_multiplier: float = 0.5
    dp_clip_norm: float = 1.5

class FederatedRoundResponse(BaseModel):
    round_number: int
    strategy: str
    secagg_enabled: bool
    dp_enabled: bool
    dp_epsilon_spent: float
    macro_f1: float
    accuracy: float
    abuse_f1: float
    fpr_normal: float
    comm_overhead_kb: float
    created_at: datetime
