"""
SQLAlchemy ORM Models for EchoSafe Platform.
"""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from backend.database import Base

def generate_uuid():
    return str(uuid.uuid4())

class User(Base):
    __tablename__ = "users"

    id = Column(String(64), primary_key=True, default=generate_uuid)
    username = Column(String(64), unique=True, index=True, nullable=False)
    email = Column(String(128), unique=True, index=True, nullable=False)
    hashed_password = Column(String(256), nullable=False)
    role = Column(String(32), default="follower", nullable=False)  # follower, influencer, moderator, admin
    avatar_url = Column(String(512), nullable=True)
    bio = Column(String(256), default="")
    archetype = Column(String(64), default="normal_positive")
    
    # Restriction status
    is_restricted = Column(Boolean, default=False)
    restriction_until = Column(DateTime, nullable=True)
    restriction_type = Column(String(64), nullable=True)  # slow_mode_15m, shadow_filter, mute_48h, block
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    influencer_profile = relationship("Influencer", back_populates="user", uselist=False)
    comments = relationship("Comment", back_populates="author", cascade="all, delete-orphan")
    likes = relationship("Like", back_populates="user", cascade="all, delete-orphan")
    notifications = relationship("Notification", back_populates="user", cascade="all, delete-orphan")
    appeals = relationship("Appeal", back_populates="user", cascade="all, delete-orphan")

class Influencer(Base):
    __tablename__ = "influencers"

    id = Column(String(64), primary_key=True, default=generate_uuid)  # e.g. inf_0, inf_1
    user_id = Column(String(64), ForeignKey("users.id"), nullable=False)
    name = Column(String(128), nullable=False)
    niche = Column(String(128), nullable=False)
    platform = Column(String(64), default="Instagram")
    follower_count = Column(Integer, default=50000)
    
    # Community Wellbeing Indicators
    community_health_score = Column(Float, default=85.0)
    toxicity_velocity = Column(Float, default=0.0)
    is_raid_detected = Column(Boolean, default=False)
    protective_recommendation = Column(String(256), default="Standard moderation active.")
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="influencer_profile")
    posts = relationship("Post", back_populates="influencer", cascade="all, delete-orphan")
    followers = relationship("Follow", back_populates="influencer", cascade="all, delete-orphan")

class Post(Base):
    __tablename__ = "posts"

    id = Column(String(64), primary_key=True, default=generate_uuid)
    influencer_id = Column(String(64), ForeignKey("influencers.id"), nullable=False)
    post_type = Column(String(32), default="image")  # image, video, reel, text, opinion, live
    caption = Column(Text, nullable=False)
    media_url = Column(String(512), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    likes_count = Column(Integer, default=0)
    comments_count = Column(Integer, default=0)

    # Relationships
    influencer = relationship("Influencer", back_populates="posts")
    comments = relationship("Comment", back_populates="post", cascade="all, delete-orphan")
    likes = relationship("Like", back_populates="post", cascade="all, delete-orphan")

class Comment(Base):
    __tablename__ = "comments"

    id = Column(String(64), primary_key=True, default=generate_uuid)
    post_id = Column(String(64), ForeignKey("posts.id"), nullable=False)
    user_id = Column(String(64), ForeignKey("users.id"), nullable=False)
    influencer_id = Column(String(64), ForeignKey("influencers.id"), nullable=False)
    
    text = Column(Text, nullable=False)
    masked_text = Column(Text, nullable=False)
    
    # Moderation states: published, held_for_review, rejected, flagged_warning
    status = Column(String(32), default="published", index=True)
    
    # Classifier outputs: positive, neutral, criticism, abusive
    label = Column(String(32), default="neutral")
    abuse_severity = Column(String(32), default="none")  # none, mild, moderate, severe
    confidence = Column(Float, default=0.95)
    
    # Behavioral risk context
    author_risk_score = Column(Float, default=0.0)
    moderation_reason = Column(Text, default="")
    top_keywords = Column(String(256), default="")
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    post = relationship("Post", back_populates="comments")
    author = relationship("User", back_populates="comments")
    appeal = relationship("Appeal", back_populates="comment", uselist=False, cascade="all, delete-orphan")

class Like(Base):
    __tablename__ = "likes"

    id = Column(String(64), primary_key=True, default=generate_uuid)
    user_id = Column(String(64), ForeignKey("users.id"), nullable=False)
    post_id = Column(String(64), ForeignKey("posts.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="likes")
    post = relationship("Post", back_populates="likes")

class Follow(Base):
    __tablename__ = "follows"

    id = Column(String(64), primary_key=True, default=generate_uuid)
    follower_id = Column(String(64), ForeignKey("users.id"), nullable=False)
    influencer_id = Column(String(64), ForeignKey("influencers.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    follower = relationship("User")
    influencer = relationship("Influencer", back_populates="followers")

class ModerationAuditLog(Base):
    __tablename__ = "moderation_audit_logs"

    id = Column(String(64), primary_key=True, default=generate_uuid)
    comment_id = Column(String(64), nullable=True)
    target_user_id = Column(String(64), nullable=False)
    moderator_id = Column(String(64), nullable=False)
    action_taken = Column(String(64), nullable=False)  # approve, warn, mute_48h, slow_mode_15m, shadow_filter, block
    reason = Column(Text, nullable=False)
    previous_status = Column(String(32), default="")
    new_status = Column(String(32), default="")
    created_at = Column(DateTime, default=datetime.utcnow)

class Appeal(Base):
    __tablename__ = "appeals"

    id = Column(String(64), primary_key=True, default=generate_uuid)
    comment_id = Column(String(64), ForeignKey("comments.id"), nullable=False)
    user_id = Column(String(64), ForeignKey("users.id"), nullable=False)
    reason = Column(Text, nullable=False)
    status = Column(String(32), default="pending")  # pending, approved, rejected
    moderator_notes = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)

    comment = relationship("Comment", back_populates="appeal")
    user = relationship("User", back_populates="appeals")

class Notification(Base):
    __tablename__ = "notifications"

    id = Column(String(64), primary_key=True, default=generate_uuid)
    user_id = Column(String(64), ForeignKey("users.id"), nullable=False)
    title = Column(String(128), nullable=False)
    message = Column(Text, nullable=False)
    type = Column(String(32), default="info")  # warning, review_hold, approval, mute, cooldown
    is_read = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="notifications")

class FederatedRoundLog(Base):
    __tablename__ = "federated_round_logs"

    id = Column(String(64), primary_key=True, default=generate_uuid)
    round_number = Column(Integer, nullable=False)
    strategy = Column(String(64), default="FedAvg")
    secagg_enabled = Column(Boolean, default=False)
    dp_enabled = Column(Boolean, default=False)
    dp_noise_multiplier = Column(Float, default=0.0)
    dp_epsilon_spent = Column(Float, default=0.0)
    macro_f1 = Column(Float, default=0.0)
    accuracy = Column(Float, default=0.0)
    abuse_f1 = Column(Float, default=0.0)
    fpr_normal = Column(Float, default=0.0)
    comm_overhead_kb = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
