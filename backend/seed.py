"""
Database Seeder for EchoSafe Platform.
Pulls from synthetic datasets in data/ and creates demo accounts for all roles.
"""

import os
import sys
import json
import random
from datetime import datetime, timedelta
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.database import engine, SessionLocal, Base
from backend.models import (
    User, Influencer, Post, Comment, Like, Follow,
    ModerationAuditLog, Appeal, Notification, FederatedRoundLog
)
from backend.auth import hash_password
from backend.config import (
    INFLUENCERS_JSON_PATH, POSTS_CSV_PATH, COMMENTS_CSV_PATH,
    USERS_JSON_PATH, LIKES_CSV_PATH, FEDAVG_LOG_PATH
)
from modules.moderation_engine import mask_text_display

def seed_database():
    print("\n" + "=" * 80)
    print(" SEEDING ECHOSAFE DATABASE (SQLite)")
    print("=" * 80)

    # 1. Create fresh database schema
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    try:
        # 2. Seed Default Demo Accounts
        demo_users = [
            {
                "id": "demo_admin",
                "username": "admin",
                "email": "admin@echosafe.internal",
                "password": "password123",
                "role": "admin",
                "avatar_url": "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=150",
                "bio": "System Administrator & FL Research Lead"
            },
            {
                "id": "demo_mod",
                "username": "moderator_sarah",
                "email": "sarah@echosafe.internal",
                "password": "password123",
                "role": "moderator",
                "avatar_url": "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=150",
                "bio": "Lead Community Safety & Appeals Moderator"
            },
            {
                "id": "demo_pos",
                "username": "user_positive",
                "email": "positive@echosafe.internal",
                "password": "password123",
                "role": "follower",
                "avatar_url": "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=150",
                "bio": "Tech enthusiast and creative storyteller! ✨"
            },
            {
                "id": "demo_critic",
                "username": "user_critic",
                "email": "critic@echosafe.internal",
                "password": "password123",
                "role": "follower",
                "avatar_url": "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=150",
                "bio": "Constructive criticism & gadget analyst."
            },
            {
                "id": "demo_abuser",
                "username": "user_abuser",
                "email": "abuser@echosafe.internal",
                "password": "password123",
                "role": "follower",
                "avatar_url": "https://images.unsplash.com/photo-1522075469751-3a6694fb2f61?w=150",
                "bio": "Just speaking the cold hard truth."
            }
        ]

        user_objects = {}
        for du in demo_users:
            u = User(
                id=du["id"],
                username=du["username"],
                email=du["email"],
                hashed_password=hash_password(du["password"]),
                role=du["role"],
                avatar_url=du["avatar_url"],
                bio=du["bio"]
            )
            db.add(u)
            user_objects[u.id] = u
        db.commit()
        print(f" [OK] Seeded {len(demo_users)} core demo role accounts.")

        # 3. Seed Influencers from synthetic_influencers.json
        influencers_list = []
        if os.path.exists(INFLUENCERS_JSON_PATH):
            with open(INFLUENCERS_JSON_PATH, "r", encoding="utf-8") as f:
                influencers_list = json.load(f)

        creator_avatars = [
            "https://images.unsplash.com/photo-1566492031773-4f4e44671857?w=150",
            "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=150",
            "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=150",
            "https://images.unsplash.com/photo-1492562080023-ab3db95bfbce?w=150",
            "https://images.unsplash.com/photo-1517841905240-472988babdf9?w=150",
            "https://images.unsplash.com/photo-1539571696357-5a69c17a67c6?w=150",
            "https://images.unsplash.com/photo-1524504388940-b1c1722653e1?w=150",
            "https://images.unsplash.com/photo-1519085360753-af0119f7cbe7?w=150"
        ]

        inf_id_to_user_id = {}
        for idx, inf_data in enumerate(influencers_list):
            inf_id = inf_data["influencer_id"]
            # Create user account for influencer
            u_id = f"user_{inf_id}"
            clean_username = inf_data["name"].lower().replace(" ", "_")
            inf_user = User(
                id=u_id,
                username=clean_username,
                email=f"{clean_username}@echosafe.creator",
                hashed_password=hash_password("password123"),
                role="influencer",
                avatar_url=creator_avatars[idx % len(creator_avatars)],
                bio=f"Official {inf_data['niche']} creator on {inf_data['platform']}."
            )
            db.add(inf_user)
            db.flush()

            # Health calculation defaults
            health = 85.0 if idx != 5 else 73.5
            vel = 0.0 if idx != 5 else 0.113
            is_raid = True if idx == 5 else False
            policy = "Standard moderation active." if idx != 5 else "Enable Slow-Mode (15m delay) on controversial posts."

            inf_profile = Influencer(
                id=inf_id,
                user_id=u_id,
                name=inf_data["name"],
                niche=inf_data["niche"],
                platform=inf_data["platform"],
                follower_count=inf_data["follower_count"],
                community_health_score=health,
                toxicity_velocity=vel,
                is_raid_detected=is_raid,
                protective_recommendation=policy
            )
            db.add(inf_profile)
            inf_id_to_user_id[inf_id] = u_id

        # Also create convenient creator shortcuts
        demo_creators = [
            ("creator_gaming", "inf_0"),
            ("creator_comedy", "inf_5")
        ]
        for cname, target_inf in demo_creators:
            shortcut_user = User(
                id=f"shortcut_{cname}",
                username=cname,
                email=f"{cname}@echosafe.demo",
                hashed_password=hash_password("password123"),
                role="influencer",
                avatar_url="https://images.unsplash.com/photo-1566492031773-4f4e44671857?w=150",
                bio="Verified EchoSafe Demo Creator"
            )
            db.add(shortcut_user)
        db.commit()
        print(f" [OK] Seeded {len(influencers_list)} influencer communities and creator profiles.")

        # 4. Seed Posts from synthetic_posts.csv
        posts_df = pd.DataFrame()
        if os.path.exists(POSTS_CSV_PATH):
            posts_df = pd.read_csv(POSTS_CSV_PATH)

        sample_media = {
            "reel": "https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?w=600",
            "image": "https://images.unsplash.com/photo-1518770660439-4636190af475?w=600",
            "video": "https://images.unsplash.com/photo-1536240478700-b869070f9279?w=600",
            "opinion": "https://images.unsplash.com/photo-1457369804613-52c61a468e7d?w=600",
            "text": "https://images.unsplash.com/photo-1455390582262-044cdead277a?w=600",
            "live": "https://images.unsplash.com/photo-1516251193007-45ef944ab0c6?w=600"
        }

        posts_added = 0
        for _, row in posts_df.iterrows():
            ptype = str(row.get("post_type", "image")).lower()
            p = Post(
                id=str(row["post_id"]),
                influencer_id=str(row["influencer_id"]),
                post_type=ptype,
                caption=str(row["caption"]),
                media_url=sample_media.get(ptype, sample_media["image"]),
                created_at=datetime.fromisoformat(str(row["timestamp"])),
                likes_count=random.randint(45, 1200),
                comments_count=random.randint(5, 140)
            )
            db.add(p)
            posts_added += 1
        db.commit()
        print(f" [OK] Seeded {posts_added} posts across all post types (reels, images, text, opinions, live).")

        # 5. Seed Synthetic Users from synthetic_users.json
        users_list = []
        if os.path.exists(USERS_JSON_PATH):
            with open(USERS_JSON_PATH, "r", encoding="utf-8") as f:
                users_list = json.load(f)

        users_added = 0
        for u_data in users_list[:400]:  # Seed top 400 active community members for swift DB performance
            u_id = str(u_data["user_id"])
            if db.query(User).filter(User.id == u_id).first():
                continue
            arch = u_data.get("archetype", "normal_positive")
            u = User(
                id=u_id,
                username=f"{u_id}_{arch[:6]}",
                email=f"{u_id}@echosafe.user",
                hashed_password=hash_password("password123"),
                role="follower",
                avatar_url=f"https://api.dicebear.com/7.x/bottts/svg?seed={u_id}",
                archetype=arch
            )
            db.add(u)
            users_added += 1
        db.commit()
        print(f" [OK] Seeded {users_added} active synthetic users with behavioral archetypes.")

        # 6. Seed Comments from synthetic_comments.csv
        comments_df = pd.DataFrame()
        if os.path.exists(COMMENTS_CSV_PATH):
            comments_df = pd.read_csv(COMMENTS_CSV_PATH)

        comments_added = 0
        flagged_count = 0
        for _, row in comments_df.head(800).iterrows():  # Seed first 800 for high-speed local demo
            c_label = str(row["label"])
            c_sev = str(row["abuse_severity"])
            c_text = str(row["text"])
            c_masked = mask_text_display(c_text)

            # Determine initial status
            if c_label == "abusive-vulgar":
                status = "held_for_review" if c_sev in ["mild", "moderate"] else "rejected"
                flagged_count += 1
                reason = f"Automated filter: hostile/vulgar terminology detected (Severity: {c_sev})."
            elif c_label == "negative-but-not-abusive":
                status = "published"
                reason = "Constructive criticism allowed under Community Free Speech guidelines."
            else:
                status = "published"
                reason = "Benign community interaction."

            c = Comment(
                id=str(row["comment_id"]),
                post_id=str(row["post_id"]),
                user_id=str(row["user_id"]),
                influencer_id=str(row["influencer_id"]),
                text=c_text,
                masked_text=c_masked,
                status=status,
                label=c_label,
                abuse_severity=c_sev,
                confidence=round(random.uniform(0.91, 0.99), 3),
                author_risk_score=0.72 if c_label == "abusive-vulgar" else 0.05,
                moderation_reason=reason,
                created_at=datetime.fromisoformat(str(row["timestamp"]))
            )
            db.add(c)
            comments_added += 1

        db.commit()
        print(f" [OK] Seeded {comments_added} comments ({flagged_count} queued for moderation).")

        # 7. Seed Initial Notifications & Appeals for Demo
        # Sample held comment for demo_abuser
        test_cmt = db.query(Comment).filter(Comment.status == "held_for_review").first()
        if test_cmt:
            # Notification to user
            notif = Notification(
                user_id="demo_abuser",
                title="Comment Held for Review",
                message=f"Your comment on post {test_cmt.post_id} was held for human moderator review. Reason: abusive language detected.",
                type="review_hold"
            )
            db.add(notif)

            # Sample appeal from user
            appeal = Appeal(
                comment_id=test_cmt.id,
                user_id="demo_abuser",
                reason="I was using internet slang for emphasis, not trying to insult the creator directly.",
                status="pending"
            )
            db.add(appeal)

            # Sample audit log
            audit = ModerationAuditLog(
                comment_id=test_cmt.id,
                target_user_id=test_cmt.user_id,
                moderator_id="demo_mod",
                action_taken="auto_quarantine",
                reason="High abuse probability from automated federated model.",
                previous_status="pending",
                new_status="held_for_review"
            )
            db.add(audit)

        # 8. Seed Initial Federated Round Logs from fedavg_log.csv
        if os.path.exists(FEDAVG_LOG_PATH):
            fl_df = pd.read_csv(FEDAVG_LOG_PATH)
            for _, r in fl_df.iterrows():
                f_log = FederatedRoundLog(
                    round_number=int(r["round"]),
                    strategy="FedAvg",
                    secagg_enabled=False,
                    dp_enabled=False,
                    macro_f1=float(r["macro_f1"]),
                    accuracy=float(r["accuracy"]),
                    abuse_f1=float(r["abuse_f1"]),
                    fpr_normal=float(r["fpr_normal"]),
                    comm_overhead_kb=float(r["comm_kb"])
                )
                db.add(f_log)

        db.commit()
        print(" [OK] Seeded demo notifications, appeals, moderation audit logs, and FL round history.")
        print("=" * 80)
        print(" DATABASE SEEDING COMPLETED SUCCESSFULLY!")
        print("=" * 80 + "\n")

    except Exception as e:
        db.rollback()
        print(f" [ERROR] Seeding failed: {e}")
        raise e
    finally:
        db.close()

if __name__ == "__main__":
    seed_database()
