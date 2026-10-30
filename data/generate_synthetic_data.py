"""
Synthetic Social Media Abuse & User Behaviour Dataset Generator.

Generates:
1. 8-10 Influencers (Clients) with distinct niches, platforms, language mixes, and abuse rates.
2. ~2,000 Users mapped to 7 distinct behavioural archetypes.
3. Posts across 6 post types (image, video, reel, text, opinion, live).
4. Comments with 4 labels (positive, neutral, negative-but-not-abusive, abusive-vulgar),
   abuse severity, template IDs, obfuscations, and realism guards (label noise, sarcasm, positive profanity).
5. Likes & Shares interaction logs for behavioural profiling.
"""

import os
import sys
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

# Ensure console supports UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def mask_text(text: str, max_chars: int = 50) -> str:
    """Mask or truncate text for safe terminal display."""
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    # Replace common vulgar roots for display safety
    masked = text
    for w in ["fuck", "shit", "bitch", "bastard", "asshole"]:
        masked = masked.replace(w, w[0] + "***")
        masked = masked.replace(w.capitalize(), w[0].capitalize() + "***")
    return masked

def load_lexicons(lexicon_dir: str = "data/lexicons") -> Dict[str, dict]:
    """Load JSON lexicon dictionaries."""
    lexicons = {}
    for fname in ["positive.json", "neutral.json", "criticism.json", "abusive_mild.json", 
                  "abusive_severe.json", "code_mixed.json", "obfuscations.json"]:
        fpath = os.path.join(lexicon_dir, fname)
        if os.path.exists(fpath):
            with open(fpath, "r", encoding="utf-8") as f:
                lexicons[fname.replace(".json", "")] = json.load(f)
    return lexicons

def apply_obfuscation(text: str, prob: float = 0.25) -> str:
    """Randomly apply l33tspeak substitutions or misspellings."""
    if random.random() > prob:
        return text
    sub_map = {"a": "@", "s": "$", "i": "1", "e": "3", "o": "0"}
    chars = list(text)
    for i, c in enumerate(chars):
        if c.lower() in sub_map and random.random() < 0.4:
            chars[i] = sub_map[c.lower()]
    return "".join(chars)

def generate_synthetic_data(
    num_influencers: int = 8,
    num_users: int = 2000,
    posts_per_influencer: int = 15,
    seed: int = 42,
    output_dir: str = "data"
) -> Dict[str, Any]:
    """
    Generate synthetic dataset for privacy-preserving federated abuse moderation.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    lex = load_lexicons()

    # 1. Define Influencers (Clients)
    influencer_templates = [
        {"niche": "Competitive Gaming", "platform": "YouTube", "abuse_rate": 0.18, "lang": "en"},
        {"niche": "Beauty & Fashion", "platform": "Instagram", "abuse_rate": 0.14, "lang": "en"},
        {"niche": "Political Commentary", "platform": "X", "abuse_rate": 0.22, "lang": "en"},
        {"niche": "Consumer Tech & Gadgets", "platform": "YouTube", "abuse_rate": 0.09, "lang": "en"},
        {"niche": "Fitness & Bodybuilding", "platform": "Instagram", "abuse_rate": 0.12, "lang": "en"},
        {"niche": "Regional Comedy & Skits", "platform": "Instagram", "abuse_rate": 0.16, "lang": "hi-en"},
        {"niche": "Bengaluru Tech & Startups", "platform": "X", "abuse_rate": 0.11, "lang": "kn-en"},
        {"niche": "Lifestyle & Daily Vlogs", "platform": "TikTok", "abuse_rate": 0.13, "lang": "en"}
    ]

    influencers = []
    for i in range(num_influencers):
        cfg = influencer_templates[i % len(influencer_templates)]
        inf_id = f"inf_{i}"
        followers = int(np.random.choice([25000, 85000, 220000, 650000, 1500000]))
        influencers.append({
            "influencer_id": inf_id,
            "client_id": i,
            "name": f"Creator_{i}_{cfg['niche'].split()[0]}",
            "niche": cfg["niche"],
            "platform": cfg["platform"],
            "follower_count": followers,
            "target_abuse_rate": cfg["abuse_rate"],
            "primary_language": cfg["lang"]
        })

    # 2. Define User Behaviour Archetypes
    # 7 archetypes as required by synopsis
    archetype_distribution = [
        ("normal_positive", 0.40),
        ("occasional_critic", 0.15),
        ("frequent_commenter", 0.15),
        ("repeat_abuser", 0.10),
        ("escalating_abuser", 0.08),
        ("burst_spammer", 0.07),
        ("lurker", 0.05)
    ]
    archetype_choices, archetype_probs = zip(*archetype_distribution)

    users = []
    user_archetype_map = {}
    for u in range(num_users):
        u_id = f"user_{u:04d}"
        arch = np.random.choice(archetype_choices, p=archetype_probs)
        user_archetype_map[u_id] = arch
        users.append({
            "user_id": u_id,
            "archetype": arch,
            "account_age_days": random.randint(30, 1200),
            "primary_platform": random.choice(["Instagram", "YouTube", "X", "TikTok", "Facebook"])
        })

    # 3. Generate Posts per Influencer
    post_types = ["image", "video", "reel", "text", "opinion", "live"]
    posts = []
    base_time = datetime(2026, 8, 1, 10, 0, 0)
    post_id_counter = 1

    for inf in influencers:
        for p in range(posts_per_influencer):
            post_id = f"post_{post_id_counter:04d}"
            post_id_counter += 1
            ptype = random.choice(post_types)
            p_time = base_time + timedelta(days=p * 2, hours=random.randint(1, 12))
            posts.append({
                "post_id": post_id,
                "influencer_id": inf["influencer_id"],
                "platform": inf["platform"],
                "post_type": ptype,
                "timestamp": p_time.isoformat(),
                "caption": f"Updates on {inf['niche']} #{ptype} insights."
            })

    # 4. Generate Comments & Likes/Shares with Behavioural Realism
    comments = []
    likes_shares = []
    comment_counter = 1
    event_counter = 1

    # Text template pools
    pos_phrases = lex.get("positive", {}).get("phrases", ["Great video!"])
    pos_adj = lex.get("positive", {}).get("adjectives", ["awesome"])
    pos_emojis = lex.get("positive", {}).get("emojis", ["🔥"])

    neu_phrases = lex.get("neutral", {}).get("phrases", ["Interesting."])
    neu_inq = lex.get("neutral", {}).get("inquiries", ["What is this?"])

    crit_phrases = lex.get("criticism", {}).get("phrases", ["Could be better."])
    crit_disc = lex.get("criticism", {}).get("disclaimers", ["Constructive feedback."])

    abu_mild = lex.get("abusive_mild", {}).get("phrases", ["This is stupid."])
    abu_mild_insults = lex.get("abusive_mild", {}).get("insults", ["clown"])

    abu_sev = lex.get("abusive_severe", {}).get("phrases", ["Shut up garbage."])
    abu_sev_prof = lex.get("abusive_severe", {}).get("profanities", ["crap"])

    pos_prof = lex.get("obfuscations", {}).get("positive_profanity_emphasis", ["Damn good!"])
    sarc_crit = lex.get("obfuscations", {}).get("sarcastic_criticism", ["Oh wow... /s"])

    code_pos = lex.get("code_mixed", {}).get("positive_slang", ["Sakath video guru!"])
    code_neu = lex.get("code_mixed", {}).get("neutral_slang", ["Guru next yavaga?"])
    code_abu = lex.get("code_mixed", {}).get("abusive_slang", ["Dabba content guru."])

    # Assign active users to influencers (community overlap)
    user_ids = [u["user_id"] for u in users]

    for post in posts:
        inf_id = post["influencer_id"]
        inf_meta = next(inf for inf in influencers if inf["influencer_id"] == inf_id)
        post_timestamp = datetime.fromisoformat(post["timestamp"])
        
        # Select active cohort for this post
        num_post_comments = random.randint(35, 65)
        commenters = random.sample(user_ids, min(num_post_comments, len(user_ids)))

        for u_id in commenters:
            arch = user_archetype_map[u_id]
            if arch == "lurker":
                # Lurker rarely comments, mostly likes/shares
                if random.random() < 0.75:
                    likes_shares.append({
                        "event_id": f"evt_{event_counter:06d}",
                        "user_id": u_id,
                        "post_id": post["post_id"],
                        "influencer_id": inf_id,
                        "post_type": post["post_type"],
                        "event_type": "like",
                        "timestamp": post["timestamp"]
                    })
                    event_counter += 1
                continue

            # Generate likes/shares for active users
            if random.random() < 0.60:
                likes_shares.append({
                    "event_id": f"evt_{event_counter:06d}",
                    "user_id": u_id,
                    "post_id": post["post_id"],
                    "influencer_id": inf_id,
                    "post_type": post["post_type"],
                    "event_type": random.choice(["like", "share"]),
                    "timestamp": post["timestamp"]
                })
                event_counter += 1

            # Determine comment frequency and burst behavior
            num_comments_by_user = 1
            if arch == "burst_spammer" and random.random() < 0.35:
                num_comments_by_user = random.randint(3, 5)  # rapid burst

            for burst_idx in range(num_comments_by_user):
                time_delta_mins = burst_idx * random.randint(1, 4) if burst_idx > 0 else random.randint(5, 720)
                c_time = post_timestamp + timedelta(minutes=time_delta_mins)

                # Determine comment sentiment based on archetype and influencer baseline
                label = "positive"
                severity = "none"
                text = ""
                template_id = "T_POS_01"

                # Archetype logic
                if arch == "normal_positive":
                    if random.random() < 0.85:
                        label = "positive"
                        template_id = f"T_POS_{random.randint(1, 10):02d}"
                        text = random.choice(pos_phrases) + " " + random.choice(pos_emojis)
                    else:
                        label = "neutral"
                        template_id = f"T_NEU_{random.randint(1, 10):02d}"
                        text = random.choice(neu_phrases)

                elif arch == "occasional_critic":
                    if random.random() < 0.60:
                        label = "negative-but-not-abusive"
                        template_id = f"T_CRIT_{random.randint(1, 10):02d}"
                        text = random.choice(crit_phrases) + ". " + random.choice(crit_disc)
                    else:
                        label = "neutral"
                        template_id = f"T_NEU_{random.randint(1, 10):02d}"
                        text = random.choice(neu_phrases)

                elif arch == "frequent_commenter":
                    p_val = random.random()
                    if p_val < 0.65:
                        label = "positive"
                        template_id = f"T_POS_{random.randint(1, 10):02d}"
                        text = random.choice(pos_phrases)
                    elif p_val < 0.90:
                        label = "neutral"
                        template_id = f"T_NEU_{random.randint(1, 10):02d}"
                        text = random.choice(neu_inq)
                    else:
                        label = "negative-but-not-abusive"
                        template_id = f"T_CRIT_{random.randint(1, 10):02d}"
                        text = random.choice(crit_phrases)

                elif arch == "repeat_abuser":
                    label = "abusive-vulgar"
                    if random.random() < 0.45:
                        severity = "severe"
                        template_id = f"T_ABU_SEV_{random.randint(1, 10):02d}"
                        text = random.choice(abu_sev)
                    else:
                        severity = "moderate"
                        template_id = f"T_ABU_MLD_{random.randint(1, 10):02d}"
                        text = random.choice(abu_mild)

                elif arch == "escalating_abuser":
                    # Escalates over post chronological progress
                    post_progress = posts.index(post) / max(1, len(posts))
                    if post_progress < 0.35:
                        label = "neutral"
                        template_id = f"T_NEU_{random.randint(1, 10):02d}"
                        text = random.choice(neu_phrases)
                    elif post_progress < 0.70:
                        label = "negative-but-not-abusive"
                        template_id = f"T_CRIT_{random.randint(1, 10):02d}"
                        text = random.choice(crit_phrases)
                    else:
                        label = "abusive-vulgar"
                        severity = "severe" if post_progress > 0.85 else "moderate"
                        template_id = f"T_ABU_ESC_{random.randint(1, 10):02d}"
                        text = random.choice(abu_sev if severity == "severe" else abu_mild)

                elif arch == "burst_spammer":
                    label = "abusive-vulgar" if random.random() < 0.75 else "negative-but-not-abusive"
                    severity = "moderate" if label == "abusive-vulgar" else "none"
                    template_id = f"T_BURST_{random.randint(1, 5):02d}"
                    text = random.choice(abu_mild if label == "abusive-vulgar" else crit_phrases)

                # Realism Guard 1: Positive Profanity (Profanity as emphasis -> POSITIVE label)
                if label == "positive" and random.random() < 0.08:
                    text = random.choice(pos_prof)
                    template_id = "T_POS_PROFANITY_EMPHASIS"
                    severity = "none"

                # Realism Guard 2: Sarcastic Criticism (Sounds polite, actually negative)
                if label == "negative-but-not-abusive" and random.random() < 0.12:
                    text = random.choice(sarc_crit)
                    template_id = "T_SARCASM_CRIT"

                # Realism Guard 3: Code-mixed slang injection for specific language influencers
                if inf_meta["primary_language"] in ["hi-en", "kn-en"] and random.random() < 0.40:
                    if label == "positive":
                        text = random.choice(code_pos)
                    elif label == "abusive-vulgar":
                        text = random.choice(code_abu)
                    else:
                        text = random.choice(code_neu)

                # Realism Guard 4: Obfuscations (l33tspeak, typos)
                if label == "abusive-vulgar":
                    text = apply_obfuscation(text, prob=0.30)

                # Realism Guard 5: Label noise (~4%)
                if random.random() < 0.04:
                    orig_label = label
                    label = random.choice(["positive", "neutral", "negative-but-not-abusive", "abusive-vulgar"])
                    if label == "abusive-vulgar" and severity == "none":
                        severity = "mild"
                    elif label != "abusive-vulgar":
                        severity = "none"

                comments.append({
                    "comment_id": f"cmt_{comment_counter:06d}",
                    "user_id": u_id,
                    "influencer_id": inf_id,
                    "post_id": post["post_id"],
                    "platform": post["platform"],
                    "timestamp": c_time.isoformat(),
                    "text": text,
                    "label": label,
                    "abuse_severity": severity,
                    "language": inf_meta["primary_language"],
                    "likes_on_comment": random.randint(0, 150) if label == "positive" else random.randint(0, 8),
                    "template_id": template_id
                })
                comment_counter += 1

    # Convert to DataFrames
    comments_df = pd.DataFrame(comments)
    users_df = pd.DataFrame(users)
    influencers_df = pd.DataFrame(influencers)
    posts_df = pd.DataFrame(posts)
    likes_shares_df = pd.DataFrame(likes_shares)

    # Save to disk
    comments_df.to_csv(os.path.join(output_dir, "synthetic_comments.csv"), index=False)
    posts_df.to_csv(os.path.join(output_dir, "synthetic_posts.csv"), index=False)
    likes_shares_df.to_csv(os.path.join(output_dir, "synthetic_likes_shares.csv"), index=False)
    
    with open(os.path.join(output_dir, "synthetic_users.json"), "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)
    with open(os.path.join(output_dir, "synthetic_influencers.json"), "w", encoding="utf-8") as f:
        json.dump(influencers, f, indent=2)

    print("\n" + "=" * 80)
    print(" SYNTHETIC DATASET GENERATION COMPLETE")
    print("=" * 80)
    print(f" Influencers Generated : {len(influencers)}")
    print(f" Users Generated       : {len(users)}")
    print(f" Posts Generated       : {len(posts)}")
    print(f" Comments Generated    : {len(comments)}")
    print(f" Interaction Events    : {len(likes_shares)}")
    print("\n--- Class Label Distribution ---")
    print(comments_df["label"].value_counts(normalize=True).mul(100).round(2).to_string())
    print("\n--- Abuse Severity Distribution ---")
    print(comments_df["abuse_severity"].value_counts(normalize=True).mul(100).round(2).to_string())
    print("\n--- Non-IID Skew: Abuse Rate per Influencer (%) ---")
    inf_skew = comments_df.groupby("influencer_id")["label"].apply(lambda s: (s == "abusive-vulgar").mean() * 100)
    print(inf_skew.round(2).to_string())
    
    print("\n--- Safe Masked Sample Comments ---")
    for _, row in comments_df.sample(5, random_state=42).iterrows():
        sample_str = mask_text(str(row['text']))
        # Clean for console display if needed
        clean_display = sample_str.encode("ascii", "replace").decode("ascii")
        print(f" [{row['label']:<24}] {clean_display}")
    print("=" * 80 + "\n")

    return {
        "comments": comments_df,
        "users": users_df,
        "influencers": influencers_df,
        "posts": posts_df,
        "likes_shares": likes_shares_df
    }

if __name__ == "__main__":
    generate_synthetic_data()
