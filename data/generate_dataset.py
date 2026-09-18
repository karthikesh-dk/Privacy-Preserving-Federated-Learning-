"""
Synthetic Dataset Generator for Social Media Influencer Engagement Prediction.

Generates realistic post-level engagement data across multiple creators/clients
with non-IID characteristics (skewed content types and engagement patterns).
"""

import os
import json
import numpy as np
import pandas as pd

def generate_social_media_dataset(
    num_samples: int = 8000,
    num_creators: int = 8,
    random_seed: int = 42,
    output_path: str = "data/social_media_engagement.csv"
) -> pd.DataFrame:
    """
    Generate synthetic social media engagement dataset.
    
    Args:
        num_samples: Total number of post records to generate.
        num_creators: Number of client creators (default: 8).
        random_seed: Random seed for reproducibility.
        output_path: Filepath where the CSV will be saved.
        
    Returns:
        pd.DataFrame containing the generated dataset.
    """
    np.random.seed(random_seed)
    
    content_types = ["image", "video", "text", "reel"]
    
    # Skew content distribution per creator to ensure realistic Non-IID properties
    # Each creator specializes in a distinct content format
    creator_content_priors = {
        0: [0.10, 0.10, 0.10, 0.70],  # Creator 0: Heavy Reel creator
        1: [0.70, 0.10, 0.10, 0.10],  # Creator 1: Heavy Image creator (photographer)
        2: [0.10, 0.70, 0.10, 0.10],  # Creator 2: Long-form Video creator
        3: [0.10, 0.10, 0.70, 0.10],  # Creator 3: Text/thought-leadership creator
        4: [0.05, 0.45, 0.05, 0.45],  # Creator 4: Video + Reel hybrid
        5: [0.50, 0.05, 0.40, 0.05],  # Creator 5: Image + Text hybrid
        6: [0.20, 0.10, 0.10, 0.60],  # Creator 6: Lifestyle Reel creator
        7: [0.10, 0.20, 0.60, 0.10],  # Creator 7: Educational Text creator
    }
    
    # Creator follower tiers and base engagement rates
    creator_profiles = {}
    for c_id in range(num_creators):
        base_followers = int(np.random.choice([15000, 45000, 120000, 350000, 750000]))
        base_engagement_rate = float(np.random.uniform(0.02, 0.08))
        creator_profiles[c_id] = {
            "follower_count": base_followers,
            "past_avg_engagement": base_engagement_rate
        }
        
    posts = []
    # Distribute samples evenly across creators
    samples_per_creator = num_samples // num_creators
    
    post_counter = 1
    for c_id in range(num_creators):
        c_meta = creator_profiles[c_id]
        p_dist = creator_content_priors[c_id]
        
        for _ in range(samples_per_creator):
            post_id = f"POST_{post_counter:05d}"
            post_counter += 1
            
            # Content Type according to creator's skewed distribution
            ctype = np.random.choice(content_types, p=p_dist)
            
            # Context features based on content type
            if ctype == "reel":
                caption_len = int(np.random.normal(80, 25))
                hashtag_count = int(np.clip(np.random.normal(7, 3), 1, 15))
                watch_time = float(np.clip(np.random.exponential(18), 3, 60))
                type_multiplier = 1.4
            elif ctype == "video":
                caption_len = int(np.random.normal(180, 50))
                hashtag_count = int(np.clip(np.random.normal(5, 2), 0, 12))
                watch_time = float(np.clip(np.random.exponential(45), 10, 300))
                type_multiplier = 1.2
            elif ctype == "image":
                caption_len = int(np.random.normal(120, 40))
                hashtag_count = int(np.clip(np.random.normal(8, 3), 1, 20))
                watch_time = float(np.clip(np.random.normal(6, 2), 1, 15))
                type_multiplier = 1.0
            else:  # text
                caption_len = int(np.random.normal(250, 70))
                hashtag_count = int(np.clip(np.random.normal(3, 2), 0, 8))
                watch_time = float(np.clip(np.random.normal(10, 4), 2, 30))
                type_multiplier = 0.85
                
            caption_len = max(10, caption_len)
            
            # Timing
            posting_hour = int(np.random.choice(range(24)))
            day_of_week = int(np.random.choice(range(7)))
            
            # Peak hours boost (e.g., 18:00 - 22:00)
            time_factor = 1.25 if 17 <= posting_hour <= 22 else 0.9
            # Weekend boost
            day_factor = 1.15 if day_of_week in [5, 6] else 1.0
            
            # 16-dimensional dense embedding (simulates multimodal semantic features)
            # Embedding cluster varies slightly per content type + random Gaussian
            embedding_center = np.zeros(16)
            embedding_center[content_types.index(ctype) * 4 : (content_types.index(ctype) + 1) * 4] = 1.0
            embedding = embedding_center + np.random.normal(0, 0.35, 16)
            embedding = embedding / np.linalg.norm(embedding)  # unit norm
            
            # Engagement calculations
            followers = c_meta["follower_count"]
            past_eng = c_meta["past_avg_engagement"]
            
            # Latent post quality factor ~ LogNormal
            latent_quality = np.random.lognormal(mean=0.0, sigma=0.45)
            
            expected_rate = past_eng * type_multiplier * time_factor * day_factor * latent_quality
            expected_rate = np.clip(expected_rate, 0.005, 0.40)
            
            # Simulated raw metrics
            likes = int(np.random.poisson(followers * expected_rate * 0.65))
            comments = int(np.random.poisson(likes * 0.08))
            shares = int(np.random.poisson(likes * (0.15 if ctype in ['reel', 'video'] else 0.04)))
            saves = int(np.random.poisson(likes * (0.18 if ctype in ['image', 'reel'] else 0.05)))
            clicks = int(np.random.poisson(likes * 0.12))
            
            # Overall Engagement Score formula:
            # Weighted interaction rate relative to audience size, normalized
            interaction_mass = (likes * 1.0) + (comments * 3.0) + (shares * 4.0) + (saves * 3.5) + (clicks * 1.5)
            raw_engagement_rate = interaction_mass / max(followers, 1000)
            
            posts.append({
                "post_id": post_id,
                "creator_id": f"creator_{c_id}",
                "client_id": c_id,
                "content_type": ctype,
                "caption_length": caption_len,
                "hashtag_count": hashtag_count,
                "posting_hour": posting_hour,
                "day_of_week": day_of_week,
                "follower_count": followers,
                "past_avg_engagement": round(past_eng, 4),
                "likes": likes,
                "comments": comments,
                "shares": shares,
                "saves": saves,
                "clicks": clicks,
                "watch_time": round(watch_time, 2),
                "raw_engagement_rate": raw_engagement_rate,
                "content_embedding": json.dumps(embedding.tolist())
            })
            
    df = pd.DataFrame(posts)
    
    # Min-Max normalize engagement rate into a clean continuous engagement_score in [0, 1]
    p01 = df["raw_engagement_rate"].quantile(0.01)
    p99 = df["raw_engagement_rate"].quantile(0.99)
    clipped_rate = df["raw_engagement_rate"].clip(lower=p01, upper=p99)
    df["engagement_score"] = (clipped_rate - p01) / (p99 - p01 + 1e-8)
    df["engagement_score"] = df["engagement_score"].round(4)
    
    # Also define 3 engagement tiers for classification if needed: 0: Low, 1: Medium, 2: High
    df["engagement_tier"] = pd.qcut(df["engagement_score"], q=3, labels=[0, 1, 2]).astype(int)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset generated successfully with {len(df)} records saved to {output_path}")
    return df

if __name__ == "__main__":
    generate_social_media_dataset()
