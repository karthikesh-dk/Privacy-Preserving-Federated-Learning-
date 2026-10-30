"""
EchoSafe Backend Configuration.
"""

import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "experiments", "results")

# Database configuration (SQLite)
DB_PATH = os.path.join(BASE_DIR, "echosafe.db")
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

# Security / JWT configuration
SECRET_KEY = os.environ.get("ECHOSAFE_SECRET_KEY", "echosafe_federated_secret_key_2026_super_secure")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# ML Model Paths
GLOBAL_MODEL_PATH = os.path.join(RESULTS_DIR, "global_fedavg_model.npz")
USER_PROFILES_PATH = os.path.join(RESULTS_DIR, "user_behaviour_profiles.csv")
COMMENTS_CSV_PATH = os.path.join(DATA_DIR, "synthetic_comments.csv")
POSTS_CSV_PATH = os.path.join(DATA_DIR, "synthetic_posts.csv")
INFLUENCERS_JSON_PATH = os.path.join(DATA_DIR, "synthetic_influencers.json")
USERS_JSON_PATH = os.path.join(DATA_DIR, "synthetic_users.json")
LIKES_CSV_PATH = os.path.join(DATA_DIR, "synthetic_likes_shares.csv")
ABLATION_JSON_PATH = os.path.join(RESULTS_DIR, "ablation_benchmark.json")
FEDAVG_LOG_PATH = os.path.join(RESULTS_DIR, "fedavg_log.csv")
