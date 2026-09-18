"""
Module 1: Data Preparation & Client Partitioning.

Handles feature engineering (cyclic temporal encoding, one-hot encoding, feature scaling),
held-out centralized 80/20 splitting, and non-IID client partitioning across 8 creators.
"""

import os
import json
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "hour_sin", "hour_cos",
    "day_sin", "day_cos",
    "content_type_image", "content_type_reel", "content_type_text", "content_type_video",
    "caption_length", "hashtag_count", "follower_count", "past_avg_engagement", "watch_time"
]

class EngagementDataset(Dataset):
    """PyTorch Dataset wrapper for engagement prediction features and target."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class DataPreprocessor:
    """
    Cleans, encodes, scales social media engagement data, and provides
    both centralized and non-IID federated splits.
    """
    def __init__(self, data_path: str = "data/social_media_engagement.csv"):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.raw_df: Optional[pd.DataFrame] = None
        self.processed_df: Optional[pd.DataFrame] = None
        self.feature_columns = FEATURE_COLUMNS

    def load_and_preprocess(self) -> pd.DataFrame:
        """Load CSV and apply feature transformations."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found at {self.data_path}. Run generate_dataset.py first.")
        
        df = pd.read_csv(self.data_path)
        self.raw_df = df.copy()

        # 1. Cyclic encoding for temporal features
        # Posting hour (0-23)
        df["hour_sin"] = np.sin(2 * np.pi * df["posting_hour"] / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * df["posting_hour"] / 24.0)
        # Day of week (0-6)
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)

        # 2. One-hot encode categorical content_type
        content_dummies = pd.get_dummies(df["content_type"], prefix="content_type", dtype=float)
        for col in ["content_type_image", "content_type_reel", "content_type_text", "content_type_video"]:
            if col not in content_dummies.columns:
                content_dummies[col] = 0.0
        df = pd.concat([df, content_dummies], axis=1)

        # 3. Standardize continuous numerical features
        num_cols = ["caption_length", "hashtag_count", "follower_count", "past_avg_engagement", "watch_time"]
        df[num_cols] = self.scaler.fit_transform(df[num_cols])

        self.processed_df = df
        return df

    def get_centralized_data(self, test_size: float = 0.20, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Produce a centralized 80/20 train/test split on the entire dataset.
        
        Returns:
            X_train, y_train, X_test, y_test
        """
        if self.processed_df is None:
            self.load_and_preprocess()

        np.random.seed(seed)
        n = len(self.processed_df)
        indices = np.arange(n)
        np.random.shuffle(indices)

        split = int(n * (1 - test_size))
        train_idx = indices[:split]
        test_idx = indices[split:]

        X = self.processed_df[self.feature_columns].values.astype(np.float32)
        y = self.processed_df["engagement_score"].values.astype(np.float32)

        return X[train_idx], y[train_idx], X[test_idx], y[test_idx]

    def get_client_partitions(self, test_size: float = 0.20, seed: int = 42) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Partition dataset across clients (creators) into non-IID private subsets.
        Each client keeps an 80/20 local train/test split.
        
        Returns:
            dict mapping client_id -> {'X_train': ..., 'y_train': ..., 'X_test': ..., 'y_test': ...}
        """
        if self.processed_df is None:
            self.load_and_preprocess()

        clients = {}
        unique_clients = sorted(self.processed_df["client_id"].unique())

        np.random.seed(seed)
        for c_id in unique_clients:
            c_mask = self.processed_df["client_id"] == c_id
            c_df = self.processed_df[c_mask]

            n = len(c_df)
            indices = np.arange(n)
            np.random.shuffle(indices)

            split = int(n * (1 - test_size))
            train_idx = indices[:split]
            test_idx = indices[split:]

            X = c_df[self.feature_columns].values.astype(np.float32)
            y = c_df["engagement_score"].values.astype(np.float32)

            clients[int(c_id)] = {
                "X_train": X[train_idx],
                "y_train": y[train_idx],
                "X_test": X[test_idx],
                "y_test": y[test_idx],
                "num_train": len(train_idx),
                "num_test": len(test_idx)
            }

        return clients

    def get_client_loaders(
        self,
        client_id: int,
        batch_size: int = 32,
        test_size: float = 0.20,
        seed: int = 42
    ) -> Tuple[DataLoader, DataLoader]:
        """Return PyTorch DataLoaders for a specific client."""
        client_data = self.get_client_partitions(test_size=test_size, seed=seed)[client_id]
        
        train_ds = EngagementDataset(client_data["X_train"], client_data["y_train"])
        test_ds = EngagementDataset(client_data["X_test"], client_data["y_test"])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader

    def print_data_distribution(self):
        """Display verification summary showing non-IID content-type distributions across clients."""
        if self.raw_df is None:
            if not os.path.exists(self.data_path):
                raise FileNotFoundError("Dataset not generated yet.")
            self.raw_df = pd.read_csv(self.data_path)

        print("\n" + "=" * 70)
        print(" MODULE 1 VERIFICATION: CLIENT NON-IID DATASET DISTRIBUTION")
        print("=" * 70)
        summary = self.raw_df.groupby(["client_id", "content_type"]).size().unstack(fill_value=0)
        summary["Total"] = summary.sum(axis=1)
        
        # Calculate percentage distributions
        perc_summary = summary.drop(columns=["Total"]).div(summary["Total"], axis=0) * 100
        
        print("\n--- Raw Counts per Client & Content Type ---")
        print(summary.to_string())

        print("\n--- Percentage Distribution per Client (Non-IID Skew Proof) ---")
        print(perc_summary.round(1).to_string())

        avg_eng = self.raw_df.groupby("client_id")["engagement_score"].agg(["mean", "std", "min", "max"])
        print("\n--- Engagement Score Distribution per Client ---")
        print(avg_eng.round(4).to_string())
        print("=" * 70 + "\n")

if __name__ == "__main__":
    prep = DataPreprocessor()
    prep.load_and_preprocess()
    prep.print_data_distribution()
