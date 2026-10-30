"""
Module 1: Data Preparation, Leakage-Free Splitting, and Client Partitioning.

Implements:
1. Pure-Python/NumPy SimpleTfidfVectorizer (zero external DLL blocks).
2. Disjoint User + Template Splitting: Prevents template phrasing memorization.
3. Per-Influencer Client Partitioning: Each influencer's community forms one federated client.
4. Batch DataLoaders for decentralized training.
"""

import os
import re
import json
import math
from collections import Counter
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd

LABEL_MAP = {
    "positive": 0,
    "neutral": 1,
    "negative-but-not-abusive": 2,
    "abusive-vulgar": 3
}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

SEVERITY_MAP = {
    "none": 0,
    "mild": 1,
    "moderate": 2,
    "severe": 3
}

class SimpleTfidfVectorizer:
    """
    Lightweight, high-performance TF-IDF Vectorizer in pure NumPy.
    Extracts unigram & bigram vocabulary, applies sublinear TF and smoothed IDF,
    and L2-normalizes representations. Zero C++ DLL dependencies.
    """
    def __init__(self, max_features: int = 1200, min_df: int = 2):
        self.max_features = max_features
        self.min_df = min_df
        self.vocabulary_: Dict[str, int] = {}
        self.idf_: Optional[np.ndarray] = None

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)
        # Generate unigrams + bigrams
        bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]
        return tokens + bigrams

    def fit(self, raw_documents: List[str]):
        doc_count = len(raw_documents)
        df_counter = Counter()

        for doc in raw_documents:
            unique_terms = set(self._tokenize(doc))
            for term in unique_terms:
                df_counter[term] += 1

        # Filter by min_df and sort by document frequency
        filtered_terms = [
            (term, count) for term, count in df_counter.items() if count >= self.min_df
        ]
        filtered_terms.sort(key=lambda x: x[1], reverse=True)
        top_terms = filtered_terms[: self.max_features]

        self.vocabulary_ = {term: idx for idx, (term, _) in enumerate(top_terms)}
        vocab_size = len(self.vocabulary_)

        # Compute smoothed IDF: log((1 + N) / (1 + df)) + 1
        self.idf_ = np.zeros(vocab_size, dtype=np.float32)
        for term, idx in self.vocabulary_.items():
            df = df_counter[term]
            self.idf_[idx] = math.log((1.0 + doc_count) / (1.0 + df)) + 1.0

        return self

    def transform(self, raw_documents: List[str]) -> np.ndarray:
        if not self.vocabulary_ or self.idf_ is None:
            raise ValueError("Vectorizer must be fitted before calling transform.")

        num_docs = len(raw_documents)
        vocab_size = len(self.vocabulary_)
        X = np.zeros((num_docs, vocab_size), dtype=np.float32)

        for i, doc in enumerate(raw_documents):
            terms = self._tokenize(doc)
            term_counts = Counter(terms)
            for term, count in term_counts.items():
                if term in self.vocabulary_:
                    idx = self.vocabulary_[term]
                    # Sublinear TF scaling: 1 + log(tf)
                    tf = 1.0 + math.log(count)
                    X[i, idx] = tf * self.idf_[idx]

            # L2 normalization
            norm = np.linalg.norm(X[i])
            if norm > 0.0:
                X[i] /= norm

        return X

    def fit_transform(self, raw_documents: List[str]) -> np.ndarray:
        return self.fit(raw_documents).transform(raw_documents)

class SimpleDataLoader:
    """Zero-dependency NumPy batch iterator for local training."""
    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32, shuffle: bool = True):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset = self

    def __len__(self):
        return len(self.X)

    def __iter__(self):
        n = len(self.X)
        indices = np.arange(n)
        if self.shuffle:
            np.random.shuffle(indices)
        for start_idx in range(0, n, self.batch_size):
            batch_idx = indices[start_idx : start_idx + self.batch_size]
            yield self.X[batch_idx], self.y[batch_idx]

class CommentDataPreprocessor:
    """
    Manages synthetic comments dataset, performs disjoint user+template train-test split,
    extracts TF-IDF features, and generates private per-influencer federated client partitions.
    """
    def __init__(
        self,
        comments_path: str = "data/synthetic_comments.csv",
        influencers_path: str = "data/synthetic_influencers.json",
        max_features: int = 1200
    ):
        self.comments_path = comments_path
        self.influencers_path = influencers_path
        self.vectorizer = SimpleTfidfVectorizer(max_features=max_features)
        self.df: Optional[pd.DataFrame] = None
        self.influencers: List[dict] = []
        self.train_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.X_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None

    def load_and_preprocess(self, test_user_ratio: float = 0.20, test_template_ratio: float = 0.20, seed: int = 42):
        if not os.path.exists(self.comments_path):
            raise FileNotFoundError(f"Missing {self.comments_path}. Run generate_synthetic_data.py first.")

        self.df = pd.read_csv(self.comments_path)
        if os.path.exists(self.influencers_path):
            with open(self.influencers_path, "r", encoding="utf-8") as f:
                self.influencers = json.load(f)

        # Map labels to numeric targets
        self.df["label_id"] = self.df["label"].map(LABEL_MAP).astype(int)
        self.df["severity_id"] = self.df["abuse_severity"].map(SEVERITY_MAP).astype(int)

        # Realism Guard: Disjoint User + Template Split
        # Ensures test set evaluates generalization on unseen phrasings & unseen users
        np.random.seed(seed)
        unique_users = list(self.df["user_id"].unique())
        unique_templates = list(self.df["template_id"].unique())

        np.random.shuffle(unique_users)
        np.random.shuffle(unique_templates)

        num_test_users = int(len(unique_users) * test_user_ratio)
        num_test_templates = int(len(unique_templates) * test_template_ratio)

        test_users = set(unique_users[:num_test_users])
        train_users = set(unique_users[num_test_users:])

        test_templates = set(unique_templates[:num_test_templates])
        train_templates = set(unique_templates[num_test_templates:])

        # Strict disjoint split:
        # Train comments: created by train_users AND using train_templates
        # Test comments: created by test_users OR using test_templates
        train_mask = self.df["user_id"].isin(train_users) & self.df["template_id"].isin(train_templates)
        test_mask = self.df["user_id"].isin(test_users) | self.df["template_id"].isin(test_templates)

        self.train_df = self.df[train_mask].copy().reset_index(drop=True)
        self.test_df = self.df[test_mask].copy().reset_index(drop=True)

        # Fit TF-IDF only on training comments
        train_texts = self.train_df["text"].astype(str).tolist()
        test_texts = self.test_df["text"].astype(str).tolist()

        self.X_train = self.vectorizer.fit_transform(train_texts)
        self.X_test = self.vectorizer.transform(test_texts)

        self.y_train = self.train_df["label_id"].values.astype(np.int64)
        self.y_test = self.test_df["label_id"].values.astype(np.int64)

        return self

    def get_centralized_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns pooled train and test data for centralized benchmarks."""
        if self.X_train is None:
            self.load_and_preprocess()
        return self.X_train, self.y_train, self.X_test, self.y_test

    def get_client_partitions(self) -> Dict[str, Dict[str, Any]]:
        """
        Partitions the data across influencers (clients).
        Each influencer represents one community with its own private comments.
        """
        if self.X_train is None:
            self.load_and_preprocess()

        partitions = {}
        unique_influencers = sorted(self.df["influencer_id"].unique())

        for inf_id in unique_influencers:
            train_sub_mask = (self.train_df["influencer_id"] == inf_id).values
            test_sub_mask = (self.test_df["influencer_id"] == inf_id).values

            inf_X_train = self.X_train[train_sub_mask]
            inf_y_train = self.y_train[train_sub_mask]

            inf_X_test = self.X_test[test_sub_mask]
            inf_y_test = self.y_test[test_sub_mask]

            # Calculate local abuse rate
            local_abuse_rate = float(np.mean(inf_y_train == LABEL_MAP["abusive-vulgar"])) if len(inf_y_train) > 0 else 0.0

            partitions[inf_id] = {
                "influencer_id": inf_id,
                "X_train": inf_X_train,
                "y_train": inf_y_train,
                "X_test": inf_X_test,
                "y_test": inf_y_test,
                "num_train": len(inf_X_train),
                "num_test": len(inf_X_test),
                "local_abuse_rate": round(local_abuse_rate * 100, 2)
            }

        return partitions

    def get_client_loaders(self, influencer_id: str, batch_size: int = 32) -> Tuple[SimpleDataLoader, SimpleDataLoader]:
        """Return train and test SimpleDataLoaders for a specific influencer client."""
        client_data = self.get_client_partitions()[influencer_id]
        train_loader = SimpleDataLoader(client_data["X_train"], client_data["y_train"], batch_size=batch_size, shuffle=True)
        test_loader = SimpleDataLoader(client_data["X_test"], client_data["y_test"], batch_size=batch_size, shuffle=False)
        return train_loader, test_loader

    def print_partition_summary(self):
        """Display verification summary of disjoint split and per-client non-IID partition."""
        if self.X_train is None:
            self.load_and_preprocess()

        print("\n" + "=" * 80)
        print(" MODULE 1 VERIFICATION: DISJOINT USER+TEMPLATE SPLIT & CLIENT PARTITIONS")
        print("=" * 80)
        print(f" Total Comments Pool   : {len(self.df)}")
        print(f" Train Split (Disjoint): {len(self.train_df)} samples ({len(self.train_df)/len(self.df)*100:.1f}%)")
        print(f" Test Split (Held-Out) : {len(self.test_df)} samples ({len(self.test_df)/len(self.df)*100:.1f}%)")
        print(f" TF-IDF Vocabulary Size: {len(self.vectorizer.vocabulary_)} features")

        partitions = self.get_client_partitions()
        print("\n--- Per-Client (Influencer) Non-IID Partitions ---")
        header = f"{'Client ID':<12} | {'Train Samples':<14} | {'Test Samples':<14} | {'Abuse Rate (%)':<16}"
        print(header)
        print("-" * len(header))
        for inf_id, data in partitions.items():
            print(f"{inf_id:<12} | {data['num_train']:<14} | {data['num_test']:<14} | {data['local_abuse_rate']:<16.2f}")
        print("=" * 80 + "\n")

if __name__ == "__main__":
    prep = CommentDataPreprocessor()
    prep.load_and_preprocess()
    prep.print_partition_summary()
