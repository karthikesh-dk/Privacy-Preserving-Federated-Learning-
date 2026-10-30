"""
Experiment 1: Centralized Baseline Benchmark for Comment Abuse Classification.

Trains CommentClassifierMLP (1200 -> 64 -> 32 -> 4) on the pooled disjoint dataset.
Evaluates:
- Macro-F1, Macro-Precision, Macro-Recall, Accuracy
- Abuse-specific Precision, Recall, F1
- False-Positive Rate (FPR) on benign/normal comments
- ROC-AUC and 4x4 Confusion Matrix
Saves results to experiments/results/centralized_results.json.
"""

import os
import sys
import json

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from modules.data_prep import CommentDataPreprocessor, SimpleDataLoader, LABEL_MAP
from modules.local_model import (
    CommentClassifierMLP,
    train_classifier_epoch,
    evaluate_classifier,
    compute_classification_metrics
)

def run_centralized_experiments(
    comments_path: str = "data/synthetic_comments.csv",
    output_path: str = "experiments/results/centralized_results.json",
    epochs: int = 15,
    batch_size: int = 64,
    learning_rate: float = 0.005
) -> dict:
    print("\n" + "=" * 80)
    print(" EXPERIMENT: CENTRALIZED BASELINE CLASSIFICATION BENCHMARK")
    print("=" * 80)

    # 1. Load data with disjoint user + template split
    prep = CommentDataPreprocessor(comments_path=comments_path)
    prep.load_and_preprocess(test_user_ratio=0.20, test_template_ratio=0.20, seed=42)
    X_train, y_train, X_test, y_test = prep.get_centralized_data()

    print(f" Loaded Centralized Data: Train = {len(X_train)} | Held-Out Test = {len(X_test)}")
    print(f" Vocabulary Features   : {X_train.shape[1]}")

    train_loader = SimpleDataLoader(X_train, y_train, batch_size=batch_size, shuffle=True)
    test_loader = SimpleDataLoader(X_test, y_test, batch_size=batch_size, shuffle=False)

    # 2. Train Centralized MLP
    print(f"\n Training Centralized CommentClassifierMLP ({epochs} epochs)...")
    model = CommentClassifierMLP(input_dim=X_train.shape[1], hidden1=64, hidden2=32, num_classes=4, seed=42)

    for ep in range(1, epochs + 1):
        loss = train_classifier_epoch(model, train_loader, lr=learning_rate)
        if ep % 3 == 0 or ep == epochs:
            print(f"   Epoch {ep:02d}/{epochs:02d} - Training CrossEntropy Loss: {loss:.5f}")

    # 3. Evaluate on held-out test split
    metrics, _, _, _ = evaluate_classifier(model, test_loader)

    print("\n" + "=" * 80)
    print(" CENTRALIZED BENCHMARK RESULTS")
    print("=" * 80)
    print(f" Test Accuracy          : {metrics['accuracy']:.4f}")
    print(f" Macro-F1 Score         : {metrics['macro_f1']:.4f}")
    print(f" Macro-Precision        : {metrics['macro_precision']:.4f}")
    print(f" Macro-Recall           : {metrics['macro_recall']:.4f}")
    print(f" Abuse Class Precision  : {metrics['abuse_precision']:.4f}")
    print(f" Abuse Class Recall     : {metrics['abuse_recall']:.4f}")
    print(f" Abuse Class F1         : {metrics['abuse_f1']:.4f}")
    print(f" FPR on Normal Users    : {metrics['fpr_normal_users']:.4f} ({metrics['fpr_normal_users']*100:.2f}%)")
    print(f" Abuse ROC-AUC          : {metrics['roc_auc']:.4f}")
    
    print("\n--- Confusion Matrix (Rows: True, Cols: Pred) ---")
    print("      [Pos, Neu, Crit, Abu]")
    cm = metrics["confusion_matrix"]
    class_names = ["Positive  ", "Neutral   ", "Criticism ", "Abusive   "]
    for name, row in zip(class_names, cm):
        print(f" {name}: {row}")
    print("=" * 80)

    # 4. Save results to JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results = {
        "model_name": "Centralized CommentClassifierMLP (1200->64->32->4)",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "epochs": epochs,
        **metrics
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n Centralized benchmark results successfully saved to: {output_path}\n")

    return results

if __name__ == "__main__":
    run_centralized_experiments()
