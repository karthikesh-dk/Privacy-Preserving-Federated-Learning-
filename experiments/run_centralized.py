"""
Experiment 1: Centralized Baseline Benchmark.

Trains and evaluates:
1. XGBoost Regressor
2. LightGBM Regressor
3. Centralized PyTorch EngagementMLP (d -> 64 -> 32 -> 1)

Evaluates MAE, RMSE, and R^2 on the held-out centralized 20% test split,
saving results to experiments/results/centralized_results.json.
"""

import os
import sys
import json

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from modules.data_prep import DataPreprocessor, EngagementDataset
from modules.local_model import (
    EngagementMLP,
    train_mlp_epoch,
    evaluate_mlp,
    train_xgboost,
    train_lightgbm,
    compute_metrics
)

def run_centralized_experiments(
    data_path: str = "data/social_media_engagement.csv",
    output_path: str = "experiments/results/centralized_results.json",
    mlp_epochs: int = 25,
    batch_size: int = 64,
    learning_rate: float = 0.005
) -> dict:
    print("\n" + "=" * 75)
    print(" EXPERIMENT: CENTRALIZED BASELINE BENCHMARKS")
    print("=" * 75)

    # 1. Load data
    prep = DataPreprocessor(data_path=data_path)
    X_train, y_train, X_test, y_test = prep.get_centralized_data(test_size=0.20, seed=42)
    print(f"Loaded Centralized Data: Train samples = {len(X_train)}, Test samples = {len(X_test)}")
    print(f"Feature dimension = {X_train.shape[1]}")

    results = {}

    # 2. XGBoost Baseline
    print("\n[1/3] Training Centralized XGBoost Regressor...")
    xgb_model = train_xgboost(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    metrics_xgb = compute_metrics(y_test, y_pred_xgb)
    results["xgboost"] = {
        "model_name": "Centralized XGBoost Regressor",
        "num_train_samples": len(X_train),
        "num_test_samples": len(X_test),
        **metrics_xgb
    }
    print(f"      XGBoost -> MAE: {metrics_xgb['mae']:.4f} | RMSE: {metrics_xgb['rmse']:.4f} | R2: {metrics_xgb['r2']:.4f}")

    # 3. LightGBM Baseline
    print("\n[2/3] Training Centralized LightGBM Regressor...")
    lgb_model = train_lightgbm(X_train, y_train)
    y_pred_lgb = lgb_model.predict(X_test)
    metrics_lgb = compute_metrics(y_test, y_pred_lgb)
    results["lightgbm"] = {
        "model_name": "Centralized LightGBM Regressor",
        "num_train_samples": len(X_train),
        "num_test_samples": len(X_test),
        **metrics_lgb
    }
    print(f"      LightGBM -> MAE: {metrics_lgb['mae']:.4f} | RMSE: {metrics_lgb['rmse']:.4f} | R2: {metrics_lgb['r2']:.4f}")

    # 4. PyTorch MLP Baseline
    print(f"\n[3/3] Training Centralized PyTorch EngagementMLP ({mlp_epochs} epochs)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = EngagementDataset(X_train, y_train)
    test_ds = EngagementDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    mlp_model = EngagementMLP(input_dim=X_train.shape[1]).to(device)
    optimizer = optim.AdamW(mlp_model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.MSELoss()

    for epoch in range(1, mlp_epochs + 1):
        loss = train_mlp_epoch(mlp_model, train_loader, optimizer, criterion, device)
        if epoch % 5 == 0 or epoch == mlp_epochs:
            print(f"      Epoch {epoch:02d}/{mlp_epochs:02d} - Training MSE Loss: {loss:.6f}")

    metrics_mlp, _, _ = evaluate_mlp(mlp_model, test_loader, device=device)
    results["mlp_centralized"] = {
        "model_name": "Centralized PyTorch MLP (d->64->32->1)",
        "num_train_samples": len(X_train),
        "num_test_samples": len(X_test),
        "epochs": mlp_epochs,
        **metrics_mlp
    }
    print(f"      PyTorch MLP -> MAE: {metrics_mlp['mae']:.4f} | RMSE: {metrics_mlp['rmse']:.4f} | R2: {metrics_mlp['r2']:.4f}")

    # 5. Summary Table
    print("\n" + "=" * 75)
    print(" CENTRALIZED BENCHMARK RESULTS SUMMARY")
    print("=" * 75)
    header = f"{'Model':<35} | {'MAE':<8} | {'RMSE':<8} | {'R2 Score':<8}"
    print(header)
    print("-" * len(header))
    for key, data in results.items():
        print(f"{data['model_name']:<35} | {data['mae']:<8.4f} | {data['rmse']:<8.4f} | {data['r2']:<8.4f}")
    print("=" * 75)

    # 6. Save JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults successfully saved to: {output_path}\n")

    return results

if __name__ == "__main__":
    run_centralized_experiments()
