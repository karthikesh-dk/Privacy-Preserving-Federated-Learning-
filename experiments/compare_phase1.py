"""
Phase I Comparison Report: Centralized vs Federated Learning (FedAvg).

Loads benchmark logs and creates:
1. Side-by-side comparison table (XGBoost vs LightGBM vs Centralized MLP vs FedAvg).
2. Publication-quality convergence plots (Loss & Error vs Communication Rounds).
3. Saves chart to experiments/results/convergence_curve.png.
"""

import os
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

def generate_phase1_comparison(
    centralized_path: str = "experiments/results/centralized_results.json",
    fedavg_path: str = "experiments/results/fedavg_results.json",
    fedavg_log_path: str = "experiments/results/fedavg_log.csv",
    chart_output_path: str = "experiments/results/convergence_curve.png"
):
    print("\n" + "=" * 80)
    print(" PHASE I COMPARATIVE EVALUATION REPORT: CENTRALIZED VS FEDERATED LEARNING")
    print("=" * 80)

    # 1. Load JSON results
    if not os.path.exists(centralized_path):
        raise FileNotFoundError(f"Missing {centralized_path}. Run run_centralized.py first.")
    if not os.path.exists(fedavg_path):
        raise FileNotFoundError(f"Missing {fedavg_path}. Run run_federated.py first.")

    with open(centralized_path, "r") as f:
        cent_res = json.load(f)
    with open(fedavg_path, "r") as f:
        fed_res = json.load(f)

    # 2. Print Comparative Table
    print("\n" + "-" * 88)
    header = f"{'Model / Architecture':<36} | {'Data Paradigm':<16} | {'MAE':<8} | {'RMSE':<8} | {'R2 Score':<8}"
    print(header)
    print("-" * 88)

    models_data = [
        ("Centralized XGBoost", "Pooled (Central)", cent_res["xgboost"]["mae"], cent_res["xgboost"]["rmse"], cent_res["xgboost"]["r2"]),
        ("Centralized LightGBM", "Pooled (Central)", cent_res["lightgbm"]["mae"], cent_res["lightgbm"]["rmse"], cent_res["lightgbm"]["r2"]),
        ("Centralized PyTorch MLP", "Pooled (Central)", cent_res["mlp_centralized"]["mae"], cent_res["mlp_centralized"]["rmse"], cent_res["mlp_centralized"]["r2"]),
        ("Federated FedAvg (12 Rnds, 8 Clients)", "On-Device (FL)", fed_res["final_mae"], fed_res["final_rmse"], fed_res["final_r2"]),
    ]

    for name, paradigm, mae, rmse, r2 in models_data:
        print(f"{name:<36} | {paradigm:<16} | {mae:<8.4f} | {rmse:<8.4f} | {r2:<8.4f}")
    print("-" * 88)

    print("\n[Analysis / Key Takeaways]:")
    print("1. Data Locality: Under Federated FedAvg, zero raw social media posts or user engagement")
    print("   records ever left the 8 client nodes. Only model weight updates were transmitted.")
    print(f"2. Utility Retention: Federated FedAvg achieved an MAE of {fed_res['final_mae']:.4f} and RMSE of {fed_res['final_rmse']:.4f},")
    print(f"   successfully learning collaborative cross-creator engagement dynamics without centralization.")
    print(f"3. Communication Efficiency: Total payload transferred was {fed_res.get('total_communication_mb', 'N/A')} MB across 12 rounds.\n")

    # 3. Generate Matplotlib Convergence Plot
    if os.path.exists(fedavg_log_path):
        log_df = pd.read_csv(fedavg_log_path)
        
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

        # Subplot 1: MSE Loss Convergence
        rounds = log_df["round"]
        ax1.plot(rounds, log_df["test_loss"], marker="o", color="#2563eb", linewidth=2, label="Global Test MSE Loss")
        if "avg_train_loss" in log_df.columns:
            ax1.plot(rounds, log_df["avg_train_loss"], marker="s", linestyle="--", color="#10b981", linewidth=1.8, label="Client Avg Train Loss")
        ax1.set_title("FedAvg Convergence Curve (Loss vs Communication Round)", fontsize=12, fontweight="bold", pad=12)
        ax1.set_xlabel("Communication Round", fontsize=11)
        ax1.set_ylabel("Mean Squared Error (MSE)", fontsize=11)
        ax1.set_xticks(rounds)
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend(frameon=True, facecolor="white", edgecolor="none", shadow=True)

        # Subplot 2: MAE & RMSE Error Metrics
        ax2.plot(rounds, log_df["test_mae"], marker="^", color="#f59e0b", linewidth=2, label="Test MAE")
        ax2.plot(rounds, log_df["test_rmse"], marker="d", color="#ef4444", linewidth=2, label="Test RMSE")
        ax2.axhline(y=cent_res["mlp_centralized"]["mae"], color="#6b7280", linestyle=":", label=f"Centralized MLP MAE ({cent_res['mlp_centralized']['mae']:.4f})")
        ax2.set_title("Engagement Error Progression Across FL Rounds", fontsize=12, fontweight="bold", pad=12)
        ax2.set_xlabel("Communication Round", fontsize=11)
        ax2.set_ylabel("Error Metric Value", fontsize=11)
        ax2.set_xticks(rounds)
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend(frameon=True, facecolor="white", edgecolor="none", shadow=True)

        plt.tight_layout()
        os.makedirs(os.path.dirname(chart_output_path), exist_ok=True)
        plt.savefig(chart_output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Convergence plot successfully saved to: {chart_output_path}")

    print("=" * 80 + "\n")

if __name__ == "__main__":
    generate_phase1_comparison()
