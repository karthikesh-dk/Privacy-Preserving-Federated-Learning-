"""
Phase I Master Runner: End-to-End Pipeline.

Executes all 4 components of Phase I in a single command:
1. Synthetic dataset generation and non-IID client partitioning verification.
2. Centralized baseline models training (XGBoost, LightGBM, Centralized PyTorch MLP).
3. Federated Learning simulation with Flower FedAvg across 8 client nodes.
4. Minimal comparison report and convergence plot generation.
"""

import os
import sys

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.generate_dataset import generate_social_media_dataset
from modules.data_prep import DataPreprocessor
from experiments.run_centralized import run_centralized_experiments
from modules.fl_coordinator import run_federated_simulation
from experiments.compare_phase1 import generate_phase1_comparison

def main():
    print("\n" + "#" * 80)
    print(" MCA FINAL YEAR PROJECT - PHASE I EXECUTION PIPELINE")
    print(" Privacy-Preserving Federated Learning for Social Media Engagement Prediction")
    print("#" * 80)

    data_file = "data/social_media_engagement.csv"

    # Step 1: Synthetic Dataset & Non-IID Partitioning
    print("\n>>> STEP 1: Generating Synthetic Dataset & Verifying Non-IID Partitioning...")
    generate_social_media_dataset(num_samples=8000, num_creators=8, output_path=data_file)
    prep = DataPreprocessor(data_path=data_file)
    prep.load_and_preprocess()
    prep.print_data_distribution()

    # Step 2: Centralized Baseline Training
    print("\n>>> STEP 2: Training Centralized Baseline Models (XGBoost, LightGBM, MLP)...")
    run_centralized_experiments(data_path=data_file, mlp_epochs=20)

    # Step 3: Federated Learning with FedAvg (Flower)
    print("\n>>> STEP 3: Running Federated Learning Simulation with FedAvg (Flower)...")
    run_federated_simulation(
        num_rounds=12,
        num_clients=8,
        local_epochs=3,
        batch_size=32,
        learning_rate=0.005,
        data_path=data_file
    )

    # Step 4: Comparison Report & Convergence Curve
    print("\n>>> STEP 4: Generating Phase I Comparison Report & Plots...")
    generate_phase1_comparison()

    print("\n" + "#" * 80)
    print(" PHASE I PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print(" Check 'experiments/results/' for comparison JSONs, CSV logs, and convergence_curve.png")
    print("#" * 80 + "\n")

if __name__ == "__main__":
    main()
