"""
Phase I Master Runner: End-to-End Pipeline.

Executes all core components of Phase I in a single command:
1. Synthetic dataset generation (8 influencers, 2,000 users, 6 post types, 4-class comments).
2. Non-IID client partitioning & vocabulary verification with disjoint user/template splits.
3. Centralized baseline model training (CommentClassifierMLP).
4. Federated Learning simulation with Flower FedAvg across 8 client nodes.
5. User Behaviour Profiler: repeat abuser detection, burst analysis, escalation trend, and moderation actions.
6. Comparative evaluation report & convergence visualization (convergence_curve.png).
"""

import os
import sys

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.generate_synthetic_data import generate_synthetic_data
from modules.data_prep import CommentDataPreprocessor
from experiments.run_centralized import run_centralized_experiments
from experiments.run_federated import run_federated_fedavg_experiment
from modules.behaviour_analyzer import run_behaviour_analysis
from experiments.compare_phase1 import generate_phase1_comparison

def main():
    print("\n" + "#" * 88)
    print(" MCA FINAL YEAR PROJECT - PHASE I EXECUTION PIPELINE")
    print(" Title: Privacy-Preserving Federated Learning for Social Media Influencer")
    print("#" * 88)

    comments_csv = "data/synthetic_comments.csv"

    # Step 1: Synthetic Dataset & Client Partitioning
    print("\n>>> STEP 1: Generating Synthetic Dataset & Verifying Client Partitions...")
    if not os.path.exists(comments_csv):
        generate_synthetic_data(num_influencers=8, num_users=2000, posts_per_influencer=15)
    else:
        print(f" [OK] Existing synthetic dataset detected at {comments_csv}")

    prep = CommentDataPreprocessor(comments_path=comments_csv)
    prep.load_and_preprocess(test_user_ratio=0.20, test_template_ratio=0.20)
    prep.print_partition_summary()

    # Step 2: Centralized Baseline Training
    print("\n>>> STEP 2: Training Centralized Baseline Classifier (CommentClassifierMLP)...")
    run_centralized_experiments(comments_path=comments_csv, epochs=15)

    # Step 3: Federated Learning with FedAvg (Flower)
    print("\n>>> STEP 3: Running Federated Learning Simulation with FedAvg (Flower)...")
    run_federated_fedavg_experiment()

    # Step 4: User Behaviour Analysis & Repeat Abuser Identification
    print("\n>>> STEP 4: Executing User Behaviour Profiler & Abuser Benchmark...")
    run_behaviour_analysis()

    # Step 5: Comparison Report & Convergence Curve Plot
    print("\n>>> STEP 5: Generating Phase I Comparison Report & High-Res Plots...")
    generate_phase1_comparison()

    print("\n" + "#" * 88)
    print(" PHASE I PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print(" Deliverables available in 'experiments/results/':")
    print("  - centralized_results.json    : Baseline classification benchmark")
    print("  - fedavg_results.json         : Federated FedAvg benchmark")
    print("  - fedavg_log.csv              : Per-round metrics & loss tracking")
    print("  - user_behaviour_profiles.csv : 1,899 user profiles with risk scores & actions")
    print("  - behaviour_benchmark.json    : Abuser identification precision/recall metrics")
    print("  - convergence_curve.png       : Loss & F1 convergence plot across FL rounds")
    print("  - global_fedavg_model.npz     : Global trained model parameters")
    print("#" * 88 + "\n")

if __name__ == "__main__":
    main()
