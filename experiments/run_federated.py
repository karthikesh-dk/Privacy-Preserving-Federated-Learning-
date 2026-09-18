"""
Experiment 2: Federated Learning with FedAvg (Flower).

Executes 12 communication rounds of FedAvg across 8 decentralized clients,
logging round-by-round convergence metrics to experiments/results/fedavg_log.csv,
and saving summary metrics to experiments/results/fedavg_results.json.
"""

import os
import sys
import json

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.fl_coordinator import run_federated_simulation

def run_federated_fedavg_experiment():
    log_path = "experiments/results/fedavg_log.csv"
    results_path = "experiments/results/fedavg_results.json"

    # Run 12 communication rounds across 8 clients
    summary = run_federated_simulation(
        num_rounds=12,
        num_clients=8,
        local_epochs=3,
        batch_size=32,
        learning_rate=0.005,
        log_csv_path=log_path
    )

    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"FedAvg results saved to: {results_path}")
    print(f"Per-round log saved to: {log_path}")

if __name__ == "__main__":
    run_federated_fedavg_experiment()
