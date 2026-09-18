"""
Module 3: Federated Learning Coordinator.

Orchestrates in-process federated learning simulations using Flower concepts:
- EngagementClient (Flower NumPyClient) for decentralized on-client training.
- Strategy-driven aggregation (FedAvg, FedProx).
- Global evaluation on held-out test data after each communication round.
- Per-round tracking of loss, MAE, RMSE, R^2, and communication cost.
"""

import os
import sys
import csv
import json
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import flwr as fl
from flwr.common import NDArrays, Scalar

# Ensure project root is accessible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.data_prep import DataPreprocessor, EngagementDataset
from modules.local_model import (
    EngagementMLP,
    get_model_parameters,
    set_model_parameters,
    train_mlp_epoch,
    evaluate_mlp,
    compute_metrics
)
from modules.strategies import FederatedStrategy, FedAvgStrategy, FedProxStrategy

class EngagementClient(fl.client.NumPyClient):
    """
    Flower NumPyClient simulating an independent creator/user client node.
    Trains locally on private data without exposing raw records.
    """
    def __init__(
        self,
        client_id: int,
        train_loader: DataLoader,
        test_loader: DataLoader,
        input_dim: int = 13,
        local_epochs: int = 3,
        lr: float = 0.005,
        proximal_mu: float = 0.0,
        device: Optional[torch.device] = None
    ):
        self.client_id = client_id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.local_epochs = local_epochs
        self.lr = lr
        self.proximal_mu = proximal_mu
        self.device = device or torch.device("cpu")
        self.model = EngagementMLP(input_dim=input_dim).to(self.device)
        self.criterion = nn.MSELoss()

    def get_parameters(self, config: Optional[Dict[str, Scalar]] = None) -> NDArrays:
        return get_model_parameters(self.model)

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        # 1. Update local model with parameters received from the server
        set_model_parameters(self.model, parameters)

        # Retain a copy of the global weights if FedProx proximal term is enabled
        prox_mu = float(config.get("proximal_mu", self.proximal_mu))
        global_params = None
        if prox_mu > 0.0:
            global_params = [p.clone().detach() for p in self.model.parameters()]

        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        # 2. Local client training for local_epochs
        last_epoch_loss = 0.0
        for _ in range(self.local_epochs):
            last_epoch_loss = train_mlp_epoch(
                model=self.model,
                train_loader=self.train_loader,
                optimizer=optimizer,
                criterion=self.criterion,
                device=self.device,
                global_params=global_params,
                proximal_mu=prox_mu
            )

        num_examples = len(self.train_loader.dataset)
        metrics = {
            "client_id": self.client_id,
            "train_loss": float(last_epoch_loss)
        }
        return get_model_parameters(self.model), num_examples, metrics

    def evaluate(self, parameters: NDArrays, config: Optional[Dict[str, Scalar]] = None) -> Tuple[float, int, Dict[str, Scalar]]:
        set_model_parameters(self.model, parameters)
        metrics, _, _ = evaluate_mlp(self.model, self.test_loader, device=self.device)
        num_examples = len(self.test_loader.dataset)
        return float(metrics["mse"]), num_examples, {
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"]
        }

def run_federated_simulation(
    strategy: Optional[FederatedStrategy] = None,
    num_rounds: int = 12,
    num_clients: int = 8,
    local_epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 0.005,
    proximal_mu: float = 0.0,
    data_path: str = "data/social_media_engagement.csv",
    log_csv_path: str = "experiments/results/fedavg_log.csv",
    model_save_path: str = "experiments/results/global_fedavg_model.pt"
) -> Dict[str, Any]:
    """
    Run the in-process federated learning simulation across all clients.
    """
    if strategy is None:
        if proximal_mu > 0.0:
            strategy = FedProxStrategy(proximal_mu=proximal_mu)
        else:
            strategy = FedAvgStrategy()

    print("\n" + "=" * 75)
    print(f" STARTING FEDERATED SIMULATION: {strategy.name}")
    print(f" Clients: {num_clients} | Rounds: {num_rounds} | Local Epochs: {local_epochs} | Batch: {batch_size}")
    print("=" * 75)

    # 1. Prepare data
    prep = DataPreprocessor(data_path=data_path)
    prep.load_and_preprocess()
    client_partitions = prep.get_client_partitions(test_size=0.20, seed=42)

    # Centralized held-out test loader for server-side evaluation
    _, _, X_test_global, y_test_global = prep.get_centralized_data(test_size=0.20, seed=42)
    global_test_ds = EngagementDataset(X_test_global, y_test_global)
    global_test_loader = DataLoader(global_test_ds, batch_size=64, shuffle=False)
    input_dim = X_test_global.shape[1]

    # 2. Instantiate clients
    clients: List[EngagementClient] = []
    for c_id in range(num_clients):
        c_data = client_partitions[c_id]
        train_ds = EngagementDataset(c_data["X_train"], c_data["y_train"])
        test_ds = EngagementDataset(c_data["X_test"], c_data["y_test"])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        client = EngagementClient(
            client_id=c_id,
            train_loader=train_loader,
            test_loader=test_loader,
            input_dim=input_dim,
            local_epochs=local_epochs,
            lr=learning_rate,
            proximal_mu=proximal_mu
        )
        clients.append(client)

    # 3. Initialize server global model
    global_model = EngagementMLP(input_dim=input_dim)
    global_params = get_model_parameters(global_model)

    # Calculate model payload size (communication cost per round)
    payload_bytes = sum(arr.nbytes for arr in global_params)
    payload_kb = payload_bytes / 1024.0

    # Initialize CSV log
    os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)
    with open(log_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "avg_train_loss", "test_loss", "test_mae", "test_rmse", "test_r2", "comm_kb"])

    # 4. Simulation communication rounds loop
    for r in range(1, num_rounds + 1):
        fit_config = strategy.configure_fit(server_round=r)
        fit_results = []
        client_train_losses = []

        # Local training per client
        for client in clients:
            updated_params, num_samples, metrics = client.fit(global_params, fit_config)
            fit_results.append((updated_params, num_samples, metrics))
            client_train_losses.append(metrics["train_loss"])

        avg_train_loss = float(np.mean(client_train_losses))

        # Server-side aggregation
        global_params = strategy.aggregate_fit(server_round=r, results=fit_results)
        set_model_parameters(global_model, global_params)

        # Evaluate aggregated global model on held-out global test set
        eval_metrics, _, _ = evaluate_mlp(global_model, global_test_loader)
        test_loss = float(eval_metrics["mse"])

        # Round communication cost: downlink (server->clients) + uplink (clients->server)
        round_comm_kb = payload_kb * 2 * num_clients

        print(
            f"   Round {r:02d}/{num_rounds:02d} | "
            f"Train Loss: {avg_train_loss:.5f} | "
            f"Test MSE: {test_loss:.5f} | "
            f"MAE: {eval_metrics['mae']:.4f} | "
            f"RMSE: {eval_metrics['rmse']:.4f} | "
            f"R2: {eval_metrics['r2']:.4f}"
        )

        # Append to log CSV
        with open(log_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                r,
                round(avg_train_loss, 5),
                round(test_loss, 5),
                eval_metrics["mae"],
                eval_metrics["rmse"],
                eval_metrics["r2"],
                round(round_comm_kb, 1)
            ])

    # Save final trained global model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(global_model.state_dict(), model_save_path)

    final_metrics, _, _ = evaluate_mlp(global_model, global_test_loader)
    summary = {
        "strategy": strategy.name,
        "num_rounds": num_rounds,
        "num_clients": num_clients,
        "local_epochs": local_epochs,
        "payload_kb_per_client": round(payload_kb, 2),
        "total_communication_mb": round((payload_kb * 2 * num_clients * num_rounds) / 1024.0, 2),
        "final_loss": round(float(final_metrics["mse"]), 5),
        "final_mae": final_metrics["mae"],
        "final_rmse": final_metrics["rmse"],
        "final_r2": final_metrics["r2"],
        "model_path": model_save_path,
        "log_path": log_csv_path
    }

    print("\n" + "=" * 75)
    print(f" FEDERATED SIMULATION COMPLETED SUCCESSFULLY!")
    print(f" Strategy: {strategy.name}")
    print(f" Final Global MAE : {summary['final_mae']}")
    print(f" Final Global RMSE: {summary['final_rmse']}")
    print(f" Final Global R2  : {summary['final_r2']}")
    print(f" Model Saved To   : {model_save_path}")
    print("=" * 75 + "\n")

    return summary

if __name__ == "__main__":
    run_federated_simulation(num_rounds=12, num_clients=8)
