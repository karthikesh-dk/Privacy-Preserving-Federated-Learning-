"""
Module 3: Federated Aggregation Strategies.

Implements:
1. FedAvg (McMahan et al., 2017): Communication-efficient learning via weighted averaging.
2. FedProx (Li et al., 2020): Federated optimization with proximal regularization for heterogeneous non-IID clients.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from flwr.server.strategy.aggregate import aggregate as flwr_aggregate

class FederatedStrategy:
    """Base class for Federated Aggregation Strategies."""
    def __init__(self, name: str = "FedAvg"):
        self.name = name

    def configure_fit(self, server_round: int) -> Dict[str, Any]:
        """Generate client configuration dictionary for the round."""
        return {"server_round": server_round}

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[List[np.ndarray], int, Dict[str, Any]]]
    ) -> List[np.ndarray]:
        """
        Aggregate client model parameters.
        Each item in results is: (parameters_list, num_examples, metrics)
        """
        weights_and_samples = [(params, num_samples) for params, num_samples, _ in results]
        return flwr_aggregate(weights_and_samples)

class FedAvgStrategy(FederatedStrategy):
    """
    Standard Federated Averaging (McMahan et al., 2017).
    Weighted average of model parameters by client dataset size.
    """
    def __init__(self):
        super().__init__(name="FedAvg")

class FedProxStrategy(FederatedStrategy):
    """
    FedProx (Li et al., 2020).
    Adds proximal regularization term (mu / 2) * ||w - w_t||^2 to client local objectives
    to stabilize convergence under severe non-IID data skew.
    """
    def __init__(self, proximal_mu: float = 0.01):
        super().__init__(name=f"FedProx (mu={proximal_mu})")
        self.proximal_mu = proximal_mu

    def configure_fit(self, server_round: int) -> Dict[str, Any]:
        return {
            "server_round": server_round,
            "proximal_mu": self.proximal_mu
        }
