"""
Module 3: Federated Aggregation Strategies.

Implements:
1. FedAvg (McMahan et al., 2017): Communication-efficient learning via weighted averaging.
2. FedProx (Li et al., 2020): Federated optimization with proximal regularization for heterogeneous non-IID clients.
3. FedAvg + SecAgg: Pairwise zero-sum additive secret masking (Bonawitz et al.).
4. FedAvg + DP: Client-level differential privacy with L2 clipping and calibrated Gaussian noise injection.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from flwr.server.strategy.aggregate import aggregate as flwr_aggregate

from modules.secure_aggregation import SecureAggregator
from modules.dp_privacy import DPModelPerturber, DifferentialPrivacyAccountant

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

class FedAvgSecAggStrategy(FederatedStrategy):
    """
    FedAvg with Secure Aggregation (Bonawitz et al.).
    Clients add pairwise zero-sum random masks so the central server only sees
    the true aggregate while individual client weights remain encrypted.
    """
    def __init__(self, client_ids: List[str], seed: int = 42):
        super().__init__(name="FedAvg + SecAgg")
        self.sec_agg = SecureAggregator(client_ids=client_ids, seed=seed)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[List[np.ndarray], int, Dict[str, Any]]]
    ) -> List[np.ndarray]:
        # Apply pairwise masks to each client's parameters
        masked_results = []
        for params, num_samples, metrics in results:
            cid = metrics.get("client_id", "unknown")
            masked_params = self.sec_agg.mask_client_weights(cid, params)
            masked_results.append((cid, masked_params, num_samples))

        # Secure aggregate (masks cancel out to exactly 0.0)
        return self.sec_agg.secure_aggregate(masked_results)

class FedAvgDPStrategy(FederatedStrategy):
    """
    FedAvg with Differential Privacy (DP).
    Applies L2-norm parameter clipping and Gaussian noise injection to provide
    formal (epsilon, delta)-differential privacy guarantees.
    """
    def __init__(
        self,
        clip_norm: float = 1.5,
        noise_multiplier: float = 0.6,
        target_delta: float = 1e-5,
        seed: int = 42
    ):
        super().__init__(name=f"FedAvg + DP (sigma={noise_multiplier})")
        self.clip_norm = clip_norm
        self.noise_multiplier = noise_multiplier
        self.perturber = DPModelPerturber(clip_norm=clip_norm, noise_multiplier=noise_multiplier, seed=seed)
        self.accountant = DifferentialPrivacyAccountant(target_delta=target_delta, clip_norm=clip_norm, noise_multiplier=noise_multiplier)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[List[np.ndarray], int, Dict[str, Any]]]
    ) -> List[np.ndarray]:
        # Track privacy budget
        self.accountant.step()

        # Apply DP clipping and calibrated noise to each client update
        privatized_results = []
        for params, num_samples, metrics in results:
            noisy_params, _ = self.perturber.privatize_update(params, num_samples=num_samples)
            privatized_results.append((noisy_params, num_samples))

        return flwr_aggregate(privatized_results)
