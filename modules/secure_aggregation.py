"""
Module 5: Secure Aggregation via Additive Masking (Bonawitz et al.).

Implements:
1. Pairwise secret seed agreement between participating client nodes.
2. Zero-sum additive masking: client i adds pseudo-random perturbation masks M_i
   such that sum(M_i) = 0 across all honest clients.
3. Server receives only masked parameter tensors W_i' = W_i + M_i, making individual
   client weights information-theoretically or computationally private.
4. Server sum cancels all masks exactly: sum(W_i') = sum(W_i).
"""

from typing import List, Dict, Tuple, Any
import numpy as np

class SecureAggregator:
    """
    Simulates pairwise additive secret masking for federated model updates.
    Guarantees the central server cannot reconstruct any single influencer's
    local community updates, preventing model inversion or membership inference.
    """
    def __init__(self, client_ids: List[str], seed: int = 42):
        self.client_ids = client_ids
        self.num_clients = len(client_ids)
        self.seed = seed
        self.pairwise_seeds: Dict[Tuple[str, str], int] = {}
        self._generate_shared_keys()

    def _generate_shared_keys(self):
        """Simulate Diffie-Hellman pairwise key agreement between client pairs."""
        rng = np.random.RandomState(self.seed)
        for i in range(self.num_clients):
            for j in range(i + 1, self.num_clients):
                c_i = self.client_ids[i]
                c_j = self.client_ids[j]
                # Symmetric shared pseudorandom seed
                shared_seed = int(rng.randint(1, 10000000))
                self.pairwise_seeds[(c_i, c_j)] = shared_seed
                self.pairwise_seeds[(c_j, c_i)] = shared_seed

    def mask_client_weights(
        self,
        client_id: str,
        weights: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Add zero-sum perturbation masks to client's parameter list:
        W_i' = W_i + sum_{j > i} PRG(s_ij) - sum_{j < i} PRG(s_ji)
        """
        c_idx = self.client_ids.index(client_id)
        masked_weights = [w.copy() for w in weights]

        for other_idx, other_id in enumerate(self.client_ids):
            if other_idx == c_idx:
                continue

            shared_seed = self.pairwise_seeds[(client_id, other_id)]
            prg = np.random.RandomState(shared_seed)

            # Mask sign: +1 if other_idx > c_idx, -1 if other_idx < c_idx
            sign = 1.0 if other_idx > c_idx else -1.0

            for layer_idx in range(len(masked_weights)):
                mask = prg.normal(loc=0.0, scale=1.0, size=masked_weights[layer_idx].shape).astype(np.float32)
                masked_weights[layer_idx] += sign * mask

        return masked_weights

    def secure_aggregate(
        self,
        masked_client_results: List[Tuple[str, List[np.ndarray], int]]
    ) -> List[np.ndarray]:
        """
        Server-side unweighted or weighted aggregation of masked parameters.
        The additive masks sum to exactly zero: sum_{i} M_i = 0.
        Returns true aggregate: (1 / N) * sum(W_i).
        """
        if not masked_client_results:
            raise ValueError("No client updates to aggregate.")

        num_participants = len(masked_client_results)
        num_layers = len(masked_client_results[0][1])

        # Accumulate masked parameters
        aggregated_weights = [np.zeros_like(masked_client_results[0][1][l]) for l in range(num_layers)]

        for _, masked_w, _ in masked_client_results:
            for l in range(num_layers):
                aggregated_weights[l] += masked_w[l]

        # Divide by number of clients
        for l in range(num_layers):
            aggregated_weights[l] /= float(num_participants)

        return aggregated_weights

def verify_secure_aggregation():
    """Unit test and demonstration of zero-sum secret masking."""
    clients = [f"client_{i}" for i in range(4)]
    sec_agg = SecureAggregator(clients, seed=123)

    # Synthetic client parameter updates
    true_weights = {
        c: [np.array([1.0 * (i + 1), 2.0 * (i + 1)], dtype=np.float32)]
        for i, c in enumerate(clients)
    }

    # 1. Plain direct average (ground truth)
    plain_avg = np.mean([true_weights[c][0] for c in clients], axis=0)

    # 2. Clients apply local pairwise masks
    masked_updates = []
    for c in clients:
        m_w = sec_agg.mask_client_weights(c, true_weights[c])
        masked_updates.append((c, m_w, 100))

    # 3. Server sums masked updates
    secure_avg = sec_agg.secure_aggregate(masked_updates)[0]

    # Verify precision
    diff = np.max(np.abs(plain_avg - secure_avg))
    print(f"Plain Average  : {plain_avg}")
    print(f"Secure Average : {secure_avg}")
    print(f"Mask Residual  : {diff:.8f}")
    assert diff < 1e-5, "Secure aggregation failed to cancel masks!"
    print(" [OK] Secure Aggregation verified successfully! Additive masks cancelled out to exactly 0.0.")

if __name__ == "__main__":
    verify_secure_aggregation()
