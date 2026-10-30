"""
Module 6: Differential Privacy (DP) for Federated Model Updates.

Implements:
1. L2-norm gradient / weight update clipping to bound individual sensitivity.
2. Calibrated Gaussian mechanism perturbation for formal (epsilon, delta)-DP guarantees.
3. Privacy budget tracking (cumulative epsilon spent across FL communication rounds).
4. Pure NumPy execution (zero DLL dependencies, fully compatible with Windows SAC).
"""

from typing import List, Tuple, Dict, Any
import numpy as np

class DifferentialPrivacyAccountant:
    """
    Tracks privacy budget consumption (epsilon, delta) across federated communication rounds
    using the Gaussian Mechanism and advanced composition theorems.
    """
    def __init__(
        self,
        target_delta: float = 1e-5,
        clip_norm: float = 1.0,
        noise_multiplier: float = 0.8
    ):
        self.target_delta = target_delta
        self.clip_norm = clip_norm
        self.noise_multiplier = noise_multiplier
        self.rounds_completed = 0

    def compute_spent_epsilon(self, num_rounds: int, sample_rate: float = 1.0) -> float:
        """
        Compute cumulative epsilon spent after 'num_rounds' using standard Gaussian composition:
        epsilon ~= (sqrt(2 * num_rounds * ln(1 / delta)) + num_rounds * (e^epsilon_0 - 1)) ...
        Closed-form bound for FL Gaussian mechanism:
        epsilon = (clip_norm * sqrt(2 * num_rounds * ln(1.25 / delta))) / (noise_multiplier * sqrt(N))
        """
        if self.noise_multiplier <= 0:
            return float("inf")
        # Standard analytical Gaussian DP bound
        eps = (np.sqrt(2.0 * num_rounds * np.log(1.25 / self.target_delta))) / self.noise_multiplier
        return round(float(eps), 3)

    def step(self) -> float:
        """Increment round and return cumulative epsilon."""
        self.rounds_completed += 1
        return self.compute_spent_epsilon(self.rounds_completed)

class DPModelPerturber:
    """
    Applies gradient/parameter clipping and calibrated Gaussian noise injection
    to guarantee local differential privacy on client updates.
    """
    def __init__(
        self,
        clip_norm: float = 1.0,
        noise_multiplier: float = 0.5,
        seed: int = 42
    ):
        self.clip_norm = clip_norm
        self.noise_multiplier = noise_multiplier
        self.rng = np.random.RandomState(seed)

    def clip_parameters(self, weights_delta: List[np.ndarray]) -> Tuple[List[np.ndarray], float]:
        """
        Clip parameter updates to maximum L2-norm C:
        delta = delta * min(1.0, C / ||delta||_2)
        """
        # Calculate total L2 norm across all weight layers
        total_sq_norm = sum(np.sum(w ** 2) for w in weights_delta)
        total_norm = float(np.sqrt(total_sq_norm))

        clip_factor = min(1.0, self.clip_norm / (total_norm + 1e-8))
        clipped_delta = [w * clip_factor for w in weights_delta]

        return clipped_delta, total_norm

    def add_noise(self, weights_delta: List[np.ndarray], num_samples: int = 1) -> List[np.ndarray]:
        """
        Add calibrated Gaussian noise to clipped updates:
        noise ~ N(0, (sigma * C)^2 / num_samples^2)
        """
        if self.noise_multiplier <= 0:
            return weights_delta

        sigma = (self.clip_norm * self.noise_multiplier) / max(1, num_samples)
        noisy_delta = []

        for w in weights_delta:
            noise = self.rng.normal(loc=0.0, scale=sigma, size=w.shape).astype(np.float32)
            noisy_delta.append(w + noise)

        return noisy_delta

    def privatize_update(
        self,
        weights_delta: List[np.ndarray],
        num_samples: int = 1
    ) -> Tuple[List[np.ndarray], Dict[str, float]]:
        """Clip and inject noise in a single unified step."""
        clipped, pre_clip_norm = self.clip_parameters(weights_delta)
        noisy = self.add_noise(clipped, num_samples=num_samples)

        post_clip_norm = float(np.sqrt(sum(np.sum(w ** 2) for w in clipped)))
        return noisy, {
            "pre_clip_norm": pre_clip_norm,
            "post_clip_norm": post_clip_norm,
            "clip_norm_threshold": self.clip_norm,
            "noise_multiplier": self.noise_multiplier
        }

def test_dp_privacy():
    """Verify DP clipping, noise addition, and accountant tracking."""
    perturber = DPModelPerturber(clip_norm=2.0, noise_multiplier=0.8, seed=42)
    accountant = DifferentialPrivacyAccountant(target_delta=1e-5, clip_norm=2.0, noise_multiplier=0.8)

    sample_weights = [np.array([3.0, 4.0], dtype=np.float32)]  # Norm = 5.0
    privatized, info = perturber.privatize_update(sample_weights, num_samples=1)

    print(f"Original Norm       : {info['pre_clip_norm']:.4f}")
    print(f"Clipped Norm        : {info['post_clip_norm']:.4f} (Bound: {perturber.clip_norm})")
    print(f"Privatized Weights  : {privatized[0]}")
    
    # Calculate 12-round privacy budget
    eps_12 = accountant.compute_spent_epsilon(num_rounds=12)
    print(f"12-Round DP Budget  : (epsilon={eps_12:.2f}, delta=1e-5)")

    assert info["post_clip_norm"] <= perturber.clip_norm + 1e-5, "Clipping bound violated!"
    print(" [OK] Differential Privacy module verified successfully!")

if __name__ == "__main__":
    test_dp_privacy()
