
from dataclasses import dataclass
import math

import numpy as np


@dataclass
class GaussianDPConfig:
    epsilon: float = 4.0
    delta: float = 1e-5
    clip_norm: float = 1.0


def gaussian_noise_multiplier(epsilon: float, delta: float) -> float:
    """
    Conservative textbook Gaussian calibration:
        sigma = sensitivity * sqrt(2 * ln(1.25 / delta)) / epsilon

    This sufficient bound is intended for 0 < epsilon <= 1.
    For larger epsilon, use a tighter Gaussian accountant.
    """
    if not (0 < epsilon <= 1):
        raise ValueError(
            "This calibration requires 0 < epsilon <= 1. "
            "Use a Gaussian DP accountant for epsilon > 1."
        )
    if not (0 < delta < 1):
        raise ValueError("delta must be between 0 and 1.")

    return math.sqrt(2 * math.log(1.25 / delta)) / epsilon


def clip_vector(vector: np.ndarray, clip_norm: float) -> np.ndarray:
    """Clip one example's score vector to a bounded L2 norm."""
    vector = np.asarray(vector, dtype=np.float64)

    if clip_norm <= 0:
        raise ValueError("clip_norm must be positive.")

    norm = np.linalg.norm(vector)
    if norm == 0 or norm <= clip_norm:
        return vector.copy()

    return vector * (clip_norm / norm)


def privatize_mean(
    per_example_scores: np.ndarray,
    config: GaussianDPConfig,
    rng=None,
) -> np.ndarray:
    """
    Add Gaussian noise to a mean of clipped score vectors.

    Assumes fixed-size datasets with add/remove adjacency implemented
    via zero contribution, or replace-one adjacency with the corresponding
    sensitivity adjustment. The caller must choose the correct adjacency
    and sensitivity before claiming a DP guarantee.
    """
    scores = np.asarray(per_example_scores, dtype=np.float64)

    if scores.ndim != 2 or scores.shape[0] == 0:
        raise ValueError("Expected a non-empty [examples, dimensions] matrix.")

    if not (0 < config.epsilon <= 1):
        raise ValueError(
            "Use epsilon <= 1 with this textbook calibration; "
            "larger epsilon needs a Gaussian accountant."
        )

    clipped = np.stack(
        [clip_vector(row, config.clip_norm) for row in scores]
    )

    n = clipped.shape[0]
    mean_scores = clipped.mean(axis=0)

    # Replace-one adjacency: sensitivity of the mean is at most 2*C/n.
    sensitivity = 2.0 * config.clip_norm / n
    multiplier = gaussian_noise_multiplier(config.epsilon, config.delta)
    noise_std = sensitivity * multiplier

    rng = np.random.default_rng() if rng is None else rng
    noise = rng.normal(0.0, noise_std, size=mean_scores.shape)

    return mean_scores + noise


if __name__ == "__main__":
    # Small smoke test only; these are synthetic scores, not model outputs.
    rng = np.random.default_rng(42)
    scores = rng.normal(size=(10, 5))

    config = GaussianDPConfig(
        epsilon=0.5,
        delta=1e-5,
        clip_norm=1.0,
    )

    private_scores = privatize_mean(scores, config, rng=rng)

    print("Input shape:", scores.shape)
    print("Output shape:", private_scores.shape)
    print("Finite output:", bool(np.isfinite(private_scores).all()))
    print("Smoke test passed.")
