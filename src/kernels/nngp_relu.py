import numpy as np
from typing import Tuple
from kernels.base_kernel import BaseKernel


class NNGPReLUKernel(BaseKernel):
    """
    Interface for computing the analytic Neural Network Gaussian Process (NNGP)
    kernel of an infinite-width fully connected ReLU network, providing closed-form
    covariance evaluations for GP priors in function-space models.

    Attributes:
        sigma_sq_p (float):
            Scaling factor applied to the pre-activation variance, controlling the
            amplitude of the resulting NNGP covariance.

    Methods:
        __call__(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
            Computes the full NNGP kernel matrix between two input sets using the
            arc-cosine covariance function induced by ReLU activations.

        diag(X: np.ndarray) -> np.ndarray:
            Returns the diagonal of the NNGP kernel, corresponding to the
            self-covariance of each input under the infinite-width ReLU GP prior.

        pairwise(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            Computes pairwise dot products and norms for a single input set,
            supporting efficient kernel construction and reuse.
    """

    def __init__(self, sigma_sq_p: float = 2.0) -> None:
        self.sigma_sq_p = sigma_sq_p

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)

        norm_X = np.linalg.norm(X, axis=1, keepdims=True)
        norm_Y = np.linalg.norm(Y, axis=1, keepdims=True)
        dot = X @ Y.T
        cos = dot / (norm_X @ norm_Y.T + 1e-10)

        theta = np.arccos(np.clip(cos, -1.0, 1.0))
        pi = np.pi

        base = ((pi - theta) * cos + np.sin(theta)) / (2.0 * pi)
        return self.sigma_sq_p * (norm_X @ norm_Y.T) * base + 1.0

    def diag(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        norm2 = np.linalg.norm(X, axis=1) ** 2
        return self.sigma_sq_p * norm2 + 1.0

    def pairwise(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        norm = np.linalg.norm(X, axis=1, keepdims=True)
        dot = X @ X.T
        return dot, norm