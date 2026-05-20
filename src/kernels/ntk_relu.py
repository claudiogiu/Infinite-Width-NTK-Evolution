import numpy as np
from typing import Tuple
from kernels.base_kernel import BaseKernel

class NTKReLUKernel(BaseKernel):
    """
    Interface for computing the analytic Neural Tangent Kernel (NTK) of an
    infinite-width fully connected ReLU network, providing covariance evaluations
    required for kernel gradient dynamics and function-space evolution models.

    Methods:
        __call__(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
            Computes the full NTK matrix between two input sets using the
            closed-form arc-cosine expressions for ReLU activations.

        diag(X: np.ndarray) -> np.ndarray:
            Returns the diagonal of the NTK, corresponding to the self-covariance
            of each input under the infinite-width ReLU kernel.

        pairwise(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            Computes pairwise dot products and norms for a single input set,
            supporting efficient kernel construction and reuse.
    """

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)

        norm_X = np.linalg.norm(X, axis=1, keepdims=True)
        norm_Y = np.linalg.norm(Y, axis=1, keepdims=True)
        dot = X @ Y.T
        cos = dot / (norm_X @ norm_Y.T + 1e-10)

        theta = np.arccos(np.clip(cos, -1.0, 1.0))
        pi = np.pi

        sigma = ((pi - theta) * cos + np.sin(theta)) / (2.0 * pi)
        sigma_prime = (pi - theta) / (2.0 * pi)

        nngp_part = (norm_X @ norm_Y.T) * sigma + 1.0
        ntk = nngp_part + dot * sigma_prime

        return ntk

    def diag(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        norm2 = np.linalg.norm(X, axis=1) ** 2
        return norm2 + 1.0

    def pairwise(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=float)
        norm = np.linalg.norm(X, axis=1, keepdims=True)
        dot = X @ X.T
        return dot, norm
