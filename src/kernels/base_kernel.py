from abc import ABC, abstractmethod
from typing import Any
import numpy as np

class BaseKernel(ABC):
    """
    Interface for analytic kernel operators used in infinite-width neural-network
    models, providing a unified contract for full kernel evaluation, diagonal
    extraction, and pairwise geometric computations.

    Methods:
        __call__(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
            Computes the full kernel matrix between two input sets.

        diag(X: np.ndarray) -> np.ndarray:
            Returns the diagonal of the kernel matrix for the given inputs.

        pairwise(X: np.ndarray) -> Any:
            Computes pairwise geometric quantities required for kernel construction.
    """

    @abstractmethod
    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def diag(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def pairwise(self, X: np.ndarray) -> Any:
        pass