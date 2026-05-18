from abc import ABC, abstractmethod
from typing import Any
import numpy as np

class BaseDynamics(ABC):
    """
    Interface for function-space evolution models governed by kernel-based
    dynamics, providing a unified contract for resetting internal state,
    advancing the time parameter, and generating predictive statistics.

    Methods:
        reset() -> None:
            Restores the internal state of the dynamics to its initial
            configuration.

        step(t: float) -> None:
            Updates the internal time parameter to the specified value,
            advancing the evolution of the underlying function-space model.

        predict(X: np.ndarray) -> Any:
            Computes the model's predictive quantities at the specified 
            input locations.
    """

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def step(self, t: float) -> None:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> Any:
        pass