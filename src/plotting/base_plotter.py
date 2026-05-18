from abc import ABC, abstractmethod
from typing import Optional
import matplotlib.pyplot as plt
import os

class BasePlotter(ABC):
    """
    Interface for plotting utilities supporting visualization of model outputs
    and experimental results, providing a unified contract for figure generation
    and optional filesystem persistence.

    Attributes:
        filename (Optional[str]): Name of the file where the generated figure
            will be saved, or None to disable saving.

    Methods:
        plot(*args, **kwargs) -> None:
            Generates a visualization using the provided inputs, delegating the
            specific plotting logic to subclasses.
    """

    def __init__(self, filename: Optional[str] = None) -> None:
        self.filename = filename

    def _save(self) -> None:
        if self.filename is None:
            return

        root_dir = os.path.join(os.path.dirname(__file__), "..", "..")
        figures_dir = os.path.join(root_dir, "figures")
        os.makedirs(figures_dir, exist_ok=True)

        path = os.path.join(figures_dir, self.filename)
        plt.savefig(path)

    @abstractmethod
    def plot(self, *args, **kwargs) -> None:
        pass