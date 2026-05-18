import numpy as np
import matplotlib.pyplot as plt
from typing import List
from plotting.base_plotter import BasePlotter
from dynamics.base_dynamics import BaseDynamics


class DynamicsPlotter(BasePlotter):
    """
    Interface for visualizing kernel-based function-space dynamics, providing
    multi-panel plots of predictive means, uncertainty bands, and training
    samples across specified time values.

    Methods:
        plot(X: np.ndarray, y: np.ndarray, X_grid: np.ndarray, times: List[float], dynamics: BaseDynamics) -> None:
            Generates a grid of subplots showing predictive mean curves, marginal
            uncertainty bands, and training points at each time snapshot.
    """

    def plot(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_grid: np.ndarray,
        times: List[float],
        dynamics: BaseDynamics
    ) -> None:

        fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=400)
        axes = axes.flatten()

        Xt = X[:, 0]
        yt = y[:, 0]

        point_colors = ["#FF5733", "#33C1FF", "#9D33FF"]

        for idx, t in enumerate(times):
            ax = axes[idx]

            dynamics.step(float(t))
            mean, std = dynamics.predict(X_grid)

            mean = mean[:, 0]
            std = std[:, 0]
            xg = X_grid[:, 0]

            ax.fill_between(
                xg, mean - std, mean + std,
                color="#83C5BE", alpha=0.35, zorder=1
            )

            ax.plot(
                xg, mean,
                color="#006D77", linewidth=1.2, zorder=2
            )

            for i in range(len(Xt)):
                ax.scatter(
                    Xt[i], yt[i],
                    color=point_colors[i],
                    edgecolor="black",
                    linewidth=0.5,
                    s=120,
                    zorder=10
                )

            ax.set_xlim(0, 1)
            ax.set_ylim(-0.5, 0.5)

            ax.set_xticks([0.0, 0.5, 1.0])
            ax.set_yticks([-0.5, 0.0, 0.5])

            ax.grid(False)
            ax.set_aspect('equal', adjustable='box')

            if idx == len(times) - 1:
                ax.set_title(r"$t = \infty$", fontsize=16)
            else:
                ax.set_title(rf"$t = {t}$", fontsize=16)

            ax.set_xlabel("Input, x", fontsize=14)
            ax.set_ylabel("Output, y", fontsize=14)

        plt.tight_layout()
        self._save()
        plt.close()