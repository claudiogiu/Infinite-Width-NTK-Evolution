import numpy as np
from typing import Callable, Tuple
from dynamics.base_dynamics import BaseDynamics


class NTKGaussianProcess(BaseDynamics):
    """
    Interface for simulating the infinite-width Neural Tangent Kernel (NTK)
    Gaussian Process dynamics, providing closed-form evolution of predictive
    mean and uncertainty under kernel gradient flow.

    Attributes:
        X_train (np.ndarray): Training inputs used to construct the NTK Gram matrix.
        y (np.ndarray): Training targets reshaped as a column vector.
        kernel_fn (Callable[[np.ndarray, np.ndarray], np.ndarray]): Kernel function implementing the NTK operator.
        cov_init_fn (Callable[[np.ndarray, np.ndarray], np.ndarray]): Covariance function defining the initial GP prior.
        K (np.ndarray): NTK Gram matrix computed on the training set.
        _eigvals (np.ndarray): Eigenvalues of the NTK Gram matrix for computing exp(-tΘ).
        _eigvecs (np.ndarray): Eigenvectors of the NTK Gram matrix forming its spectral basis.
        K_inv (np.ndarray): Regularized inverse of the NTK Gram matrix for closed-form regression.
        f0_XX (np.ndarray): Initial GP covariance on the training set.
        t (float): Current time parameter governing the NTK dynamics.

    Methods:
        reset() -> None:
            Resets the internal time parameter to zero.

        step(t: float) -> None:
            Updates the internal time parameter to the specified value.

        _exp_minus_Kt(t: float) -> np.ndarray:
            Computes the matrix exponential exp(-tΘ) via spectral decomposition.

        predict(X_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            Returns the predictive mean and marginal standard deviation at the
            specified input grid under NTK Gaussian Process dynamics.
    """

    def __init__(
        self,
        X_train: np.ndarray,
        y: np.ndarray,
        kernel_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        cov_init_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> None:
        self.X_train: np.ndarray = np.asarray(X_train)
        self.y: np.ndarray = np.asarray(y, dtype=float).reshape(-1, 1)
        self.kernel_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] = kernel_fn
        self.cov_init_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] = cov_init_fn

        self.K: np.ndarray = self.kernel_fn(self.X_train, self.X_train)
        eigvals, eigvecs = np.linalg.eigh(self.K)
        self._eigvals: np.ndarray = eigvals
        self._eigvecs: np.ndarray = eigvecs
        self.K_inv: np.ndarray = eigvecs @ np.diag(1.0 / (eigvals + 1e-10)) @ eigvecs.T

        self.f0_XX: np.ndarray = self.cov_init_fn(self.X_train, self.X_train)
        self.t: float = 0.0

    def reset(self) -> None:
        self.t = 0.0

    def step(self, t: float) -> None:
        self.t = float(t)

    def _exp_minus_Kt(self, t: float) -> np.ndarray:
        exp_eigs: np.ndarray = np.exp(-self._eigvals * t)
        return self._eigvecs @ np.diag(exp_eigs) @ self._eigvecs.T

    def predict(self, X_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_grid = np.asarray(X_grid, dtype=float)
        K_xX: np.ndarray = self.kernel_fn(X_grid, self.X_train)

        E: np.ndarray = self._exp_minus_Kt(self.t)
        I: np.ndarray = np.eye(self.K.shape[0], dtype=float)
        Z: np.ndarray = K_xX @ self.K_inv @ (I - E)

        mean: np.ndarray = Z @ self.y

        f0_xx: np.ndarray = self.cov_init_fn(X_grid, X_grid)
        f0_Xx: np.ndarray = self.cov_init_fn(self.X_train, X_grid)

        A: np.ndarray = Z @ self.f0_XX @ Z.T
        B: np.ndarray = Z @ f0_Xx
        C: np.ndarray = f0_xx + A - B - B.T
        diag_C: np.ndarray = np.clip(np.diag(C), 0.0, np.inf)
        sigma: np.ndarray = np.sqrt(diag_C).reshape(-1, 1)

        return mean, sigma