"""Public API for function-space dynamics and kernel-driven evolution models."""

from .ntk_gp_dynamics import NTKGaussianProcess

__all__ = [
    "NTKGaussianProcess",
]