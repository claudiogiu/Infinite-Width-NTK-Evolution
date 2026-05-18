import numpy as np
import warnings
from kernels.ntk_relu import NTKReLUKernel
from kernels.nngp_relu import NNGPReLUKernel
from dynamics.ntk_gp_dynamics import NTKGaussianProcess
from plotting.plot_dynamics import DynamicsPlotter

warnings.filterwarnings("ignore")

def main() -> None:
    X = np.array([[0.2, 1.0],
                  [0.4, 1.0],
                  [0.8, 1.0]])
    y = np.array([[-0.05], [-0.2], [0.3]])

    x_vals = np.linspace(0.0, 1.0, 200)
    X_grid = np.stack([x_vals, np.ones_like(x_vals)], axis=1)

    times = [0.0, 0.5, 1.5, 5.0, 25.0, 500.0]

    ntk = NTKReLUKernel()
    nngp = NNGPReLUKernel()

    dynamics = NTKGaussianProcess(
        X_train=X,
        y=y,
        kernel_fn=ntk,
        cov_init_fn=nngp,
    )

    dyn_plotter = DynamicsPlotter(filename="ntk_gaussian_process.png")
    dyn_plotter.plot(
        X=X[:, 0:1],
        y=y,
        X_grid=X_grid,
        times=times,
        dynamics=dynamics,
    )

if __name__ == "__main__":
    main()