# Infinite-Width NTK Evolution

## Introduction  

This repository is designed for investigating the evolution of infinite‑width neural networks under the training regime induced by the Neural Tangent Kernel (NTK). The implemented methodology corresponds to the infinite‑width linearization formalism originally introduced by LEE J., XIAO L., SCHOENHOLZ S. S., BAHRI Y., NOVAK R., SOHL‑DICKSTEIN J., and PENNINGTON J. (2019) in their paper *“Wide Neural Networks of Any Depth Evolve as Linear Models Under Gradient Descent”* (Proceedings of the 33rd International Conference on Neural Information Processing Systems, DOI: [10.5555/3454287.3455056](https://dl.acm.org/doi/10.5555/3454287.3455056)).

An NTK provides a deterministic characterization of training behavior in the infinite‑width limit, where parameter updates linearize around initialization and induce a fixed kernel governing function evolution. Under these conditions, predictive response evolves through kernel gradient flow, yielding closed‑form trajectories for the mean and variance of the induced function over time.

## Getting Started

To set up the repository properly, follow these steps:

**1.** **Set Up the Python Environment**  

- To create and activate the virtual environment defined in `pyproject.toml` and `uv.lock`, execute the following command:

  ```bash
  uv sync
  source .venv/bin/activate  # On Windows use: .venv\Scripts\activate 
  ```

**2.** **Run the NTK Implementation**  

- The `src/` folder contains the modular components of the NTK implementation:
  - `dynamics/ntk_gp_dynamics.py`: Defines the NTK Gaussian Process dynamics, enabling closed‑form evolution of predictive means and uncertainties.
  - `kernels/nngp_relu.py`: Computes the corresponding NNGP kernel defining the initialization‑time covariance structure of the infinite‑width ReLU model. 
  - `kernels/ntk_relu.py`: Computes the analytic Neural Tangent Kernel for infinite‑width ReLU networks using arc‑cosine closed‑form expressions.
  - `plotting/plot_dynamics.py`: Generates multi‑panel visualizations of NTK evolution across selected time values.
  - `main.py`: Executes the NTK computation workflow, orchestrating kernel construction, dynamics evaluation, and saving the final figure in the `figures/` directory.

- Run the following command to execute the full workflow:

  ```bash
  python main.py
  ```


## License  

This project is licensed under the **MIT License**, which allows for open-source use, modification, and distribution with minimal restrictions. For more details, refer to the file included in this repository. 
