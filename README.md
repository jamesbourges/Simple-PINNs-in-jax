# Physics-Informed Neural Networks (PINNs) with JAX & Equinox

This repository contains two proof-of-concept implementations of Physics-Informed Neural Networks (PINNs) built with **JAX**, **Equinox**, and **Optax**. The notebooks explore progressively complex physical systems, demonstrating how neural networks can leverage partial physical knowledge to reconstruct trajectories and solve inverse problems from highly degraded, sparse data.

## 🧠 Core Architecture & Design Patterns

Both notebooks share a unified PINN design pattern:

* **Total Loss Formulation:** The objective function balances fitting the observed data with obeying the underlying physical laws.
    $$\mathcal{L}_{total} = \lambda_{phys} \sum \|	ext{PDE residual}\|^2 + \lambda_{data} \sum \|u_{	heta} - u_{obs}\|^2$$
* **Stochastic Regularization:** Collocation points (where the PDE residual is evaluated) are resampled randomly at every training step. This prevents the network from overfitting to a fixed grid and enforces the governing equations continuously across the domain, even where no empirical data exists.
* **Input Normalization:** Spatial and temporal coordinates are linearly scaled to $[-1, 1]$ prior to the first layer, keeping the network well-conditioned.

---

## PINN 1D: Damped Harmonic Oscillator (ODE)

This notebook models a damped harmonic oscillator governed by the equation:

$$m\ddot{u} + c\dot{u} + ku = 0$$

The network learns to predict the 1D trajectory $u(t)$ across three scenarios of increasing difficulty:

| Part | State of $m, c, k$ | Data Available | Objective |
| :--- | :--- | :--- | :--- |
| **1** | Known | 100 dense points | Learn the trajectory |
| **2** | Known | 10 sparse points | Reconstruct trajectory from sparse data |
| **3** | Unknown | 10 sparse points | **Inverse Problem:** Learn trajectory + parameters |

### The Inverse Problem & Non-Identifiability
Part 3 demonstrates a crucial property of parameter discovery. The parameters $m$, $c$, and $k$ are defined as trainable scalars inside the network. Because the underlying ODE is non-identifiable based purely on trajectory data—meaning any transformation $(m,c,k) 	o  lpha(m,c,k)$ yields the exact same solution—the absolute values of the parameters cannot be recovered. However, the network successfully recovers the physically meaningful ratios $c/m$ and $k/m$ to ~0.1% accuracy.

---

## PINN 2D: 1D Wave Equation (PDE)

This notebook scales the approach to a Partial Differential Equation with two inputs $(x, t)$, modeling a string with Dirichlet boundary conditions:

$$\partial_{tt}u = c^2\partial_{xx}u$$

### Key Features:
* **Extreme Data Sparsity:** The network is tasked with reconstructing the full 2D wave surface $u(x, t)$ using only ~300 randomly scattered observations out of a possible 10,000 grid points (approx. 3% of the data).
* **Parameter Discovery:** The wave speed $c$ is a trainable parameter. The PINN recovers it with high precision ($c  pprox 0.993$ vs. the true $c = 1.0$).
* **Computational Efficiency:** Computing the full Hessian for a PDE can be prohibitively expensive. Instead of materializing the full matrix, second-order partial derivatives ($\partial_{xx}$ and $\partial_{tt}$) are computed efficiently using forward-over-reverse automatic differentiation (`jax.jvp` over `jax.grad`), isolating only the required diagonal elements.

---

## Requirements

* `jax`
* `equinox`
* `optax`
* `scipy` (for the reference RK45 method of lines solver)
* `matplotlib` / `numpy`
