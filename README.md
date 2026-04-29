# Advanced Computing for Physics | UAM

This repository contains computational physics simulations and numerical methods developed for the **Advanced Computing** course at Universidad Autónoma de Madrid (UAM).

## Overview
The projects focus on implementing numerical solutions for physical systems where analytical methods are insufficient. Each module explores different computational paradigms, from deterministic ODE solvers to stochastic Monte Carlo methods and Molecular Dynamics.


## Repository Structure

The code is organized into thematic modules following the physical phenomena studied:

### 1 & 2. Introduction to Python Methods
- **Basic operations and functions**: Creation of personal functions using basic operations
- **Importing data**: Obtaining and reading data from a .txt file.
  
### 3. Numerical Foundations & Ballistics
- **Numerical Methods:** Implementation of integration and differentiation algorithms.
- **Projectile Motion:** Realistic simulations including air resistance, altitude-dependent density, and the Magnus effect.

### 4. Nonlinear Dynamics & Chaos
- **Pendulum Systems:** Comparative analysis of Euler-Cromer vs. standard methods.
- **Phase Space:** Poincaré sections and state-space visualization.
- **Lorenz Attractor:** Exploration of chaotic behavior and strange attractors.

### 5. Orbital Mechanics (N-Body Problems)
- **Keplerian Motion:** Precision of planetary orbits and the inverse square law.
- **Three-Body Problem:** Stability analysis and precession of Halley’s perihelion.
- **Lagrange points:** Location and orbits around Lagrange points.

### 6. Stochastic Processes & Statistical Physics
- **Random Walks:** Diffusion models and entropy evolution.

### 7. Monte Carlo Methods
- **Monte Carlo Integration**: Numerical estimation of multidimensional integrals using stochastic sampling techniques.
- **Ising Model**: Simulation of spin systems via the Metropolis algorithm.
 
### 8. Potentials & Elliptic PDEs (Boundary Value Problems)
- **Finite Differences**: Solving the Laplace ($\nabla^2 \phi = 0$) and Poisson equations using Gauss-Seidel and Successive Over-Relaxation (SOR) methods.
- **Computer Vision**: Artificial detection of contours and edges of images for the automatic extraction of geometrical features that can be interpreted as physical boundary conditions or charge distributions, enabling the numerical solution of the Poisson equation on arbitrary domains derived directly from visual input. 

### 9. Diffusion & Parabolic PDEs (Initial Value Problems)
- **Heat Transfer**: Modeling the diffusion equation through finite difference schemes in both time and space.
- **Numerical Schemes**: Implementation and stability analysis of explicit (Forward Time Centered Space) and implicit (Backward Time Centered Space) methods.

### 10. Waves & Hyperbolic PDEs (Initial Value Problems)
- **Wave Equation**: Simulation of wave propagation in ideal media and analysis of the resulting frequency spectrum.
- **Advanced Solvers**: Implementation of the Crank-Nicolson algorithm and spectral methods for high-precision dynamic modeling.

## Reports & Documentation
Detailed physical analysis and numerical results are available in the `/reports` directory. Each study includes:

* **PDF Reports:** Compiled documents featuring theoretical derivations, error analysis, and graphical results.
* **LaTeX Source:** Original `.tex` files and assets, ensuring full transparency and reproducibility of the scientific documentation.

## Technical Stack
- **Language:** Python
- **Core Libraries:** `NumPy`, `SciPy`, `Matplotlib`.
- **Environment:** VS Code.

## Author
**Jorge Acebes**: BSc Physics Student – Universidad Autónoma de Madrid (UAM).
