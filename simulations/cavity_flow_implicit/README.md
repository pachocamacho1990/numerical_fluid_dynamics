# Lid-Driven Cavity Flow - Implicit Scheme

> **Status:** 📚 *Documentation Phase* - Mathematical formulation ready for study. Implementation pending.

## Overview

This directory will contain an **implicit numerical scheme** for the 2D lid-driven cavity flow problem, designed to overcome the CFL stability limitations of the explicit method in [`../cavity_flow/`](../cavity_flow/).

## Why Implicit?

The explicit scheme has **conditional stability**:

- **CFL Condition:** $\Delta t \cdot \left(\frac{|u|}{\Delta x} + \frac{|v|}{\Delta y}\right) < 1$

- **Diffusion Stability:** $\Delta t < \frac{\Delta x^2}{4\nu}$

For high Reynolds numbers, these constraints force very small time steps, making simulations expensive or impossible.

**Implicit schemes** remove these constraints, allowing:
- Stable simulations at **any Reynolds number**
- Much larger time steps (100× or more)
- Faster convergence to steady-state

## Mathematical Background

See the detailed theory documentation:
- **[Implicit Schemes Theory](../../docs/implicit_schemes_theory.md)** - Comprehensive mathematical derivation
- **[Explicit Scheme Theory](../../docs/cavity_flow_theory.md)** - Original explicit method (for comparison)

## Planned Method: Fractional Step (Projection)

The implementation will use the **Chorin-Temam Projection Method**:

1. **Predictor:** Compute intermediate velocity with implicit diffusion
2. **Pressure Solve:** Enforce incompressibility via Poisson equation
3. **Corrector:** Update velocity to be divergence-free

### Key Algorithms
- **Crank-Nicolson:** 2nd-order accurate implicit time stepping for diffusion
- **ADI (Alternating Direction Implicit):** Efficient solution of 2D implicit systems

## Directory Structure

```
cavity_flow_implicit/
├── README.md                       # This file
├── cavity_flow_implicit.py         # NumPy implementation (planned)
├── cavity_flow_implicit_jax.py     # JAX implementation (planned)
├── outputs/
│   └── baseline/                   # Simulation data
└── plots/                          # Visualization outputs
```

## Usage (Planned)

```bash
# Basic run
python simulations/cavity_flow_implicit/cavity_flow_implicit.py

# High Reynolds number (stable unlike explicit scheme!)
python simulations/cavity_flow_implicit/cavity_flow_implicit.py --re 10000 --nx 201 --nt 10000

# With stability monitoring
python simulations/cavity_flow_implicit/cavity_flow_implicit.py --verbose
```

## References

- Chorin, A.J. (1968). "Numerical solution of the Navier-Stokes equations"
- Temam, R. (1969). "Sur l'approximation de la solution des équations de Navier-Stokes"
- Peyret, R. & Taylor, T.D. (1983). "Computational Methods for Fluid Flow"
- Original inspiration: [barbagroup/CFDPython](https://github.com/barbagroup/CFDPython)

## Acknowledgment

This builds upon the explicit implementation based on Lorena A. Barba's [CFD Python](https://github.com/barbagroup/CFDPython) course.
