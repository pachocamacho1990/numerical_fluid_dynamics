# 3D Backward-Facing Step Simulation

A fully implicit 3D Navier-Stokes solver for the backward-facing step geometry, extending the classic Armaly et al. benchmark to 3D.

## Features
- **Method**: 3D Fractional Step with Masked ADI
- **Advection**: First-order upwind (stable for high Re)
- **Time Integration**: Crank-Nicolson (2nd order)
- **Grid**: Staggered grid formulation
- **Parallelization**: JAX-based `vmap` for efficient batch processing

## Usage

### 1. Visualization Setup
This 3D simulation requires a specific environment for visualization due to VTK/PyVista compatibility.
Please see [Visualization Setup](visualization_setup.md) for detailed instructions.

```bash
# Quick setup
./setup_vis_env.sh
```

### 2. Running Simulations
The solver uses JAX and can run on CPU or GPU.

```bash
# Standard run (Re=400, default grid)
venv/bin/python simulations/backward_facing_step_3d/solver_3d.py

# High Reynolds number
venv/bin/python simulations/backward_facing_step_3d/solver_3d.py --re 800 --nx 201 --ny 51 --nz 51
```

### 3. Visualization
```bash
# Interactive 3D view
venv_vis/bin/python simulations/backward_facing_step_3d/visualize_3d_isosurfaces.py --interactive
```

## Theory
Detailed mathematical derivation is available in [`docs/backward_facing_step_3d_theory.md`](../../docs/backward_facing_step_3d_theory.md).

## Files
- `solver_3d.py`: Main JAX solver
- `geometry_3d.py`: Domain generation and masking
- `thomas_solver_jax.py`: Tridiagonal solver utility
- `validate_3d.py`: Validation script and plotting
- `visualize_3d_isosurfaces.py`: Advanced PyVista visualization
