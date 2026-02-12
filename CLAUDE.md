# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CFD learning project implementing 2D and 3D incompressible Navier-Stokes solvers using Finite Difference Methods. Each problem has NumPy baselines and JAX-accelerated variants (CPU and Apple Metal GPU). Extends from Barba's "12 Steps to Navier-Stokes" into implicit methods, complex geometries, and Large Eddy Simulation.

## Environment Setup

```bash
# JAX CPU (recommended)
python -m venv venv && source venv/bin/activate && pip install -r requirements-cpu.txt

# JAX Metal GPU (Apple Silicon — pinned to JAX 0.5.0 / jax-metal 0.1.1)
python -m venv venv_metal && source venv_metal/bin/activate && pip install -r requirements-metal.txt

# 3D Visualization (PyVista/VTK — requires Python 3.10-3.13)
./setup_vis_env.sh
```

## Running Simulations

All scripts use `argparse`. Common flags: `--nx`, `--nt`, `--dt`, `--re`, `--output-dir`, `--benchmark`, `--verbose`.

```bash
# 2D Explicit cavity flow
python simulations/cavity_flow/cavity_flow.py --re 1000 --nx 201 --nt 5000 --dt 0.0001
python simulations/cavity_flow/cavity_flow_jax.py

# 2D Implicit cavity flow
python simulations/cavity_flow_implicit/cavity_flow_implicit.py
python simulations/cavity_flow_implicit/cavity_flow_implicit_jax.py

# 2D Backward-facing step
python simulations/backward_facing_step/backward_facing_step.py
python simulations/backward_facing_step_implicit/backward_facing_step_implicit.py

# 3D Backward-facing step
python simulations/backward_facing_step_3d/run_turbulence_3d.py

# 3D LES
python simulations/les/run_les_3d.py --re 5000 --nx 128 --ny 32 --nz 32 --nt 1000 --cs 0.12
python simulations/les/run_dns_comparison.py --re 2000 --nx 192 --ny 48 --nz 48 --nt 120000

# Channel flow validation
python simulations/channel_flow_3d/run_channel_les.py --re_tau 180
```

## Running Tests

Tests are standalone scripts (no pytest framework):

```bash
# Ghia et al. benchmark validation
python simulations/experiments/test_ghia_benchmark.py
python simulations/experiments/test_ghia_benchmark.py --solver jax_implicit --re-list 100,400,1000

# Fractional step unit tests
python simulations/cavity_flow_implicit/test_fractional.py

# LES unit tests
python simulations/les/tests/test_sgs_models.py
python simulations/les/tests/test_variable_adi.py
python simulations/les/tests/test_wall_treatment.py

# LES physics verification (full integration test)
python simulations/les/verify_physics.py

# Reattachment length validation (Armaly benchmark)
python simulations/experiments/validate_reattachment.py
python simulations/les/compare_armaly.py
```

## Architecture

### Solver Progression

The project follows an increasing-complexity progression:

1. **Explicit 2D** — Pressure-Poisson (Jacobi) + explicit Euler. Stability-limited: CFL < 1.0, `dt < dx²/(4ν)`.
2. **Implicit 2D** — Chorin-Temam fractional step (predictor → pressure Poisson → corrector). Crank-Nicolson diffusion via ADI. Unconditionally stable for diffusion.
3. **Implicit 3D** — Extended ADI with 3 directional sweeps (x, y, z). Geometry masking for backward-facing step.
4. **LES 3D** — Smagorinsky SGS model + Van Driest wall damping + variable-viscosity ADI. Validated against DNS (1.5% error, 24x speedup).

### Key Module Relationships

```
simulations/common/
  solvers.py              — SimulationConfig/Result dataclasses + SOLVER_REGISTRY
  plotting.py             — Shared visualization utilities

simulations/cavity_flow_implicit/
  cavity_flow_implicit.py → fractional_step.py → adi_solver.py → thomas_solver.py
  (JAX variants: _jax suffix on each file)

simulations/backward_facing_step/
  backward_facing_step.py → geometry.py (Armaly benchmark mask, ER=1.94)

simulations/backward_facing_step_3d/
  solver_3d.py → geometry_3d.py + thomas_solver_jax.py

simulations/les/
  solver_3d_les.py → sgs_models.py (Smagorinsky eddy viscosity)
                   → variable_viscosity_adi.py → thomas algorithm (jax.lax.scan)
                   → wall_treatment.py (Van Driest damping)
                   → config.py (LESConfig dataclass)
```

### Solver Registry (`simulations/common/solvers.py`)

Registered solvers: `jax_explicit`, `numpy_implicit`, `jax_implicit`. Used by `experiments/runner.py` and `test_ghia_benchmark.py` for solver-agnostic benchmarking.

### Geometry Masking

Complex domains use binary mask arrays (1=fluid, 0=solid). In 3D: `mask[k,j,i]`. Applied after every momentum update. The step solid region is `{(x,y,z) : x < 0 AND y < h}`.

### LES Pipeline (per timestep)

1. Compute strain rate tensor (6 components) → Smagorinsky viscosity `ν_t = (C_s·Δ)²·|S|`
2. Apply Van Driest wall damping → `ν_eff = ν + ν_t`
3. Explicit advection (upwind) + explicit diffusion (flux form with ν_eff)
4. Implicit diffusion via variable-viscosity ADI (3 directional sweeps, modified tridiagonal coefficients)
5. Pressure Poisson → velocity correction → boundary conditions

The variable-viscosity ADI uses interface-averaged viscosities `ν_{i±1/2}` in the tridiagonal coefficients, rebuilding them each timestep.

## File Conventions

- `outputs/` — data files (`.npz`, `.csv`), **gitignored**
- `plots/` — visualization PNGs/GIFs, **tracked in git**
- `docs/` — mathematical derivations (LaTeX in GitHub-flavored Markdown)
- JAX variants suffixed `_jax.py` (e.g., `cavity_flow.py` → `cavity_flow_jax.py`)
- Each simulation folder has its own `README.md`

## Coding Conventions

- NumPy for baselines, JAX for acceleration (`jax.jit`, `jax.vmap`, `lax.fori_loop`, `lax.scan`)
- Docstrings document the mathematical equations being solved
- Output paths: `os.path.join()` with `os.makedirs(exist_ok=True)`
- CLI: `argparse` with `--nx`, `--nt`, `--dt`, `--re`, `--output-dir`, `--benchmark`
- LaTeX in docs: blank lines before/after `$$` blocks, never indent `$$` inside lists

## Workflow for New Simulations

1. Create theory doc in `docs/`
2. Create folder in `simulations/` with NumPy and JAX implementations
3. Add `argparse`, `--output-dir`/`--input-dir`, visualization saving to `plots/`
4. Verify stability and physical realism
5. Add README with usage and results

## Current Version

Latest release: check with `gh release list --limit 5`
