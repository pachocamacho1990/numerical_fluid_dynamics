# Lid-Driven Cavity Flow

A 2D simulation of flow in a square cavity with a moving top lid, implemented using the Finite Difference Method. This follows "Step 11" of Lorena Barba's "12 Steps to Navier-Stokes".

## Method
- **Equations**: 2D Incompressible Navier-Stokes
- **Advection**: Backward Difference (Upwind)
- **Diffusion**: Central Difference
- **Pressure**: Iterative Poisson Solver
- **Time Stepping**: First-Order Explicit

## Outcome
The simulation produces a characteristic central vortex.

![Pressure Field and Streamlines](cavity_flow_result.png)


## JAX Implementation
A JAX-accelerated version is available in `cavity_flow_jax.py`.
- **Method**: Same physical scheme, but uses `jax.jit` and `jax.lax.fori_loop` for the pressure solver.
- **Run**:
  ```bash
  python simulations/cavity_flow/cavity_flow_jax.py
  ```

> [!NOTE]
> For small grid sizes (like 41x41), the **JAX CPU** backend is typically fastest due to lower overhead compared to Metal/GPU.

## CFL Analysis
The Courant-Friedrichs-Lewy (CFL) number is tracked for each time step to monitor stability.

![CFL Comparison](cfl_comparison.png)

## Solutions Comparison
Comparison of the final state (Pressure + Streamlines) across all implementations.

![Solutions Comparison](solutions_comparison.png)

## Stability Analysis
We investigated the stability of the numerical scheme by increasing the Reynolds number (Re) on a finer grid ($81 \times 81$) with $\Delta t = 0.0005$.

![Stability Velocity](stability_velocity.png)
![Stability CFL](stability_cfl.png)

> [!IMPORTANT]
> **Resolution Matters**: increasing the grid to $201 \times 201$ makes the simulation **stable at Re=1000** (which previously crashed). However, instability returns at higher Reynolds numbers ($Re \ge 2500$) where it crashes around step 4500. This confirms that higher Re requires finer meshes ($Re_{\Delta x}$ constraint).

## High Reynolds Comparison ($Re=1000$)
Using the finer $201 \times 201$ grid, we successfully computed the stable flow at $Re=1000$.

![Solution Re1000](solutions_comparison_re1000.png)

> [!NOTE]
> **Performance**:
> - **NumPy**: ~39s
> - **JAX (CPU)**: ~11s (**3.5x faster**)
> - **JAX (Metal)**: ~22s (Slower than CPU for this grid size due to dispatch overhead)

## Usage
Run from the repository root:
```bash
python simulations/cavity_flow/cavity_flow.py
```

