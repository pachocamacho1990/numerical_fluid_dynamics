# Lid-Driven Cavity Flow

A 2D simulation of flow in a square cavity with a moving top lid, implemented using the Finite Difference Method. This follows "Step 11" of Lorena Barba's "12 Steps to Navier-Stokes".

## Method
- **Equations**: 2D Incompressible Navier-Stokes
- **Advection**: Backward Difference (Upwind)
- **Diffusion**: Central Difference
- **Pressure**: Iterative Poisson Solver (50 iterations)
- **Time Stepping**: First-Order Explicit Euler

## Outcome
The simulation produces a characteristic central vortex with secondary corner vortices at higher Reynolds numbers.

![Pressure Field and Streamlines](plots/cavity_flow_result.png)

## Implementations

### NumPy Baseline (`cavity_flow.py`)
Standard implementation using NumPy arrays.

**Usage:**
```bash
# Default: 41x41 grid, Re~20, 500 steps
python simulations/cavity_flow/cavity_flow.py

# Custom Reynolds number
python simulations/cavity_flow/cavity_flow.py --re 1000 --nx 201 --nt 5000 --dt 0.0001

# Benchmark mode (minimal output)
python simulations/cavity_flow/cavity_flow.py --benchmark
```

**Arguments:**
- `--nx INT`: Grid points in X and Y (default: 41)
- `--nt INT`: Number of time steps (default: 500)
- `--re FLOAT`: Reynolds number (overrides nu)
- `--dt FLOAT`: Time step size (default: 0.001)
- `--output-dir PATH`: Output directory for data files (default: outputs/baseline)
- `--benchmark`: Minimal output mode

### JAX Implementation (`cavity_flow_jax.py`)
Accelerated version using JAX with JIT compilation.

**Usage:**
```bash
# Auto-detect backend (CPU or Metal)
python simulations/cavity_flow/cavity_flow_jax.py

# Force CPU backend
JAX_PLATFORMS=cpu python simulations/cavity_flow/cavity_flow_jax.py

# High Reynolds number
python simulations/cavity_flow/cavity_flow_jax.py --re 1000 --nx 201 --nt 5000 --dt 0.0001
```

**Arguments:** Same as NumPy version

> [!NOTE]
> For small grids (41x41), **JAX CPU** is fastest (~5.7x faster than NumPy). Metal GPU has overhead that dominates for small problems.

## Output Structure

```
simulations/cavity_flow/
├── outputs/
│   ├── baseline/          # Default benchmark data
│   │   ├── solution_numpy.npz
│   │   ├── solution_jax_cpu.npz
│   │   ├── solution_jax_metal.npz
│   │   ├── cfl_numpy.csv
│   │   ├── cfl_jax_cpu.csv
│   │   └── cfl_jax_metal.csv
│   └── high_re/           # High Reynolds data
│       └── ...
└── plots/                 # Visualizations (tracked in git)
    ├── cavity_flow_result.png
    ├── cfl_comparison.png
    ├── solutions_comparison.png
    ├── stability_velocity.png
    └── stability_cfl.png
```

## Benchmarking

### Baseline Benchmark
Compare all three implementations (NumPy, JAX CPU, JAX Metal) on default settings:
```bash
./run_benchmark.sh
```

Generates:
- `plots/cfl_comparison.png` - CFL evolution comparison
- `plots/solutions_comparison.png` - Side-by-side solution comparison

### High Reynolds Benchmark
Test Re=1000 on 201x201 grid:
```bash
./run_high_re.sh
```

Generates:
- `plots/solutions_comparison_re1000.png`

## Analysis Results

### CFL Analysis
The Courant-Friedrichs-Lewy (CFL) number is tracked for each time step to monitor stability.

![CFL Comparison](plots/cfl_comparison.png)

### Solutions Comparison
Comparison of the final state (Pressure + Streamlines) across all implementations.

![Solutions Comparison](plots/solutions_comparison.png)

### Stability Analysis
We investigated the stability of the numerical scheme by increasing the Reynolds number on a $201 \times 201$ grid with $\Delta t = 0.0001$.

![Stability Velocity](plots/stability_velocity.png)
![Stability CFL](plots/stability_cfl.png)

> [!IMPORTANT]
> **Resolution Matters**: The simulation is **stable at Re=1000** on a 201×201 grid. However, instability occurs at Re≥2500 where the simulation crashes around step 4500. This confirms that higher Re requires finer meshes to resolve the thinner boundary layers.

### High Reynolds Comparison ($Re=1000$)
Using the $201 \times 201$ grid, we successfully computed stable flow at $Re=1000$.

![Solution Re1000](plots/solutions_comparison_re1000.png)

> [!NOTE]
> **Performance** (201×201 grid, 5000 steps):
> - **NumPy**: ~39s
> - **JAX (CPU)**: ~11s (**3.5x faster**)
> - **JAX (Metal)**: ~22s (Slower than CPU due to dispatch overhead)

## Theory

See [Cavity Flow Theory](../../docs/cavity_flow_theory.md) for detailed mathematical derivations and discretization schemes.
