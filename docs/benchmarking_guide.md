# Benchmarking Guide

This repository contains three implementations of the Lid-Driven Cavity Flow simulation, designed to test performance across different backends on Apple Silicon (M1/M2/M3).

## Implementations

1.  **NumPy (`cavity_flow.py`)**:
    *   **Backend**: CPU (Python Interpreter).
    *   **Pros**: Simple, standard, works everywhere.
    *   **Cons**: Slow for large iterations due to interpreter overhead.

2.  **JAX CPU (`cavity_flow_jax.py`)**:
    *   **Backend**: CPU (JIT Compiled).
    *   **Pros**: Extremely fast for small/medium grids due to Just-In-Time compilation.
    *   **Cons**: Requires JAX installation.

3.  **JAX Metal (`cavity_flow_jax.py` in `venv_metal`)**:
    *   **Backend**: Apple GPU (Metal).
    *   **Pros**: Massively parallel. Best for very large grids ($> 256 \times 256$).
    *   **Cons**: High latency/overhead for small grids.

## Running Benchmarks

### 1. Automated Suite
The easiest way is to use the master script:

```bash
./run_benchmark.sh
```

**Custom Parameters:**
You can pass arguments to control grid size and time steps:

```bash
# High resolution test (101x101 grid)
./run_benchmark.sh --nx 101 --ny 101 --nt 1000

# Stability test with smaller time step
./run_benchmark.sh --dt 0.0001
```

### 2. Manual Execution
You can run individual scripts manually:

```bash
# NumPy
source venv/bin/activate
python simulations/cavity_flow/cavity_flow.py --benchmark --nx 51 --nt 500

# JAX (CPU)
source venv/bin/activate
export JAX_PLATFORMS=cpu
python simulations/cavity_flow/cavity_flow_jax.py --nx 51 --nt 500

# JAX (GPU)
source venv_metal/bin/activate
# Ensure Metal backend is used
python simulations/cavity_flow/cavity_flow_jax.py --nx 51 --nt 500
```

### 3. Stability & Resource Verification (Dry Run)

Before running a heavy simulation, you can check if your parameters are numerically stable (CFL condition) and estimate memory usage:

```bash
# Check if 200x200 grid with dt=0.001 is stable
./run_benchmark.sh --check --nx 200 --ny 200 --dt 0.001
```

**Output Example:**
```text
----------------------------------------
RESOURCE & STABILITY CHECK
----------------------------------------
Grid: 200x200  |  Cells: 40000
Memory (Est): 1.83 MB
...
❌ UNSTABLE: Time step too large for viscosity (Sigma > 0.5)
   Recommendation: dt < 0.000488
----------------------------------------
```

## Performance Expectations (Apple Silicon)

On an M1/M2/M3 chip, typical results for a **41x41 grid** are:

*   **NumPy**: ~1,200 it/s
*   **JAX CPU**: ~3,800 it/s (Fastest here due to low overhead)
*   **JAX Metal**: ~600 it/s (Slower due to GPU launch latency)

**When to use GPU?**
If you increase resolution to `--nx 256 --ny 256` or higher, the GPU (Metal) implementation will start to outperform the CPU version significantly.
