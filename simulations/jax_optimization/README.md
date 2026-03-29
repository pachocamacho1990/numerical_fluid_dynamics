# JAX CPU Optimization Experiments

This folder contains experiments for optimizing JAX CPU performance in numerical fluid dynamics simulations.

## Files

| File | Description |
|------|-------------|
| `bfs_baseline.py` | Original 2D backward-facing step solver |
| `bfs_optimized.py` | Optimized with `donate_argnums` + `lax.scan` |
| `benchmark_jax.py` | Benchmark comparison script |
| `profile_jax.py` | JAX profiler for bottleneck identification |

## Optimizations Applied

1. **Red-Black SOR Solver** - Replaced Jacobi solver (50 iterations) with Red-Black Successive Over-Relaxation ($\omega=1.9$, 15 iterations). This provided the largest speedup by reducing the pressure solve cost by ~3x while maintaining stability.
2. **`lax.scan` Main Loop** - Compiled the entire transient simulation loop into a single XLA kernel, eliminating Python dispatch overhead.
3. **`donate_argnums`** - Buffer reuse to minimize memory allocation (applied to Python loop version).
4. **`jax_enable_x64`** - Consistent 64-bit precision.

## Benchmark Results

### Final Performance (Grid 201x51, 20,000 steps)

| Configuration | Steps/sec | Speedup | Max U (Final) |
|---------------|-----------|---------|---------------|
| **Baseline (Jacobi-50)** | 857 | 1.00x | 1.4999 |
| **Optimized (SOR-15 + scan)** | **1324** | **1.54x** | **1.4999** |

**Key Findings:**
- The **Red-Black SOR solver** significantly outperforms the baseline Jacobi solver. By using optimal relaxation ($\omega=1.9$), we achieve faster convergence, allowing us to reduce iterations from 50 to 15 without reduced accuracy.
- **XLA compilation** (`lax.scan`) provides additional overhead reduction, crucial for smaller grids.
- The combination yields a **54% overall speedup** for production-length simulations with verified long-term stability.

## Usage

```bash
# Run optimized solver
JAX_PLATFORMS=cpu python simulations/jax_optimization/bfs_optimized.py --nt 10000

# Compare modes
JAX_PLATFORMS=cpu python simulations/jax_optimization/bfs_optimized.py --nt 5000 --use-python-loop

# Run profiler
JAX_PLATFORMS=cpu python simulations/jax_optimization/profile_jax.py
```
