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

1. **`donate_argnums`** - Buffer reuse to avoid memory allocation overhead
2. **`lax.scan` main loop** - Compile entire simulation as single XLA kernel  
3. **`jax_enable_x64`** - Consistent 64-bit precision

## Benchmark Results

### Grid Scaling (2000 steps each)

| Grid | Cells | lax.scan | Python Loop | Speedup |
|------|-------|----------|-------------|---------|
| 101×26 | 2.6K | 3104 steps/s | 2680 steps/s | **1.16x** |
| 201×51 | 10K | 845 steps/s | 816 steps/s | **1.04x** |
| 401×101 | 40K | 233 steps/s | 227 steps/s | **1.03x** |
| 801×201 | 161K | 84 steps/s | 81 steps/s | **1.04x** |

**Finding:** `lax.scan` shows best gains at small grids (16%) where Python loop overhead is significant. At larger grids, gains converge to 3-4% as compute dominates.

## Usage

```bash
# Run optimized solver
JAX_PLATFORMS=cpu python simulations/jax_optimization/bfs_optimized.py --nt 10000

# Compare modes
JAX_PLATFORMS=cpu python simulations/jax_optimization/bfs_optimized.py --nt 5000 --use-python-loop

# Run profiler
JAX_PLATFORMS=cpu python simulations/jax_optimization/profile_jax.py
```
