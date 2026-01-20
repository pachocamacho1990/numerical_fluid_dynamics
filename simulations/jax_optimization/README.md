# JAX CPU Optimization Experiments

This folder contains experiments for optimizing JAX CPU performance in numerical fluid dynamics simulations.

## Files

| File | Description |
|------|-------------|
| `bfs_baseline.py` | Original 2D backward-facing step solver (baseline) |
| `bfs_optimized.py` | Optimized version with `donate_argnums` + `lax.scan` |
| `benchmark_jax.py` | Benchmark script comparing baseline vs optimized |

## Optimizations Applied

1. **`donate_argnums`** - Buffer reuse to avoid memory allocation overhead
2. **`lax.scan` main loop** - Compile entire simulation as single XLA kernel  
3. **`jax_enable_x64`** - Consistent 64-bit precision

## Usage

```bash
# Run benchmark comparison
JAX_PLATFORMS=cpu python simulations/jax_optimization/benchmark_jax.py --nt 2000

# Run optimized solver directly
JAX_PLATFORMS=cpu python simulations/jax_optimization/bfs_optimized.py --nt 10000

# Compare with Python loop version
JAX_PLATFORMS=cpu python simulations/jax_optimization/bfs_optimized.py --nt 10000 --use-python-loop
```

## Expected Results

| Configuration | Speedup |
|--------------|---------|
| Baseline (Python loop) | 1.0x |
| donate_argnums only | ~1.1-1.3x |
| lax.scan + donate_argnums | ~2-5x |
