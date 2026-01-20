#!/usr/bin/env python3
"""
JAX Performance Profiler - Backward-Facing Step Solver

Times individual components in isolation.

Usage:
    JAX_PLATFORMS=cpu python profile_jax.py
    JAX_PLATFORMS=cpu python profile_jax.py --nx 401 --ny 101
"""

import os
import sys
import time
import argparse
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import jit, lax

from geometry import setup_geometry, get_inlet_profile, calculate_nu_from_reynolds
from bfs_optimized import time_step_jax_inner, build_up_b_jax, solve_pressure_jax


def time_function(fn, n_runs=50):
    """Time a function that returns a JAX array."""
    # Warmup
    result = fn()
    if hasattr(result, 'block_until_ready'):
        result.block_until_ready()
    elif isinstance(result, tuple):
        result[0].block_until_ready()
    
    # Time
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = fn()
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()
        elif isinstance(result, tuple):
            result[0].block_until_ready()
        times.append(time.perf_counter() - start)
    
    return np.mean(times) * 1000, np.std(times) * 1000  # ms


def profile_simulation(args):
    """Run timing breakdown."""
    
    print("=" * 60)
    print("JAX PERFORMANCE PROFILER")
    print("=" * 60)
    print(f"Backend: {jax.default_backend()}")
    print(f"Grid: {args.nx} x {args.ny} ({args.nx * args.ny:,} cells)")
    
    # Setup
    geom = setup_geometry(args.nx, args.ny)
    dims = geom['dims']
    dx, dy = geom['dx'], geom['dy']
    nu = calculate_nu_from_reynolds(args.re, args.u_max, dims['h'])
    rho = 1.0
    nit = 50
    dt = args.dt
    
    # Initialize
    u = jnp.zeros((args.ny, args.nx))
    v = jnp.zeros((args.ny, args.nx))
    p = jnp.zeros((args.ny, args.nx))
    
    mask = jnp.array(geom['mask'])
    inlet_y = geom['y'][geom['j_step']:] - geom['y'][geom['j_step']]
    u_inlet = jnp.array(get_inlet_profile(inlet_y, dims['h_inlet'], args.u_max))
    j_step = geom['j_step']
    i_step = geom['i_step']
    
    # Pre-compute b for pressure timing
    b = build_up_b_jax(rho, dt, u, v, dx, dy, mask)
    b.block_until_ready()
    
    n_runs = args.n_runs
    
    print(f"\nTiming components ({n_runs} runs each)...\n")
    
    # Time each component
    t_rhs, std_rhs = time_function(
        lambda: build_up_b_jax(rho, dt, u, v, dx, dy, mask), n_runs)
    print(f"  Pressure RHS (build_up_b): {t_rhs:8.3f} ± {std_rhs:.3f} ms")
    
    t_pressure, std_pressure = time_function(
        lambda: solve_pressure_jax(p, dx, dy, b, nit), n_runs)
    print(f"  Pressure Poisson (50 it):  {t_pressure:8.3f} ± {std_pressure:.3f} ms")
    
    t_full, std_full = time_function(
        lambda: time_step_jax_inner(u, v, p, dt, dx, dy, rho, nu, nit,
                                     mask, u_inlet, j_step, i_step), n_runs)
    print(f"  Full time step:            {t_full:8.3f} ± {std_full:.3f} ms")
    
    # Calculate derived metrics
    t_other = t_full - t_rhs - t_pressure  # Advection + BC + CFL
    
    print(f"\n" + "=" * 60)
    print("BREAKDOWN")
    print("=" * 60)
    print(f"\n  {'Component':<25} {'Time (ms)':<12} {'Share':<8}")
    print(f"  {'-'*50}")
    print(f"  {'Pressure RHS':<25} {t_rhs:8.3f}     {t_rhs/t_full*100:5.1f}%")
    print(f"  {'Pressure Poisson (50 it)':<25} {t_pressure:8.3f}     {t_pressure/t_full*100:5.1f}%")
    print(f"  {'Advection + BC + CFL':<25} {t_other:8.3f}     {t_other/t_full*100:5.1f}%")
    print(f"  {'-'*50}")
    print(f"  {'TOTAL':<25} {t_full:8.3f}     100.0%")
    print(f"\n  Effective rate: {1000/t_full:.0f} steps/s")
    
    # Analysis
    print(f"\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    
    components = [
        ("Pressure Poisson", t_pressure),
        ("Pressure RHS", t_rhs),
        ("Advection/BC", t_other)
    ]
    
    bottleneck_name, bottleneck_time = max(components, key=lambda x: x[1])
    
    print(f"\n  ⮕ Bottleneck: {bottleneck_name} ({bottleneck_time/t_full*100:.0f}% of time)")
    
    if bottleneck_name == "Pressure Poisson":
        print(f"""
  The iterative Jacobi pressure solver dominates.
  50 iterations × {t_pressure/50:.3f} ms/iter = {t_pressure:.1f} ms
  
  Optimization options:
    1. Reduce iterations if convergence is reached earlier
    2. Implement multigrid (10-100x faster convergence)
    3. Use SOR instead of Jacobi (1.5-2x faster)
    4. Red-Black Gauss-Seidel (better parallelism)
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JAX Performance Profiler")
    parser.add_argument("--nx", type=int, default=201)
    parser.add_argument("--ny", type=int, default=51)
    parser.add_argument("--re", type=float, default=100)
    parser.add_argument("--dt", type=float, default=0.0002)
    parser.add_argument("--u-max", type=float, default=1.5)
    parser.add_argument("--n-runs", type=int, default=50, help="Timing iterations")
    args = parser.parse_args()
    
    profile_simulation(args)
