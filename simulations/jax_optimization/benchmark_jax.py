#!/usr/bin/env python3
"""
JAX CPU Optimization Benchmark Script

Compares performance between:
1. Baseline solver (Python loop)
2. Optimized solver with lax.scan
3. Optimized solver with Python loop (donate_argnums only)

Usage:
    JAX_PLATFORMS=cpu python benchmark_jax.py
    JAX_PLATFORMS=cpu python benchmark_jax.py --nt 5000 --runs 3
"""

import os
import sys
import time
import argparse
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

# Import geometry from local copy
from geometry import setup_geometry, get_inlet_profile, calculate_nu_from_reynolds


def run_benchmark(solver_module, solver_func, nt, geom, params, label):
    """Run a single benchmark and return timing."""
    
    u = jnp.zeros((params['ny'], params['nx']))
    v = jnp.zeros((params['ny'], params['nx']))
    p = jnp.zeros((params['ny'], params['nx']))
    
    mask = jnp.array(geom['mask'])
    inlet_y = geom['y'][geom['j_step']:] - geom['y'][geom['j_step']]
    u_inlet = jnp.array(get_inlet_profile(inlet_y, geom['dims']['h_inlet'], params['u_max']))
    
    # Warmup
    start_compile = time.time()
    solver_module.time_step_jax(
        u, v, p, params['dt'], geom['dx'], geom['dy'], 
        params['rho'], params['nu'], params['nit'],
        mask, u_inlet, geom['j_step'], geom['i_step']
    )
    u.block_until_ready()
    compile_time = time.time() - start_compile
    
    # Run benchmark
    start = time.time()
    u_final, v_final, p_final, cfl_history = solver_func(
        nt, u, v, p, params['dt'], geom['dx'], geom['dy'],
        params['rho'], params['nu'], params['nit'],
        mask, u_inlet, geom['j_step'], geom['i_step']
    )
    duration = time.time() - start
    
    return {
        'label': label,
        'nt': nt,
        'compile_time': compile_time,
        'run_time': duration,
        'steps_per_sec': nt / duration,
        'final_max_u': float(np.max(np.abs(u_final))),
        'final_cfl': cfl_history[-1] if cfl_history else None
    }


def main():
    parser = argparse.ArgumentParser(description="JAX CPU Optimization Benchmark")
    parser.add_argument("--nx", type=int, default=201)
    parser.add_argument("--ny", type=int, default=51)
    parser.add_argument("--nt", type=int, default=2000, help="Steps per benchmark run")
    parser.add_argument("--re", type=float, default=100)
    parser.add_argument("--runs", type=int, default=1, help="Number of runs per config")
    args = parser.parse_args()
    
    print("=" * 70)
    print("JAX CPU OPTIMIZATION BENCHMARK")
    print("=" * 70)
    print(f"Backend: {jax.default_backend()}")
    print(f"Grid: {args.nx} x {args.ny}")
    print(f"Steps: {args.nt}")
    print(f"Runs: {args.runs}")
    print()
    
    # Setup
    geom = setup_geometry(args.nx, args.ny)
    nu = calculate_nu_from_reynolds(args.re, 1.5, geom['dims']['h'])
    
    params = {
        'nx': args.nx,
        'ny': args.ny,
        'dt': 0.0002,
        'rho': 1.0,
        'nu': nu,
        'nit': 50,
        'u_max': 1.5
    }
    
    results = []
    
    # =========================================================================
    # Benchmark 1: Baseline (Python loop)
    # =========================================================================
    print("Running: Baseline (Python loop)...")
    import bfs_baseline
    
    def baseline_runner(nt, u, v, p, dt, dx, dy, rho, nu, nit, mask, u_inlet, j_step, i_step):
        from tqdm import tqdm
        cfl_history = []
        for _ in tqdm(range(nt), desc="Baseline", leave=False):
            u, v, p, cfl = bfs_baseline.time_step_jax(
                u, v, p, dt, dx, dy, rho, nu, nit, mask, u_inlet, j_step, i_step
            )
            cfl_history.append(float(cfl))
        u = u.block_until_ready()
        return np.array(u), np.array(v), np.array(p), cfl_history
    
    for run in range(args.runs):
        result = run_benchmark(bfs_baseline, baseline_runner, args.nt, geom, params, 
                                f"Baseline (run {run+1})")
        results.append(result)
        print(f"  Run {run+1}: {result['steps_per_sec']:.0f} steps/s")
    
    # =========================================================================
    # Benchmark 2: Optimized (lax.scan)
    # =========================================================================
    print("\nRunning: Optimized (lax.scan)...")
    import bfs_optimized
    
    for run in range(args.runs):
        result = run_benchmark(bfs_optimized, 
                                lambda *args: bfs_optimized.run_simulation_scan(*args, chunk_size=args[0]),
                                args.nt, geom, params, 
                                f"Optimized lax.scan (run {run+1})")
        results.append(result)
        print(f"  Run {run+1}: {result['steps_per_sec']:.0f} steps/s")
    
    # =========================================================================
    # Benchmark 3: Optimized (Python loop, donate_argnums only)
    # =========================================================================
    print("\nRunning: Optimized (Python loop, donate_argnums)...")
    
    for run in range(args.runs):
        result = run_benchmark(bfs_optimized, bfs_optimized.run_simulation_python_loop,
                                args.nt, geom, params, 
                                f"Optimized Python loop (run {run+1})")
        results.append(result)
        print(f"  Run {run+1}: {result['steps_per_sec']:.0f} steps/s")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Configuration':<35} {'Steps/s':>10} {'Time (s)':>10} {'Speedup':>10}")
    print("-" * 70)
    
    baseline_speed = results[0]['steps_per_sec']
    
    for r in results:
        speedup = r['steps_per_sec'] / baseline_speed
        print(f"{r['label']:<35} {r['steps_per_sec']:>10.0f} {r['run_time']:>10.2f} {speedup:>10.2f}x")
    
    print("\n✓ Benchmark complete!")


if __name__ == "__main__":
    main()
