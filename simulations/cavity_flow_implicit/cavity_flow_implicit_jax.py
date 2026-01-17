#!/usr/bin/env python3
"""
2D Lid-Driven Cavity Flow - Implicit Scheme (JAX Accelerated)

Solves the incompressible Navier-Stokes equations using the Fractional Step
(Projection) method with Crank-Nicolson time integration for diffusion.

This version uses JAX for GPU/TPU acceleration.
"""

import jax
# Enable 64-bit precision only on backends that support it (not Metal)
import os as _os
if _os.environ.get('JAX_PLATFORMS', '') != 'METAL' and 'metal' not in str(jax.devices()).lower():
    try:
        jax.config.update("jax_enable_x64", True)
    except Exception:
        pass

import jax.numpy as jnp
from jax import jit, lax
from functools import partial
import numpy as np
import argparse
import time
import os
import sys

# Import JAX components
# Ensure current directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fractional_step_jax import (
    predictor_step_jax, pressure_poisson_jax, corrector_step_jax,
    apply_boundary_conditions_jax, compute_divergence_jax
)


def initialize_fields(nx, ny):
    """Initialize velocity and pressure fields."""
    u = jnp.zeros((ny, nx))
    v = jnp.zeros((ny, nx))
    p = jnp.zeros((ny, nx))
    return u, v, p


@partial(jit, static_argnums=(8, 9))
def simulation_chunk(u, v, p, dt, dx, dy, rho, nu, nit_pressure, nt_chunk):
    """
    Run a chunk of time steps using lax.scan for efficiency.
    """
    
    def step_fn(carry, _):
        u, v, p = carry
        
        # 1. Predictor: intermediate velocity with implicit diffusion
        u_star, v_star = predictor_step_jax(u, v, dt, dx, dy, nu)
        
        # 2. Pressure Poisson: enforce incompressibility
        p = pressure_poisson_jax(p, u_star, v_star, dt, dx, dy, rho, nit=nit_pressure)
        
        # 3. Corrector: project onto divergence-free space
        u, v = corrector_step_jax(u_star, v_star, p, dt, dx, dy, rho)
        
        # 4. Apply boundary conditions
        u, v = apply_boundary_conditions_jax(u, v, u_lid=1.0)
        
        return (u, v, p), None

    # Run the scan loop
    (u, v, p), _ = lax.scan(step_fn, (u, v, p), None, length=nt_chunk)
    
    return u, v, p


def cavity_flow_implicit_jax(nt, u, v, p, dt, dx, dy, rho, nu, verbose=False, nit_pressure=50):
    """
    Solve 2D cavity flow using implicit Fractional Step method (JAX).
    """
    cfl_history = []
    monitoring_data = []
    
    # Initial BCs
    u, v = apply_boundary_conditions_jax(u, v, u_lid=1.0)
    
    # Chunk size for monitoring frequency
    chunk_size = 50
    num_chunks = (nt + chunk_size - 1) // chunk_size
    
    if verbose:
        print("\n" + "="*80)
        print("REAL-TIME STABILITY MONITORING (IMPLICIT JAX)")
        print(f"Device: {jax.devices()[0]}")
        print("="*80)
        print(f"{'Step':<8} {'Time':<10} {'u_max':<8} {'v_max':<8} {'CFL':<8} {'Div':<10} {'Status':<10}")
        print("-" * 80)

    start_time = time.time()
    current_step = 0
    
    # Main Loop (Chunks)
    for chunk_idx in range(num_chunks):
        # Determine steps in this chunk
        steps_in_this_chunk = min(chunk_size, nt - current_step)
        
        # Run chunk
        u, v, p = simulation_chunk(u, v, p, dt, dx, dy, rho, nu, nit_pressure, steps_in_this_chunk)
        
        # Block until computation is done for timing/nan checks
        u.block_until_ready()
        
        current_step += steps_in_this_chunk
        
        # Metrics (computed on host or device, pull to host for printing)
        u_np = np.array(u)
        v_np = np.array(v)
        
        u_max = np.max(np.abs(u_np))
        v_max = np.max(np.abs(v_np))
        cfl = dt * (u_max / dx + v_max / dy)
        
        # Divergence
        div = compute_divergence_jax(u, v, dx, dy)
        max_div = np.max(np.abs(np.array(div)[1:-1, 1:-1]))
        
        cfl_history.extend([cfl] * steps_in_this_chunk) # Approx history for chunk
        
        # Check for NaNs
        if np.isnan(u_max) or np.isnan(v_max) or np.isnan(cfl):
            t_now = current_step * dt
            if verbose:
                print(f"\n{'='*80}")
                print(f"SIMULATION CRASHED at step {current_step} (t={t_now:.4f})")
                print(f"NaN detected")
                print(f"{'='*80}\n")
            return u_np, v_np, np.array(p), cfl_history, monitoring_data
            
        # Logging
        if verbose:
            t_now = current_step * dt
            
            if max_div > 0.1:
                status = "⚠ HIGH DIV"
            elif max_div > 0.01:
                status = "○ CONV"
            else:
                status = "✓ STABLE"
                
            print(f"{current_step:<8} {t_now:<10.4f} {u_max:<8.4f} {v_max:<8.4f} {cfl:<8.4f} {max_div:<10.2e} {status:<10}")
            
            monitoring_data.append({
                'step': current_step,
                'time': t_now,
                'u_max': float(u_max),
                'v_max': float(v_max),
                'cfl': float(cfl),
                'divergence': float(max_div)
            })

    if verbose:
        print("="*80 + "\n")
        
    return np.array(u), np.array(v), np.array(p), cfl_history, monitoring_data


def plot_results(u, v, p, nx, ny, L, output_file='cavity_flow_implicit_jax_result.png'):
    """Plot velocity field, pressure, and streamlines."""
    import matplotlib.pyplot as plt
    from matplotlib import cm
    
    x = np.linspace(0, L, nx)
    y = np.linspace(0, L, ny)
    X, Y = np.meshgrid(x, y)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pressure contours
    ax = axes[0]
    levels = np.linspace(p.min(), p.max(), 20)
    contourf = ax.contourf(X, Y, p, levels=levels, cmap=cm.viridis)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Pressure Field (Implicit JAX)')
    ax.set_aspect('equal')
    plt.colorbar(contourf, ax=ax)
    
    # Velocity magnitude and streamlines
    ax = axes[1]
    velocity_mag = np.sqrt(u**2 + v**2)
    contourf = ax.contourf(X, Y, velocity_mag, levels=20, cmap=cm.viridis)
    
    # Streamlines
    ax.streamplot(X, Y, u, v, color='white', linewidth=0.5, density=1.5, arrowsize=1.0)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Velocity Magnitude & Streamlines (Implicit JAX)')
    ax.set_aspect('equal')
    plt.colorbar(contourf, ax=ax, label='|V|')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Result saved to {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='2D Cavity Flow - Implicit Scheme (JAX)')
    parser.add_argument('--nx', type=int, default=41, help='Grid points in x')
    parser.add_argument('--nt', type=int, default=500, help='Number of time steps')
    parser.add_argument('--dt', type=float, default=0.01, help='Time step')
    parser.add_argument('--re', type=float, default=100, help='Reynolds number')
    parser.add_argument('--verbose', action='store_true', help='Enable monitoring')
    parser.add_argument('--output-dir', type=str, default='simulations/cavity_flow_implicit/outputs/jax_baseline',
                        help='Output directory')
    parser.add_argument('--nit-pressure', type=int, default=50, help='Pressure iterations')
    
    args = parser.parse_args()
    
    nx = args.nx
    ny = args.nx
    L = 2.0
    dx = L / (nx - 1)
    dy = L / (ny - 1)
    
    rho = 1.0
    nu = 1.0 / args.re
    dt = args.dt
    nt = args.nt
    
    os.makedirs(args.output_dir, exist_ok=True)
    plots_dir = os.path.join(os.path.dirname(args.output_dir), '../plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    print(f"2D Cavity Flow - Implicit Scheme (JAX accelerated)")
    print(f"Grid: {nx}x{ny}")
    print(f"Re: {args.re}, dt: {dt}")
    
    u, v, p = initialize_fields(nx, ny)
    
    # Warmup compile
    print("Compiling JAX functions...")
    warmup_u = jnp.zeros((ny, nx))
    warmup_v = jnp.zeros((ny, nx))
    warmup_p = jnp.zeros((ny, nx))
    _ = simulation_chunk(warmup_u, warmup_v, warmup_p, dt, dx, dy, rho, nu, args.nit_pressure, 2)
    print("Compilation complete.")
    
    start_t = time.time()
    
    u, v, p, cfl, data = cavity_flow_implicit_jax(
        nt, u, v, p, dt, dx, dy, rho, nu,
        verbose=args.verbose,
        nit_pressure=args.nit_pressure
    )
    
    duration = time.time() - start_t
    print(f"Simulation complete. Duration: {duration:.4f}s ({nt/duration:.2f} steps/s)")
    
    # Save
    np.savez(os.path.join(args.output_dir, 'solution_implicit_jax.npz'),
             u=u, v=v, p=p, nx=nx, ny=ny, L=L, Re=args.re, dt=dt, nt=nt)
    
    # Check for NaNs before plotting
    if not (np.isnan(u).any() or np.isnan(v).any()):
        plot_results(u, v, p, nx, ny, L, 
                    output_file=os.path.join(plots_dir, 'cavity_flow_implicit_jax_result.png'))
    else:
        print("Skipping plot due to NaNs")

if __name__ == "__main__":
    main()
