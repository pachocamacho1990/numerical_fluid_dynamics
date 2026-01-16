#!/usr/bin/env python3
"""
2D Lid-Driven Cavity Flow - Implicit Scheme

Solves the incompressible Navier-Stokes equations using the Fractional Step
(Projection) method with Crank-Nicolson time integration for diffusion.

This implementation is unconditionally stable for diffusion, allowing much
larger time steps than the explicit scheme, especially at high Reynolds numbers.

Based on the Chorin-Temam projection method (1967-1968).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import argparse
import time
import os

from fractional_step import (
    predictor_step, pressure_poisson, corrector_step,
    apply_boundary_conditions, compute_divergence
)


def initialize_fields(nx, ny):
    """Initialize velocity and pressure fields."""
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.zeros((ny, nx))
    return u, v, p


def cavity_flow_implicit(nt, u, v, p, dt, dx, dy, rho, nu, verbose=False, nit_pressure=50):
    """
    Solve 2D cavity flow using implicit Fractional Step method.
    
    Args:
        nt: Number of time steps
        u, v: Initial velocity fields (ny, nx)
        p: Initial pressure field (ny, nx)
        dt: Time step
        dx, dy: Grid spacing
        rho: Density
        nu: Kinematic viscosity
        verbose: Print monitoring info
        nit_pressure: Pressure Poisson iterations
    
    Returns:
        u, v, p: Final fields
        cfl_history: CFL number at each step
        monitoring_data: Detailed monitoring (if verbose)
    """
    from tqdm import tqdm
    
    cfl_history = []
    monitoring_data = []
    
    # Apply initial boundary conditions
    u, v = apply_boundary_conditions(u, v, u_lid=1.0)
    
    # Verbose monitoring setup
    if verbose:
        print("\n" + "="*80)
        print("REAL-TIME STABILITY MONITORING (IMPLICIT)")
        print("="*80)
        print(f"{'Step':<8} {'Time':<10} {'u_max':<8} {'v_max':<8} {'CFL':<8} {'Div':<10} {'Status':<10}")
        print("-" * 80)
        iter_range = range(nt)
    else:
        iter_range = tqdm(range(nt), desc="Time Stepping (Implicit)")
    
    for n in iter_range:
        # Fractional Step method
        # 1. Predictor: intermediate velocity with implicit diffusion
        u_star, v_star = predictor_step(u, v, dt, dx, dy, nu)
        
        # 2. Pressure Poisson: enforce incompressibility
        p = pressure_poisson(p, u_star, v_star, dt, dx, dy, rho, nit=nit_pressure)
        
        # 3. Corrector: project onto divergence-free space
        u, v = corrector_step(u_star, v_star, p, dt, dx, dy, rho)
        
        # 4. Apply boundary conditions
        u, v = apply_boundary_conditions(u, v, u_lid=1.0)
        
        # Calculate monitoring metrics
        u_max = np.max(np.abs(u))
        v_max = np.max(np.abs(v))
        cfl = dt * (u_max / dx + v_max / dy)
        
        # Compute divergence
        div = compute_divergence(u, v, dx, dy)
        max_div = np.max(np.abs(div[1:-1, 1:-1]))
        
        cfl_history.append(cfl)
        
        # Verbose monitoring
        if verbose and (n % 50 == 0 or n == nt - 1):
            current_time = n * dt
            
            # Status based on divergence (implicit is always diffusion-stable)
            if max_div > 0.1:
                status = "⚠ HIGH DIV"
            elif max_div > 0.01:
                status = "○ CONV"
            else:
                status = "✓ STABLE"
            
            print(f"{n:<8} {current_time:<10.4f} {u_max:<8.4f} {v_max:<8.4f} {cfl:<8.4f} {max_div:<10.2e} {status:<10}")
            
            # Store monitoring data
            monitoring_data.append({
                'step': n,
                'time': current_time,
                'u_max': u_max,
                'v_max': v_max,
                'cfl': cfl,
                'divergence': max_div,
                'converged': max_div < 0.01
            })
    
    if verbose:
        print("="*80 + "\n")
    
    return u, v, p, cfl_history, monitoring_data if verbose else None


def plot_results(u, v, p, nx, ny, L, output_file='cavity_flow_implicit_result.png'):
    """Plot velocity field, pressure, and streamlines."""
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
    ax.set_title('Pressure Field (Implicit)')
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
    ax.set_title('Velocity Magnitude & Streamlines (Implicit)')
    ax.set_aspect('equal')
    plt.colorbar(contourf, ax=ax, label='|V|')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Result saved to {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='2D Cavity Flow - Implicit Scheme')
    parser.add_argument('--nx', type=int, default=41, help='Grid points in x')
    parser.add_argument('--nt', type=int, default=500, help='Number of time steps')
    parser.add_argument('--dt', type=float, default=0.01, help='Time step (can be large!)')
    parser.add_argument('--re', type=float, default=100, help='Reynolds number')
    parser.add_argument('--verbose', action='store_true', help='Enable real-time monitoring')
    parser.add_argument('--benchmark', action='store_true', help='Benchmark mode (no output)')
    parser.add_argument('--output-dir', type=str, default='simulations/cavity_flow_implicit/outputs/baseline',
                        help='Output directory for data files')
    parser.add_argument('--nit-pressure', type=int, default=50, help='Pressure Poisson iterations')
    
    args = parser.parse_args()
    
    # Grid parameters
    nx = args.nx
    ny = args.nx  # Square domain
    L = 2.0
    dx = L / (nx - 1)
    dy = L / (ny - 1)
    
    # Physical parameters
    rho = 1.0
    nu = 1.0 / args.re  # Kinematic viscosity
    dt = args.dt
    nt = args.nt
    
    # Create output directories
    output_dir = args.output_dir
    plots_dir = 'simulations/cavity_flow_implicit/plots'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    if not args.benchmark:
        print(f"2D Cavity Flow - Implicit Scheme")
        print(f"Grid: {nx}x{ny}")
        print(f"Reynolds number: {args.re}")
        print(f"Time step: {dt}")
        print(f"Total steps: {nt}")
        print(f"Kinematic viscosity: {nu:.6f}")
        
        # Stability info
        dt_max_diffusion = dx**2 / (4 * nu)
        print(f"\nStability Analysis:")
        print(f"  Explicit diffusion limit: dt < {dt_max_diffusion:.6f}")
        print(f"  Your dt: {dt:.6f}")
        if dt > dt_max_diffusion:
            print(f"  ✓ Implicit allows {dt/dt_max_diffusion:.1f}x larger timestep!")
        print()
    
    # Initialize fields
    u, v, p = initialize_fields(nx, ny)
    
    # Run simulation
    start_time = time.time()
    u, v, p, cfl_history, monitoring_data = cavity_flow_implicit(
        nt, u, v, p, dt, dx, dy, rho, nu,
        verbose=args.verbose,
        nit_pressure=args.nit_pressure
    )
    end_time = time.time()
    
    duration = end_time - start_time
    
    if not args.benchmark:
        print(f"Simulation complete. Duration: {duration:.4f}s ({nt/duration:.2f} steps/s)")
    
    # Save results
    np.savez(os.path.join(output_dir, 'solution_implicit.npz'),
             u=u, v=v, p=p, nx=nx, ny=ny, L=L, Re=args.re, dt=dt, nt=nt)
    
    # Save CFL history
    np.savetxt(os.path.join(output_dir, 'cfl_implicit.csv'),
               np.column_stack([np.arange(nt), cfl_history]),
               delimiter=',', header='step,cfl', comments='')
    
    # Save monitoring data if verbose
    if args.verbose and monitoring_data:
        import csv
        with open(os.path.join(output_dir, 'monitoring_implicit.csv'), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=monitoring_data[0].keys())
            writer.writeheader()
            writer.writerows(monitoring_data)
        print(f"Monitoring data saved to {os.path.join(output_dir, 'monitoring_implicit.csv')}")
    
    if not args.benchmark:
        print(f"Solution saved to {os.path.join(output_dir, 'solution_implicit.npz')}")
        print(f"CFL history saved to {os.path.join(output_dir, 'cfl_implicit.csv')}")
        
        # Generate plot
        print("Generating plot...")
        plot_results(u, v, p, nx, ny, L,
                    output_file=os.path.join(plots_dir, 'cavity_flow_implicit_result.png'))
    
    if args.benchmark:
        # Print only timing for benchmarking
        print(f"{duration:.4f}")


if __name__ == "__main__":
    main()
