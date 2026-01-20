"""
2D Flow Over a Backward-Facing Step - JAX Implementation

JAX-accelerated solver for flow over a backward-facing step.
Supports CPU and Metal GPU backends.

Usage:
    python backward_facing_step_jax.py --re 100 --nt 10000
    JAX_PLATFORMS=cpu python backward_facing_step_jax.py --re 100
"""

import jax
import jax.numpy as jnp
from jax import jit, lax
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import argparse
import os

from geometry import (
    setup_geometry,
    get_inlet_profile,
    calculate_nu_from_reynolds,
    H_STEP
)
from plots import plot_results


# ============================================================================
# JIT-Compiled Core Functions
# ============================================================================

@jit
def build_up_b_jax(rho, dt, u, v, dx, dy, mask):
    """Compute RHS of pressure Poisson equation (JAX version)."""
    mask_inner = mask[1:-1, 1:-1]
    
    # Velocity gradients
    du_dx = (u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx)
    dv_dy = (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)
    du_dy = (u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy)
    dv_dx = (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx)
    
    # Compute b term
    b_inner = (rho * (1 / dt * (du_dx + dv_dy) -
                      du_dx**2 -
                      2 * du_dy * dv_dx -
                      dv_dy**2))
    
    # Apply mask
    b = jnp.zeros_like(u)
    b = b.at[1:-1, 1:-1].set(b_inner * mask_inner)
    
    return b


def pressure_poisson_step_jax(carry, _):
    """Single iteration of pressure Poisson solver for lax.scan."""
    p, dx, dy, b = carry
    pn = p
    
    # Update interior points
    new_p_inner = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 + 
                    (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                   (2 * (dx**2 + dy**2)) -
                   dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * 
                   b[1:-1, 1:-1])
    
    p = p.at[1:-1, 1:-1].set(new_p_inner)
    
    # Boundary conditions for backward-facing step
    p = p.at[-1, :].set(p[-2, :])  # Top: dp/dy = 0
    p = p.at[0, :].set(p[1, :])    # Bottom: dp/dy = 0
    p = p.at[:, 0].set(p[:, 1])    # Inlet: dp/dx = 0
    p = p.at[:, -1].set(0)         # Outlet: p = 0 (reference)
    
    return (p, dx, dy, b), None


def solve_pressure_jax(p, dx, dy, b, nit):
    """Solve pressure Poisson using lax.fori_loop."""
    def body_fn(i, p_val):
        pn = p_val
        
        new_p_inner = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 + 
                        (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                       (2 * (dx**2 + dy**2)) -
                       dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * 
                       b[1:-1, 1:-1])
        
        p_val = p_val.at[1:-1, 1:-1].set(new_p_inner)
        p_val = p_val.at[-1, :].set(p_val[-2, :])  # Top: dp/dy = 0
        p_val = p_val.at[0, :].set(p_val[1, :])    # Bottom: dp/dy = 0
        p_val = p_val.at[:, 0].set(p_val[:, 1])    # Inlet: dp/dx = 0
        p_val = p_val.at[:, -1].set(0)             # Outlet: p = 0
        return p_val
    
    return lax.fori_loop(0, nit, body_fn, p)


from functools import partial

@partial(jit, static_argnums=(8, 11, 12))  # nit, j_step, i_step are static
def time_step_jax(u, v, p, dt, dx, dy, rho, nu, nit, mask, u_inlet, j_step, i_step):
    """
    Single time step of the simulation (JIT-compiled).
    
    Parameters
    ----------
    u, v, p : arrays
        Velocity and pressure fields
    dt, dx, dy, rho, nu : floats
        Physical parameters
    nit : int
        Pressure iterations
    mask : array
        Solid/fluid mask
    u_inlet : array
        Inlet velocity profile
    j_step : int
        Y-index of step top
    i_step : int
        X-index of step end
    """
    un = u
    vn = v
    mask_inner = mask[1:-1, 1:-1]
    
    # 1. Build RHS for pressure Poisson
    b = build_up_b_jax(rho, dt, u, v, dx, dy, mask)
    
    # 2. Solve pressure Poisson
    p = solve_pressure_jax(p, dx, dy, b, nit)
    
    # Advection terms with conditional upwinding
    # u advection: u * du/dx + v * du/dy
    u_c = un[1:-1, 1:-1]
    v_c = vn[1:-1, 1:-1]
    
    # u * du/dx
    du_dx_b = (un[1:-1, 1:-1] - un[1:-1, 0:-2]) / dx  # Backward
    du_dx_f = (un[1:-1, 2:] - un[1:-1, 1:-1]) / dx    # Forward
    u_adv_x = jnp.where(u_c > 0, u_c * du_dx_b, u_c * du_dx_f)
    
    # v * du/dy
    du_dy_b = (un[1:-1, 1:-1] - un[0:-2, 1:-1]) / dy  # Backward
    du_dy_f = (un[2:, 1:-1] - un[1:-1, 1:-1]) / dy    # Forward
    u_adv_y = jnp.where(v_c > 0, v_c * du_dy_b, v_c * du_dy_f)
    
    # Diffusion terms
    u_diff = nu * (dt / dx**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                   dt / dy**2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1]))
    
    # Pressure term
    u_press = -dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2])
    
    u_new = u_c - dt * (u_adv_x + u_adv_y) + u_press + u_diff

    # v advection: u * dv/dx + v * dv/dy
    # u * dv/dx
    dv_dx_b = (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) / dx
    dv_dx_f = (vn[1:-1, 2:] - vn[1:-1, 1:-1]) / dx
    v_adv_x = jnp.where(u_c > 0, u_c * dv_dx_b, u_c * dv_dx_f)
    
    # v * dv/dy
    dv_dy_b = (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) / dy
    dv_dy_f = (vn[2:, 1:-1] - vn[1:-1, 1:-1]) / dy
    v_adv_y = jnp.where(v_c > 0, v_c * dv_dy_b, v_c * dv_dy_f)
    
    # Diffusion
    v_diff = nu * (dt / dx**2 * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                   dt / dy**2 * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1]))
                   
    # Pressure
    v_press = -dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1])
    
    v_new = v_c - dt * (v_adv_x + v_adv_y) + v_press + v_diff
    
    # Apply mask during assignment
    u = u.at[1:-1, 1:-1].set(jnp.where(mask_inner == 1, u_new, 0.0))
    v = v.at[1:-1, 1:-1].set(jnp.where(mask_inner == 1, v_new, 0.0))
    
    # 4. Boundary conditions
    # Inlet (left)
    u = u.at[j_step:, 0].set(u_inlet)
    v = v.at[:, 0].set(0)
    u = u.at[:j_step, 0].set(0)
    
    # Outlet (right) - zero gradient
    u = u.at[:, -1].set(u[:, -2])
    v = v.at[:, -1].set(v[:, -2])
    
    # Top wall
    u = u.at[-1, :].set(0)
    v = v.at[-1, :].set(0)
    
    # Bottom wall (downstream of step)
    u = u.at[0, i_step:].set(0)
    v = v.at[0, i_step:].set(0)
    
    # Enforce mask
    u = u * mask
    v = v * mask
    
    # Calculate CFL
    cfl = dt * (jnp.max(jnp.abs(u)) / dx + jnp.max(jnp.abs(v)) / dy)
    
    return u, v, p, cfl


# ============================================================================
# Main Simulation Loop
# ============================================================================

def run_simulation(nt, u, v, p, dt, dx, dy, rho, nu, nit, geom, u_max, verbose=False):
    """Run the backward-facing step simulation with JAX acceleration."""
    
    mask = jnp.array(geom['mask'])
    j_step = geom['j_step']
    i_step = geom['i_step']
    dims = geom['dims']
    
    # Pre-compute inlet profiles for ramp-up
    ramp_steps = min(500, nt // 4)
    inlet_y = geom['y'][j_step:] - geom['y'][j_step]
    
    cfl_history = []
    
    if verbose:
        print("\n" + "="*80)
        print("REAL-TIME STABILITY MONITORING (JAX)")
        print("="*80)
        print(f"{'Step':<8} {'Time':<10} {'u_max':<8} {'v_max':<8} {'CFL':<8} {'Status':<10}")
        print("-"*80)
        iter_range = range(nt)
    else:
        iter_range = tqdm(range(nt), desc="JAX Time Stepping")
    
    for n in iter_range:
        # Gradual ramp-up
        if n < ramp_steps:
            ramp_factor = (n + 1) / ramp_steps
            current_u_max = u_max * ramp_factor
        else:
            current_u_max = u_max
        
        # Compute inlet profile for current ramp
        u_inlet = jnp.array(get_inlet_profile(inlet_y, dims['h_inlet'], current_u_max))
        
        # Time step
        u, v, p, cfl = time_step_jax(u, v, p, dt, dx, dy, rho, nu, nit, 
                                      mask, u_inlet, j_step, i_step)
        
        cfl_val = float(cfl)
        cfl_history.append(cfl_val)
        
        # Monitoring
        if verbose and (n % 500 == 0 or n == nt - 1):
            u_sync = u.block_until_ready()
            v_sync = v.block_until_ready()
            u_max_val = float(jnp.max(jnp.abs(u_sync)))
            v_max_val = float(jnp.max(jnp.abs(v_sync)))
            
            import math
            if math.isnan(u_max_val) or math.isnan(cfl_val):
                print(f"\nSimulation crashed at step {n}: NaN detected")
                break
            
            status = "✓ STABLE" if cfl_val < 0.8 else "⚠ WARNING" if cfl_val < 1.0 else "✗ UNSTABLE"
            print(f"{n:<8} {n*dt:<10.4f} {u_max_val:<8.4f} {v_max_val:<8.4f} {cfl_val:<8.4f} {status:<10}")
    
    if verbose:
        print("="*80 + "\n")
    
    # Sync final results
    u = u.block_until_ready()
    v = v.block_until_ready()
    p = p.block_until_ready()
    
    return np.array(u), np.array(v), np.array(p), cfl_history


def calculate_reattachment_length(u, geom):
    """Calculate reattachment length from wall shear stress."""
    dy = geom['dy']
    x = geom['x']
    i_step = geom['i_step']
    h = geom['dims']['h']
    
    tau_w = (u[1, i_step:] - u[0, i_step:]) / dy
    x_downstream = x[i_step:]
    
    sign_change = np.where(np.diff(np.sign(tau_w)))[0]
    
    if len(sign_change) > 0:
        idx = sign_change[0]
        x1, x2 = x_downstream[idx], x_downstream[idx + 1]
        t1, t2 = tau_w[idx], tau_w[idx + 1]
        x_r = x1 - t1 * (x2 - x1) / (t2 - t1)
        return x_r / h
    return None


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backward-Facing Step Flow (JAX)")
    parser.add_argument("--nx", type=int, default=201, help="Grid points in x")
    parser.add_argument("--ny", type=int, default=51, help="Grid points in y")
    parser.add_argument("--nt", type=int, default=10000, help="Number of time steps")
    parser.add_argument("--re", type=float, default=100, help="Reynolds number")
    parser.add_argument("--dt", type=float, default=0.0002, help="Time step")
    parser.add_argument("--u-max", type=float, default=1.5, help="Max inlet velocity")
    parser.add_argument("--verbose", action="store_true", help="Real-time monitoring")
    parser.add_argument("--output-dir", type=str, default="simulations/backward_facing_step",
                        help="Output directory")
    args = parser.parse_args()
    
    backend = jax.default_backend()
    
    print("=" * 60)
    print("2D BACKWARD-FACING STEP FLOW (JAX)")
    print("=" * 60)
    print(f"JAX Backend: {backend}")
    
    # Setup geometry
    geom = setup_geometry(args.nx, args.ny)
    dims = geom['dims']
    dx, dy = geom['dx'], geom['dy']
    
    print(f"\nGeometry: {dims['L_total']:.2f} x {dims['H']:.2f}")
    print(f"Grid: {args.nx} x {args.ny}")
    
    # Calculate nu
    nu = calculate_nu_from_reynolds(args.re, args.u_max, dims['h'])
    rho = 1.0
    nit = 50
    
    print(f"\nRe = {args.re}, nu = {nu:.6f}, dt = {args.dt}")
    
    # Initialize (JAX arrays)
    u = jnp.zeros((args.ny, args.nx))
    v = jnp.zeros((args.ny, args.nx))
    p = jnp.zeros((args.ny, args.nx))
    
    # Warmup / compile
    print("\nCompiling JIT functions...")
    start_compile = time.time()
    mask = jnp.array(geom['mask'])
    inlet_y = geom['y'][geom['j_step']:] - geom['y'][geom['j_step']]
    u_inlet = jnp.array(get_inlet_profile(inlet_y, dims['h_inlet'], args.u_max))
    
    _, _, _, _ = time_step_jax(u, v, p, args.dt, dx, dy, rho, nu, nit,
                                mask, u_inlet, geom['j_step'], geom['i_step'])
    u.block_until_ready()
    print(f"Compilation: {time.time() - start_compile:.2f}s")
    
    # Run simulation
    print("\nStarting simulation...")
    start_time = time.time()
    
    u_final, v_final, p_final, cfl_history = run_simulation(
        args.nt, u, v, p, args.dt, dx, dy, rho, nu, nit, geom, args.u_max,
        verbose=args.verbose
    )
    
    duration = time.time() - start_time
    print(f"\nSimulation complete: {duration:.2f}s ({args.nt/duration:.0f} steps/s)")
    
    # Reattachment length
    x_r_h = calculate_reattachment_length(u_final, geom)
    if x_r_h is not None:
        print(f"Reattachment length: X_r/h = {x_r_h:.2f}")
    
    # Save results
    os.makedirs(os.path.join(args.output_dir, 'outputs'), exist_ok=True)
    solution_path = os.path.join(args.output_dir, 'outputs', f'solution_jax_{backend.lower()}.npz')
    np.savez(solution_path, u=u_final, v=v_final, p=p_final,
             x=geom['x'], y=geom['y'], mask=geom['mask'], Re=args.re)
    print(f"Solution saved to {solution_path}")
    
    # Plot
    print("\nGenerating plots...")
    plot_results(u_final, v_final, p_final, geom, args.output_dir, args.re, backend)
    
    print("\n✓ Simulation complete!")
