"""
2D Flow Over a Backward-Facing Step - NumPy Implementation

Solves the 2D incompressible Navier-Stokes equations for flow over a
backward-facing step using finite differences (explicit Euler time stepping).

Based on the Armaly et al. (1983) benchmark geometry.

Usage:
    python backward_facing_step.py --re 100 --nt 5000
    python backward_facing_step.py --re 200 --verbose
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import os

from geometry import (
    setup_geometry, 
    get_inlet_profile, 
    apply_mask,
    calculate_nu_from_reynolds,
    H_STEP
)
from plots import plot_results


def build_up_b(b, rho, dt, u, v, dx, dy, mask):
    """
    Compute the RHS of the pressure Poisson equation.
    
    Only compute for fluid cells (mask == 1).
    Handle solid regions carefully to avoid numerical issues.
    """
    # Extract interior slices
    mask_inner = mask[1:-1, 1:-1]
    
    # Velocity gradients
    du_dx = (u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx)
    dv_dy = (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)
    du_dy = (u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy)
    dv_dx = (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx)
    
    # Compute b term only for fluid cells
    b_inner = (rho * (1 / dt * (du_dx + dv_dy) -
                      du_dx**2 -
                      2 * du_dy * dv_dx -
                      dv_dy**2))
    
    # Apply mask to zero out solid regions BEFORE assignment
    b[1:-1, 1:-1] = b_inner * mask_inner
    
    return b


def pressure_poisson(p, dx, dy, b, mask, nit=50):
    """
    Solve the pressure Poisson equation using Jacobi iteration.
    
    Boundary conditions:
    - Walls (top, bottom, step): dp/dn = 0 (Neumann)
    - Inlet (left): dp/dx = 0 (Neumann)  
    - Outlet (right): p = 0 (Dirichlet reference)
    """
    pn = np.empty_like(p)
    
    for _ in range(nit):
        pn = p.copy()
        
        # Update interior points
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 + 
                          (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                          (2 * (dx**2 + dy**2)) -
                          dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * 
                          b[1:-1, 1:-1])
        
        # Boundary Conditions
        # Top wall: dp/dy = 0
        p[-1, :] = p[-2, :]
        
        # Bottom wall (downstream of step): dp/dy = 0
        p[0, :] = p[1, :]
        
        # Inlet (left): dp/dx = 0
        p[:, 0] = p[:, 1]
        
        # Outlet (right): p = 0 (reference pressure)
        p[:, -1] = 0
        
        # Step surface: enforce Neumann conditions
        # The mask handles zeroing velocities; pressure naturally adjusts
        
    return p


def apply_boundary_conditions(u, v, geom, u_max):
    """
    Apply all boundary conditions for backward-facing step.
    
    Parameters
    ----------
    u, v : ndarray
        Velocity fields
    geom : dict
        Geometry dictionary from setup_geometry()
    u_max : float
        Maximum inlet velocity
    """
    mask = geom['mask']
    i_step = geom['i_step']
    j_step = geom['j_step']
    dims = geom['dims']
    ny, nx = u.shape
    
    # === INLET (left boundary, x = -L_in) ===
    # Parabolic profile in inlet channel (y from step top to channel top)
    inlet_y = geom['y'][j_step:] - geom['y'][j_step]
    u_inlet = get_inlet_profile(inlet_y, dims['h_inlet'], u_max)
    u[j_step:, 0] = u_inlet
    v[:, 0] = 0  # No vertical velocity at inlet
    
    # Below step at inlet (solid wall)
    u[:j_step, 0] = 0
    
    # === OUTLET (right boundary, x = L_out) ===
    # Zero gradient (convective outflow)
    u[:, -1] = u[:, -2]
    v[:, -1] = v[:, -2]
    
    # === TOP WALL (y = H) ===
    u[-1, :] = 0
    v[-1, :] = 0
    
    # === BOTTOM WALL (y = 0, for x > 0) ===
    u[0, i_step:] = 0
    v[0, i_step:] = 0
    
    # === STEP SURFACES ===
    # Enforce no-slip via mask (already done in main loop)
    # Here we ensure the step face (x=0, y<h) and top (y=h, x<0) are zero
    u, v = apply_mask(u, v, mask)
    
    return u, v


def backward_facing_step(nt, u, v, dt, dx, dy, p, rho, nu, geom, u_max, verbose=False):
    """
    Main time-stepping loop for backward-facing step flow.
    
    Uses a gradual ramp-up of inlet velocity to avoid impulsive start instabilities.
    """
    un = np.empty_like(u)
    vn = np.empty_like(v)
    ny, nx = u.shape
    b = np.zeros((ny, nx))
    mask = geom['mask']
    
    # Ramp-up parameters (gradually increase inlet velocity over first 500 steps)
    ramp_steps = min(500, nt // 4)
    
    # Monitoring
    cfl_history = []
    monitoring_data = []
    
    if verbose:
        print("\n" + "="*80)
        print("REAL-TIME STABILITY MONITORING")
        print("="*80)
        print(f"{'Step':<8} {'Time':<10} {'u_max':<8} {'v_max':<8} {'CFL':<8} {'D_num':<8} {'Status':<10}")
        print("-"*80)
        iter_range = range(nt)
    else:
        iter_range = tqdm(range(nt), desc="Time Stepping")
    
    for n in iter_range:
        # Gradual ramp-up of inlet velocity
        if n < ramp_steps:
            ramp_factor = (n + 1) / ramp_steps
            current_u_max = u_max * ramp_factor
        else:
            current_u_max = u_max
        un = u.copy()
        vn = v.copy()
        
        # 1. Build RHS for pressure Poisson
        b = build_up_b(b, rho, dt, u, v, dx, dy, mask)
        
        # 2. Solve pressure Poisson
        p = pressure_poisson(p, dx, dy, b, mask)
        
        # 3. Update velocities (momentum equations)
        # Only update fluid cells (where mask is 1)
        mask_inner = mask[1:-1, 1:-1]
        
        u_new = (un[1:-1, 1:-1] -
                 un[1:-1, 1:-1] * dt / dx *
                (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                 vn[1:-1, 1:-1] * dt / dy *
                (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                 dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                 nu * (dt / dx**2 *
                (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                 dt / dy**2 *
                (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))

        v_new = (vn[1:-1, 1:-1] -
                un[1:-1, 1:-1] * dt / dx *
               (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                vn[1:-1, 1:-1] * dt / dy *
               (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                nu * (dt / dx**2 *
               (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                dt / dy**2 *
               (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))
        
        # Apply mask during assignment to prevent updates in solid regions
        u[1:-1, 1:-1] = np.where(mask_inner == 1, u_new, 0.0)
        v[1:-1, 1:-1] = np.where(mask_inner == 1, v_new, 0.0)
        
        # 4. Apply boundary conditions (also enforces mask)
        u, v = apply_boundary_conditions(u, v, geom, current_u_max)
        
        # Calculate monitoring metrics
        u_max_val = np.max(np.abs(u))
        v_max_val = np.max(np.abs(v))
        cfl = dt * (u_max_val / dx + v_max_val / dy)
        diffusion_num = nu * dt / (min(dx, dy)**2)
        
        cfl_history.append(cfl)
        
        # Check for NaN
        if np.isnan(u_max_val) or np.isnan(v_max_val):
            if verbose:
                print(f"\n{'='*80}")
                print(f"SIMULATION CRASHED: NaN detected at step {n}")
                print(f"{'='*80}\n")
            else:
                print(f"\nSimulation crashed at step {n}: NaN detected")
            break
        
        # Verbose monitoring
        if verbose and (n % 100 == 0 or n == nt - 1):
            current_time = n * dt
            if cfl > 1.0 or diffusion_num > 0.25:
                status = "✗ UNSTABLE"
            elif cfl > 0.8 or diffusion_num > 0.2:
                status = "⚠ WARNING"
            else:
                status = "✓ STABLE"
            
            print(f"{n:<8} {current_time:<10.4f} {u_max_val:<8.4f} {v_max_val:<8.4f} {cfl:<8.4f} {diffusion_num:<8.4f} {status:<10}")
            
            monitoring_data.append({
                'step': n, 'time': current_time,
                'u_max': u_max_val, 'v_max': v_max_val,
                'cfl': cfl, 'diffusion_num': diffusion_num
            })
    
    if verbose:
        print("="*80 + "\n")
    
    return u, v, p, cfl_history, monitoring_data


def calculate_reattachment_length(u, geom):
    """
    Calculate reattachment length from wall shear stress.
    
    The reattachment point is where du/dy at the bottom wall changes sign
    from negative (recirculation) to positive (attached flow).
    
    Returns X_r/h (normalized by step height).
    """
    dy = geom['dy']
    x = geom['x']
    i_step = geom['i_step']
    h = geom['dims']['h']
    
    # Wall shear stress proportional to du/dy at y=0 (bottom wall)
    # We look at the second row (j=1) since j=0 is the wall
    tau_w = (u[1, i_step:] - u[0, i_step:]) / dy
    x_downstream = x[i_step:]
    
    # Find where tau_w changes sign (zero crossing)
    sign_change = np.where(np.diff(np.sign(tau_w)))[0]
    
    if len(sign_change) > 0:
        # First sign change is reattachment
        idx = sign_change[0]
        # Linear interpolation for more accurate location
        x1, x2 = x_downstream[idx], x_downstream[idx + 1]
        t1, t2 = tau_w[idx], tau_w[idx + 1]
        x_r = x1 - t1 * (x2 - x1) / (t2 - t1)
        return x_r / h
    else:
        return None  # No reattachment found


    # Generate plots
    print("\nGenerating plots...")
    plot_results(u, v, p, geom, output_dir, args.re, backend="numpy")
    
    print("\n✓ Simulation complete!")
