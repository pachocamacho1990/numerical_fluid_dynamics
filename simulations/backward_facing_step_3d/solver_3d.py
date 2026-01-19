"""
3D Implicit Flow Over a Backward-Facing Step - JAX Implementation

Extends the 2D solver to 3D with sidewall effects.

Method:
1. Predictor: Crank-Nicolson for diffusion (3D ADI), Explicit for advection.
2. Pressure Poisson: 3D iterative solver with Neumann BCs at step walls.
3. Corrector: Projection step.
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import argparse
import os
import sys

from functools import partial

# Add path for geometry
sys.path.append(os.path.dirname(__file__))
from geometry_3d import (
    setup_geometry_3d,
    get_inlet_profile_3d,
    calculate_nu_from_reynolds
)

# Reuse Thomas solver from 2D implementation
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backward_facing_step_implicit"))
from thomas_solver_jax import thomas_algorithm_jax

# ============================================================================
# 3D Masked ADI Solver Components
# ============================================================================

@partial(jit, static_argnums=(0,))
def build_tridiagonal_masked_jax(n, r, mask_1d):
    """Build tridiagonal coefficients for Masked ADI (reused from 2D)."""
    a = jnp.full(n, -r)
    b = jnp.full(n, 1.0 + 2.0 * r)
    c = jnp.full(n, -r)
    
    # Boundary conditions
    b = b.at[0].set(1.0); c = c.at[0].set(0.0)
    a = a.at[-1].set(0.0); b = b.at[-1].set(1.0)
    
    # Mask: solid => a=0, b=1, c=0
    is_solid = (mask_1d == 0)
    a = jnp.where(is_solid, 0.0, a)
    b = jnp.where(is_solid, 1.0, b)
    c = jnp.where(is_solid, 0.0, c)
    
    return a, b, c


def adi_step_3d(u, rhs, alpha, dx, dy, dz, mask):
    """
    Perform one 3D ADI step (X-sweep, Y-sweep, Z-sweep).
    
    Solves: (I - alpha*Lap) u = rhs
    """
    nz, ny, nx = u.shape
    rx = alpha / dx**2
    ry = alpha / dy**2
    rz = alpha / dz**2
    
    # ========================================
    # STEP 1: X-SWEEP (solve in x-direction)
    # ========================================
    # For each (k, j) plane, solve tridiagonal system in x
    
    # Compute explicit Y and Z diffusion contributions
    d2u_dy2 = jnp.zeros_like(u)
    d2u_dy2 = d2u_dy2.at[:, 1:-1, :].set(
        (u[:, 2:, :] - 2*u[:, 1:-1, :] + u[:, :-2, :]) / dy**2
    )
    
    d2u_dz2 = jnp.zeros_like(u)
    d2u_dz2 = d2u_dz2.at[1:-1, :, :].set(
        (u[2:, :, :] - 2*u[1:-1, :, :] + u[:-2, :, :]) / dz**2
    )
    
    rhs_x = rhs + alpha * (d2u_dy2 + d2u_dz2)
    rhs_x = jnp.where(mask == 0, 0.0, rhs_x)
    
    def solve_x_line(k, j, rhs_row, mask_row):
        a, b_coef, c = build_tridiagonal_masked_jax(nx, rx, mask_row)
        return thomas_algorithm_jax(a, b_coef, c, rhs_row)
    
    # Vectorize over k and j
    def solve_x_plane(k, rhs_plane, mask_plane):
        return vmap(lambda j: solve_x_line(k, j, rhs_plane[j, :], mask_plane[j, :]))(jnp.arange(ny))
    
    u_star = vmap(lambda k: solve_x_plane(k, rhs_x[k], mask[k]))(jnp.arange(nz))
    u_star = u_star * mask
    
    # ========================================
    # STEP 2: Y-SWEEP (solve in y-direction)
    # ========================================
    rhs_y = u_star - alpha * d2u_dy2  # Remove explicit Y part
    rhs_y = jnp.where(mask == 0, 0.0, rhs_y)
    
    def solve_y_line(k, i, rhs_col, mask_col):
        a, b_coef, c = build_tridiagonal_masked_jax(ny, ry, mask_col)
        return thomas_algorithm_jax(a, b_coef, c, rhs_col)
    
    def solve_y_plane(k, rhs_plane, mask_plane):
        # Transpose to iterate over i (columns)
        return vmap(lambda i: solve_y_line(k, i, rhs_plane[:, i], mask_plane[:, i]))(jnp.arange(nx)).T
    
    u_star2 = vmap(lambda k: solve_y_plane(k, rhs_y[k], mask[k]))(jnp.arange(nz))
    u_star2 = u_star2 * mask
    
    # ========================================
    # STEP 3: Z-SWEEP (solve in z-direction)
    # ========================================
    rhs_z = u_star2 - alpha * d2u_dz2  # Remove explicit Z part
    rhs_z = jnp.where(mask == 0, 0.0, rhs_z)
    
    def solve_z_line(rhs_line, mask_line):
        a, b_coef, c = build_tridiagonal_masked_jax(nz, rz, mask_line)
        return thomas_algorithm_jax(a, b_coef, c, rhs_line)
    
    # For Z-sweep, we need to solve along z for each (j, i) pair
    # rhs_z shape: (nz, ny, nx), mask shape: (nz, ny, nx)
    # We want to solve nz-length systems for each (j, i)
    # Reshape to (ny*nx, nz), solve, reshape back
    
    rhs_z_flat = rhs_z.transpose(1, 2, 0).reshape(-1, nz)  # (ny*nx, nz)
    mask_z_flat = mask.transpose(1, 2, 0).reshape(-1, nz)  # (ny*nx, nz)
    
    u_new_flat = vmap(solve_z_line)(rhs_z_flat, mask_z_flat)  # (ny*nx, nz)
    u_new = u_new_flat.reshape(ny, nx, nz).transpose(2, 0, 1)  # (nz, ny, nx)
    
    u_new = u_new * mask
    
    return u_new


# ============================================================================
# 3D Pressure Solver
# ============================================================================

@partial(jit, static_argnums=(5, 7, 8))
def pressure_poisson_3d(p, b, dx, dy, dz, nit, mask, i_step, j_step):
    """
    Solve 3D Pressure Poisson equation with Jacobi iteration.
    """
    
    def body_fn(p_old, _):
        # Standard 3D Jacobi update
        # All neighbor terms must use interior slices [1:-1] for all 3 dimensions
        p_new = (
            ((p_old[1:-1, 1:-1, 2:] + p_old[1:-1, 1:-1, :-2]) * dy**2 * dz**2 +
             (p_old[1:-1, 2:, 1:-1] + p_old[1:-1, :-2, 1:-1]) * dx**2 * dz**2 +
             (p_old[2:, 1:-1, 1:-1] + p_old[:-2, 1:-1, 1:-1]) * dx**2 * dy**2 -
             b[1:-1, 1:-1, 1:-1] * dx**2 * dy**2 * dz**2) /
            (2 * (dy**2 * dz**2 + dx**2 * dz**2 + dx**2 * dy**2))
        )
        
        p_full = p_old.at[1:-1, 1:-1, 1:-1].set(p_new)
        
        # --- Global BCs ---
        # Inlet: dp/dx = 0
        p_full = p_full.at[:, j_step:, 0].set(p_full[:, j_step:, 1])
        
        # Outlet: p = 0
        p_full = p_full.at[:, :, -1].set(0.0)
        
        # Top/Bottom walls: dp/dy = 0
        p_full = p_full.at[:, -1, :].set(p_full[:, -2, :])
        p_full = p_full.at[:, 0, i_step:].set(p_full[:, 1, i_step:])
        
        # Sidewalls: dp/dz = 0
        p_full = p_full.at[0, :, :].set(p_full[1, :, :])
        p_full = p_full.at[-1, :, :].set(p_full[-2, :, :])
        
        # Step BCs
        p_full = p_full.at[:, j_step-1, :i_step].set(p_full[:, j_step, :i_step])
        p_full = p_full.at[:, :j_step, i_step-1].set(p_full[:, :j_step, i_step])
        
        return p_full, None
    
    p_final, _ = lax.scan(body_fn, p, None, length=nit)
    return p_final


# ============================================================================
# 3D Advection
# ============================================================================

@jit
def compute_advection_3d(u, v, w, dx, dy, dz):
    """First-order upwind advection for 3D."""
    u_c = u[1:-1, 1:-1, 1:-1]
    v_c = v[1:-1, 1:-1, 1:-1]
    w_c = w[1:-1, 1:-1, 1:-1]
    
    # u advection
    du_dx_b = (u[1:-1, 1:-1, 1:-1] - u[1:-1, 1:-1, :-2]) / dx
    du_dx_f = (u[1:-1, 1:-1, 2:] - u[1:-1, 1:-1, 1:-1]) / dx
    du_dy_b = (u[1:-1, 1:-1, 1:-1] - u[1:-1, :-2, 1:-1]) / dy
    du_dy_f = (u[1:-1, 2:, 1:-1] - u[1:-1, 1:-1, 1:-1]) / dy
    du_dz_b = (u[1:-1, 1:-1, 1:-1] - u[:-2, 1:-1, 1:-1]) / dz
    du_dz_f = (u[2:, 1:-1, 1:-1] - u[1:-1, 1:-1, 1:-1]) / dz
    
    adv_u = jnp.zeros_like(u)
    val_u = (jnp.where(u_c > 0, u_c * du_dx_b, u_c * du_dx_f) +
             jnp.where(v_c > 0, v_c * du_dy_b, v_c * du_dy_f) +
             jnp.where(w_c > 0, w_c * du_dz_b, w_c * du_dz_f))
    adv_u = adv_u.at[1:-1, 1:-1, 1:-1].set(val_u)
    
    # v advection
    dv_dx_b = (v[1:-1, 1:-1, 1:-1] - v[1:-1, 1:-1, :-2]) / dx
    dv_dx_f = (v[1:-1, 1:-1, 2:] - v[1:-1, 1:-1, 1:-1]) / dx
    dv_dy_b = (v[1:-1, 1:-1, 1:-1] - v[1:-1, :-2, 1:-1]) / dy
    dv_dy_f = (v[1:-1, 2:, 1:-1] - v[1:-1, 1:-1, 1:-1]) / dy
    dv_dz_b = (v[1:-1, 1:-1, 1:-1] - v[:-2, 1:-1, 1:-1]) / dz
    dv_dz_f = (v[2:, 1:-1, 1:-1] - v[1:-1, 1:-1, 1:-1]) / dz
    
    adv_v = jnp.zeros_like(v)
    val_v = (jnp.where(u_c > 0, u_c * dv_dx_b, u_c * dv_dx_f) +
             jnp.where(v_c > 0, v_c * dv_dy_b, v_c * dv_dy_f) +
             jnp.where(w_c > 0, w_c * dv_dz_b, w_c * dv_dz_f))
    adv_v = adv_v.at[1:-1, 1:-1, 1:-1].set(val_v)
    
    # w advection
    dw_dx_b = (w[1:-1, 1:-1, 1:-1] - w[1:-1, 1:-1, :-2]) / dx
    dw_dx_f = (w[1:-1, 1:-1, 2:] - w[1:-1, 1:-1, 1:-1]) / dx
    dw_dy_b = (w[1:-1, 1:-1, 1:-1] - w[1:-1, :-2, 1:-1]) / dy
    dw_dy_f = (w[1:-1, 2:, 1:-1] - w[1:-1, 1:-1, 1:-1]) / dy
    dw_dz_b = (w[1:-1, 1:-1, 1:-1] - w[:-2, 1:-1, 1:-1]) / dz
    dw_dz_f = (w[2:, 1:-1, 1:-1] - w[1:-1, 1:-1, 1:-1]) / dz
    
    adv_w = jnp.zeros_like(w)
    val_w = (jnp.where(u_c > 0, u_c * dw_dx_b, u_c * dw_dx_f) +
             jnp.where(v_c > 0, v_c * dw_dy_b, v_c * dw_dy_f) +
             jnp.where(w_c > 0, w_c * dw_dz_b, w_c * dw_dz_f))
    adv_w = adv_w.at[1:-1, 1:-1, 1:-1].set(val_w)
    
    return adv_u, adv_v, adv_w


# ============================================================================
# Core Time Step
# ============================================================================

@partial(jit, static_argnums=(10, 13))
def time_step_3d(u, v, w, p, dt, dx, dy, dz, rho, nu, nit, mask, u_inlet, geom_params):
    """Single 3D Implicit Fractional Step."""
    j_step = geom_params[0]
    i_step = geom_params[1]
    nz, ny, nx = u.shape
    
    # 1. Advection (Explicit)
    adv_u, adv_v, adv_w = compute_advection_3d(u, v, w, dx, dy, dz)
    
    # 2. Diffusion Predictor (Implicit Crank-Nicolson via 3D ADI)
    alpha = nu * dt / 2.0
    
    # Compute Laplacians for RHS
    def compute_laplacian_3d(field):
        lap = jnp.zeros_like(field)
        lap = lap.at[1:-1, 1:-1, 1:-1].set(
            (field[1:-1, 1:-1, 2:] - 2*field[1:-1, 1:-1, 1:-1] + field[1:-1, 1:-1, :-2])/dx**2 +
            (field[1:-1, 2:, 1:-1] - 2*field[1:-1, 1:-1, 1:-1] + field[1:-1, :-2, 1:-1])/dy**2 +
            (field[2:, 1:-1, 1:-1] - 2*field[1:-1, 1:-1, 1:-1] + field[:-2, 1:-1, 1:-1])/dz**2
        )
        return lap
    
    lap_u = compute_laplacian_3d(u)
    lap_v = compute_laplacian_3d(v)
    lap_w = compute_laplacian_3d(w)
    
    rhs_u = u + alpha * lap_u - dt * adv_u
    rhs_v = v + alpha * lap_v - dt * adv_v
    rhs_w = w + alpha * lap_w - dt * adv_w
    
    # Solve via 3D ADI
    u_star = adi_step_3d(u, rhs_u, alpha, dx, dy, dz, mask)
    v_star = adi_step_3d(v, rhs_v, alpha, dx, dy, dz, mask)
    w_star = adi_step_3d(w, rhs_w, alpha, dx, dy, dz, mask)
    
    # Apply inlet BCs (u_inlet shape is (nz, ny_inlet), need to match [:, j_step:, 0] slice)
    u_star = u_star.at[:, j_step:, 0].set(u_inlet)
    v_star = v_star.at[:, j_step:, 0].set(0.0)
    w_star = w_star.at[:, j_step:, 0].set(0.0)
    
    # 3. Pressure Solve
    div = jnp.zeros_like(p)
    div = div.at[1:-1, 1:-1, 1:-1].set(
        (u_star[1:-1, 1:-1, 2:] - u_star[1:-1, 1:-1, :-2]) / (2 * dx) +
        (v_star[1:-1, 2:, 1:-1] - v_star[1:-1, :-2, 1:-1]) / (2 * dy) +
        (w_star[2:, 1:-1, 1:-1] - w_star[:-2, 1:-1, 1:-1]) / (2 * dz)
    )
    b_term = (rho / dt) * div
    
    p_new = pressure_poisson_3d(p, b_term, dx, dy, dz, nit, mask, i_step, j_step)
    
    # 4. Corrector
    dp_dx = (p_new[1:-1, 1:-1, 2:] - p_new[1:-1, 1:-1, :-2]) / (2 * dx)
    dp_dy = (p_new[1:-1, 2:, 1:-1] - p_new[1:-1, :-2, 1:-1]) / (2 * dy)
    dp_dz = (p_new[2:, 1:-1, 1:-1] - p_new[:-2, 1:-1, 1:-1]) / (2 * dz)
    
    u_new = u_star.at[1:-1, 1:-1, 1:-1].add(-(dt/rho) * dp_dx)
    v_new = v_star.at[1:-1, 1:-1, 1:-1].add(-(dt/rho) * dp_dy)
    w_new = w_star.at[1:-1, 1:-1, 1:-1].add(-(dt/rho) * dp_dz)
    
    # Enforce Mask
    u_new = u_new * mask
    v_new = v_new * mask
    w_new = w_new * mask
    
    # Inlet
    u_new = u_new.at[:, j_step:, 0].set(u_inlet)
    v_new = v_new.at[:, j_step:, 0].set(0.0)
    w_new = w_new.at[:, j_step:, 0].set(0.0)
    
    # Top/Bottom walls
    u_new = u_new.at[:, -1, :].set(0.0); v_new = v_new.at[:, -1, :].set(0.0); w_new = w_new.at[:, -1, :].set(0.0)
    u_new = u_new.at[:, 0, :].set(0.0); v_new = v_new.at[:, 0, :].set(0.0); w_new = w_new.at[:, 0, :].set(0.0)
    
    # Sidewalls (z=0, z=W)
    u_new = u_new.at[0, :, :].set(0.0); v_new = v_new.at[0, :, :].set(0.0); w_new = w_new.at[0, :, :].set(0.0)
    u_new = u_new.at[-1, :, :].set(0.0); v_new = v_new.at[-1, :, :].set(0.0); w_new = w_new.at[-1, :, :].set(0.0)
    
    # Outlet (Neumann)
    u_new = u_new.at[:, :, -1].set(u_new[:, :, -2])
    v_new = v_new.at[:, :, -1].set(v_new[:, :, -2])
    w_new = w_new.at[:, :, -1].set(w_new[:, :, -2])
    
    # CFL
    cfl = dt * (jnp.max(jnp.abs(u_new))/dx + jnp.max(jnp.abs(v_new))/dy + jnp.max(jnp.abs(w_new))/dz)
    
    return u_new, v_new, w_new, p_new, cfl


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="3D Implicit BFS Solver (JAX)")
    parser.add_argument("--re", type=float, default=100.0)
    parser.add_argument("--nx", type=int, default=101)
    parser.add_argument("--ny", type=int, default=26)
    parser.add_argument("--nz", type=int, default=26)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--nt", type=int, default=1000)
    parser.add_argument("--nit", type=int, default=50)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"3D Implicit Backward-Facing Step (Re={args.re})")
    print("=" * 60)
    
    # Geometry
    geom = setup_geometry_3d(args.nx, args.ny, args.nz)
    mask = jnp.array(geom['mask'])
    dx, dy, dz = geom['dx'], geom['dy'], geom['dz']
    j_step, i_step = geom['j_step'], geom['i_step']
    
    # Physics
    u_max = 1.5
    h_step = geom['dims']['h']
    W = geom['dims']['W']
    nu = calculate_nu_from_reynolds(args.re, u_max, h_step)
    rho = 1.0
    
    print(f"Grid: {args.nx}x{args.ny}x{args.nz} ({args.nx * args.ny * args.nz:,} cells)")
    print(f"Nu: {nu:.6f}")
    
    # Init Fields
    u = jnp.zeros((args.nz, args.ny, args.nx))
    v = jnp.zeros((args.nz, args.ny, args.nx))
    w = jnp.zeros((args.nz, args.ny, args.nx))
    p = jnp.zeros((args.nz, args.ny, args.nx))
    
    # Inlet Profile (3D)
    inlet_y = geom['inlet_y']
    u_inlet_2d = jnp.array(get_inlet_profile_3d(inlet_y, geom['z'], geom['dims']['h_inlet'], W, u_max))
    
    # Compilation
    print("Compiling JAX functions...")
    geom_params = (j_step, i_step)
    _ = time_step_3d(u, v, w, p, args.dt, dx, dy, dz, rho, nu, args.nit, mask, u_inlet_2d, geom_params)
    print("Compilation complete.")
    
    # Loop
    start = time.time()
    for n in tqdm(range(args.nt), disable=args.verbose):
        u, v, w, p, cfl = time_step_3d(u, v, w, p, args.dt, dx, dy, dz, rho, nu, args.nit, mask, u_inlet_2d, geom_params)
        
        if args.verbose and n % 100 == 0:
            cfl_val = float(cfl)
            print(f"Step {n}: CFL={cfl_val:.4f}, Max U={float(jnp.max(u)):.4f}")
            if jnp.isnan(cfl):
                print("NaN detected!")
                break
    
    duration = time.time() - start
    print(f"\nCompleted in {duration:.2f}s ({args.nt/duration:.1f} iter/s)")
    
    # Save
    os.makedirs("simulations/backward_facing_step_3d/outputs", exist_ok=True)
    np.savez("simulations/backward_facing_step_3d/outputs/solution_3d.npz",
             u=u, v=v, w=w, p=p,
             x=geom['x'], y=geom['y'], z=geom['z'],
             mask=geom['mask'])
    print("Saved solution.")


if __name__ == "__main__":
    main()
