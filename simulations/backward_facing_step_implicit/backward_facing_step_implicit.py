"""
Implicit 2D Flow Over a Backward-Facing Step - JAX Implementation

Uses a Fractional Step Method with a Masked ADI (Alternating Direction Implicit) scheme
to handle the step geometry.

Method:
1. Predictor: Crank-Nicolson for diffusion (Masked ADI), Explicit for advection.
2. Pressure Poisson: Iterative solver with Neumann BCs at step walls.
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

# Add parent directory to path to import geometry
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backward_facing_step"))
from geometry import (
    setup_geometry,
    get_inlet_profile,
    calculate_nu_from_reynolds
)
try:
    from plots import plot_results
except ImportError:
    # If plots.py is not in the path or has issues, we define a dummy or copy it later
    pass

# Import local masked thomas solver
from thomas_solver_jax import thomas_algorithm_jax

# ============================================================================
# Masked ADI Solver Components
# ============================================================================

from functools import partial

@partial(jit, static_argnums=(0,))
def build_tridiagonal_masked_jax(n, r, mask_1d, bc_type='dirichlet'):
    """
    Build tridiagonal coefficients for Masked ADI.
    
    If mask_1d[i] == 0 (Solid):
        Eq: 1 * u[i] = 0  => a=0, b=1, c=0 (RHS handled in solver)
    If mask_1d[i] == 1 (Fluid):
        Standard diffusion eq: -r*u[i-1] + (1+2r)*u[i] - r*u[i+1] = rhs
        
    Args:
        n: Number of points
        r: Diffusion number
        mask_1d: 1D mask for the current row/col (1=fluid, 0=solid)
    """
    # Standard Fluid Coefficients
    a = jnp.full(n, -r)
    b = jnp.full(n, 1.0 + 2.0 * r)
    c = jnp.full(n, -r)
    
    # Boundary Conditions (Global Domain Boundaries)
    if bc_type == 'dirichlet':
        b = b.at[0].set(1.0); c = c.at[0].set(0.0)
        a = a.at[-1].set(0.0); b = b.at[-1].set(1.0) # Fixed value boundaries
    
    # Apply Mask logic
    # Where mask is 0 (solid): a=0, b=1, c=0
    # We use jnp.where to vectorise this
    
    is_solid = (mask_1d == 0)
    
    a = jnp.where(is_solid, 0.0, a)
    b = jnp.where(is_solid, 1.0, b)
    c = jnp.where(is_solid, 0.0, c)
    
    return a, b, c

@jit
def solve_masked_check(u_val, mask_val):
    """
    Helper to zero out solution in solid regions just in case.
    """
    return jnp.where(mask_val == 0, 0.0, u_val)

def adi_step_masked(u, rhs, alpha, dx, dy, mask):
    """
    Perform one ADI step (X-sweep then Y-sweep) with Masking.
    
    (I - alpha*Lap) u = rhs
    """
    ny, nx = u.shape
    rx = alpha / dx**2
    ry = alpha / dy**2
    
    # ========================================
    # X-SWEEP
    # ========================================
    def solve_x_row(j, u, rhs, mask):
        mask_row = mask[j, :]
        rhs_row = rhs[j, :]
        
        # Add explicit Y-diffusion term for interior fluid points
        # If j is boundary, handled by BCs/Thomas setup but explicit term needs care
        # For simplicity in ADI, usually:
        # Eq: (1 - ax dxx) u* = (1 + ay dyy) u_n + dt*adv ... (Wait, this is splitting)
        # Here we assume standard Douglas-Rachford or Peaceman-Rachford splitting
        # Implementation: Solve (1 - alpha dxx) u* = rhs + alpha dyy u_old
        
        # Calculate d2u/dy2 explicitly
        d2u_dy2 = (u[j+1, :] - 2*u[j, :] + u[j-1, :]) / dy**2
        
        # If we are at top/bottom global boundary, d2u_dy2 is different or 0
        # lax.cond to handle j=0, j=ny-1 safely
        d2u_dy2 = jnp.where((j > 0) & (j < ny-1), d2u_dy2, 0.0)
        
        rhs_total = rhs_row + alpha * d2u_dy2
        
        # If solid, RHS should be 0 (u=0)
        rhs_total = jnp.where(mask_row == 0, 0.0, rhs_total)
        
        # Build masked tridiagonal
        a, b_coef, c = build_tridiagonal_masked_jax(nx, rx, mask_row)
        
        # Solve
        u_star_row = thomas_algorithm_jax(a, b_coef, c, rhs_total)
        
        return u_star_row

    # Vmap over all rows
    u_star = vmap(solve_x_row, in_axes=(0, None, None, None))(jnp.arange(ny), u, rhs, mask)
    
    # Force BCs/Mask explicitly if needed
    u_star = u_star * mask
    
    # ========================================
    # Y-SWEEP
    # ========================================
    def solve_y_col(i, u_star, u_old, rhs, mask):
        mask_col = mask[:, i]
        rhs_col = u_star[:, i] # For Y-sweep in ADI, rhs is typically u_star
        
        # Peaceman-Rachford:
        # Step 1 (X): (1 - A_x) u* = (1 + A_y) u_n  [Already done-ish, we passed generic RHS]
        # Step 2 (Y): (1 - A_y) u** = (1 + A_x) u* - (1 + A_y) u_n + ... 
        # Actually simplified ADI for Helmholtz (I - alpha Lap) u = f:
        # (1 - alpha dxx) u* = f + alpha dyy u_old
        # (1 - alpha dyy) u_new = u* - alpha dyy u_old ? No, that's approximate.
        
        # Standard Douglas-Rachford for Diffusion:
        # (I - A_x) u* = (I + A_y) u_n
        # (I - A_y) u_new = u* - A_y u_n
        # Here we are solving (I - alpha Lap) u_new = RHS
        # Split: (1 - alpha dxx)(1 - alpha dyy) u_new = RHS + alpha^2 dxx dyy u_new (error)
        # Approx: (1 - alpha dxx) u* = RHS
        #         (1 - alpha dyy) u_new = u*
        # But RHS already included (alpha dyy u_old) in X-sweep?
        # Let's use the implementation from adi_solver_jax.py which does:
        # X-sweep: Solve (1-alpha dxx) u* = RHS + alpha dyy u_old ( wait, checking previous code... )
        # Previous code `adi_solver_jax.py`:
        # X-sweep: rhs_x = rhs + alpha * d2u_dy2
        # Y-sweep: rhs_y = u_star - alpha * d2u_dy2 (This subtracts the explicit part added)
        # This solves (1 - alpha dxx - alpha dyy) u = rhs approximately.
        
        # Calculate d2u/dy2 again to subtract it
        d2u_dy2 = (u_old[:, i+1] - 2*u_old[:, i] + u_old[:, i-1]) / dy**2 # WAIT, indices are wrong for Y-col fn
        # For Y-sweep we need local d2u_dy2. No, in Y-sweep we solve for Y implicit.
        # The equation is (1 - alpha dyy) u_new = u_star - alpha * d2u_dy2_explicit
        # But wait, d2u_dy2 was computed using u_old.
        
        # Let's reconstruct consistent d2u_dy2 term used in X-sweep
        # In X-sweep we are iterating rows (j), so for row j, we used (j+1, j-1).
        # In Y-sweep we are iterating cols (i).
        # We need the d2u_dy2 term corresponding to (j, i) using u_old.
        
        # It is easier to pass the subtraction term: (alpha * d2u_dy2)
        # But efficiently doing that inside vmap is tricky.
        # Let's recompute:
        # d2u_dy2 at (j, i) = (u_old[j+1, i] - 2u[j,i] + u[j-1,i]) / dy^2
        return mask_col # Placeholder return
        
    # To implement Y-sweep cleanly with the "subtract explicit part" logic:
    # 1. Compute full Field of "Explicit Y-Diffusion" first
    d2u_dy2_all = jnp.zeros_like(u)
    d2u_dy2_all = d2u_dy2_all.at[1:-1, :].set(
        (u[2:, :] - 2*u[1:-1, :] + u[:-2, :]) / dy**2
    )
    
    # 2. X-Sweep
    def solve_x_row_2(j, rhs_row, explicit_y_row, mask_row):
        # RHS for tridiagonal solve
        rhs_target = rhs_row + alpha * explicit_y_row
        rhs_target = jnp.where(mask_row == 0, 0.0, rhs_target)
        
        a, b_coef, c = build_tridiagonal_masked_jax(nx, rx, mask_row)
        return thomas_algorithm_jax(a, b_coef, c, rhs_target)

    u_star = vmap(solve_x_row_2)(jnp.arange(ny), rhs, d2u_dy2_all, mask)
    
    # 3. Y-Sweep
    def solve_y_col_2(i, u_star_col, explicit_y_col, mask_col):
        # RHS for tridiagonal solve: u* - alpha * d2u_dy2 (removing the explicit part)
        rhs_target = u_star_col - alpha * explicit_y_col
        rhs_target = jnp.where(mask_col == 0, 0.0, rhs_target)
        
        a, b_coef, c = build_tridiagonal_masked_jax(ny, ry, mask_col)
        return thomas_algorithm_jax(a, b_coef, c, rhs_target)
        
    u_new = vmap(solve_y_col_2, in_axes=(0, 1, 1, 1))(jnp.arange(nx), u_star, d2u_dy2_all, mask)
    u_new = u_new.T # Transpose back
    
    return u_new * mask

# ============================================================================
# Pressure Solver (Masked + Neumann BCs)
# ============================================================================

@partial(jit, static_argnums=(4, 6, 7))
def pressure_poisson_masked(p, b, dx, dy, nit, mask, i_step, j_step):
    """
    Solve Pressure Poisson with Mask and Step BCs.
    
    BCs:
    - Global Walls: Neumann (dp/dn = 0) or Dirichlet (p=0 at outlet)
    - Step Walls: Neumann.
      - Top of step (y=h, x<0): dp/dy = 0 => p[j_step-1, :i_step] = p[j_step, :i_step]
         (Actually fluid is at j>=j_step. Solid is j<j_step.
          So at boundary j_step, ghost point at j_step-1 should equal j_step for dp/dy=0.
          We can set p[j_step-1] = p[j_step] in the solver logic.)
      - Vertical wall (x=0, y<h): dp/dx = 0 => p[:j_step, i_step-1] = p[:j_step, i_step]
    
    Using Jacobi/SOR iteration for simplicity with complex BCs.
    """
    
    def body_fn(p_old, _):
        # Standard Interior Update (Jacobi)
        # p_new = (dy^2(p_e + p_w) + dx^2(p_n + p_s) - b dx^2 dy^2) / (2(dx^2+dy^2))
        p_new = (((p_old[1:-1, 2:] + p_old[1:-1, :-2]) * dy**2 + 
                  (p_old[2:, 1:-1] + p_old[:-2, 1:-1]) * dx**2 - 
                  b[1:-1, 1:-1] * dx**2 * dy**2) / 
                 (2 * (dx**2 + dy**2)))
        
        p_full = p_old.at[1:-1, 1:-1].set(p_new)
        
        # --- Global BCs ---
        # Inlet (Left, x=0, y>h): dp/dx=0
        # (Indices: j from j_step to end, i=0)
        p_full = p_full.at[j_step:, 0].set(p_full[j_step:, 1])
        
        # Outlet (Right, x=L): p=0 (Dirichlet)
        p_full = p_full.at[:, -1].set(0.0)
        
        # Top Wall (y=H): dp/dy=0
        p_full = p_full.at[-1, :].set(p_full[-2, :])
        
        # Bottom Wall (y=0, x>0): dp/dy=0
        # (Indices: j=0, i from i_step to end)
        p_full = p_full.at[0, i_step:].set(p_full[1, i_step:])
        
        # --- Step BCs (Implementing Neumann at interface) ---
        # The solver stencil uses p[j-1] and p[j+1].
        # If we are at j=j_step (fluid), p[j_step-1] is solid.
        # To enforce dp/dy=0 at y=h (between j_step-1 and j_step),
        # we effectively want p[j_step-1] (ghost) = p[j_step].
        # So we explicitly set the solid "ghost" layer.
        
        # 1. Step Top Surface (Horizontal interface)
        # Fluid is at j_step. Solid is at j_step-1. Range x < 0 (i < i_step).
        p_full = p_full.at[j_step-1, :i_step].set(p_full[j_step, :i_step])
        
        # 2. Step Vertical Face (Vertical interface)
        # Fluid is at i_step. Solid is at i_step-1. Range y < h (j < j_step).
        p_full = p_full.at[:j_step, i_step-1].set(p_full[:j_step, i_step])
        
        # Zero out deep solid region to keep it clean (optional but good for plots)
        # Deep solid: j < j_step-1, i < i_step-1
        # p_full = p_full.at[:j_step-1, :i_step-1].set(0.0)
        
        return p_full, None

    # Scan loop
    p_final, _ = lax.scan(body_fn, p, None, length=nit)
    
    return p_final

# ============================================================================
# Core Time Step
# ============================================================================

@jit
def compute_advection(u, v, dx, dy):
    # First order upwind (stable but dissipative)
    u_c = u[1:-1, 1:-1]
    v_c = v[1:-1, 1:-1]
    
    # u advection
    du_dx_b = (u[1:-1, 1:-1] - u[1:-1, :-2]) / dx
    du_dx_f = (u[1:-1, 2:] - u[1:-1, 1:-1]) / dx
    du_dy_b = (u[1:-1, 1:-1] - u[:-2, 1:-1]) / dy
    du_dy_f = (u[2:, 1:-1] - u[1:-1, 1:-1]) / dy
    
    adv_u = jnp.zeros_like(u)
    val_u = (jnp.where(u_c > 0, u_c * du_dx_b, u_c * du_dx_f) +
             jnp.where(v_c > 0, v_c * du_dy_b, v_c * du_dy_f))
    adv_u = adv_u.at[1:-1, 1:-1].set(val_u)
    
    # v advection
    dv_dx_b = (v[1:-1, 1:-1] - v[1:-1, :-2]) / dx
    dv_dx_f = (v[1:-1, 2:] - v[1:-1, 1:-1]) / dx
    dv_dy_b = (v[1:-1, 1:-1] - v[:-2, 1:-1]) / dy
    dv_dy_f = (v[2:, 1:-1] - v[1:-1, 1:-1]) / dy
    
    adv_v = jnp.zeros_like(v)
    val_v = (jnp.where(u_c > 0, u_c * dv_dx_b, u_c * dv_dx_f) +
             jnp.where(v_c > 0, v_c * dv_dy_b, v_c * dv_dy_f))
    adv_v = adv_v.at[1:-1, 1:-1].set(val_v)
    
    return adv_u, adv_v

@partial(jit, static_argnums=(8, 11))
def time_step_implicit(u, v, p, dt, dx, dy, rho, nu, nit, mask, u_inlet, geom_params):
    """
    Single Implicit Fractional Step.
    """
    j_step = geom_params[0]
    i_step = geom_params[1]
    
    # 1. Advection (Explicit)
    adv_u, adv_v = compute_advection(u, v, dx, dy)
    
    # 2. Diffusion Predictor (Implicit Crank-Nicolson via ADI)
    alpha = nu * dt / 2.0
    
    # RHS for Helmholtz: u + alpha*Lap(u) - dt*adv
    # Explicit Lap for RHS
    lap_u = (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2])/dx**2 + \
            (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1])/dy**2
    
    rhs_u_inner = u[1:-1, 1:-1] + alpha * lap_u - dt * adv_u[1:-1, 1:-1]
    rhs_u = u.at[1:-1, 1:-1].set(rhs_u_inner)
    
    lap_v = (v[1:-1, 2:] - 2*v[1:-1, 1:-1] + v[1:-1, :-2])/dx**2 + \
            (v[2:, 1:-1] - 2*v[1:-1, 1:-1] + v[:-2, 1:-1])/dy**2
            
    rhs_v_inner = v[1:-1, 1:-1] + alpha * lap_v - dt * adv_v[1:-1, 1:-1]
    rhs_v = v.at[1:-1, 1:-1].set(rhs_v_inner)
    
    # Solve masked Helmholtz
    u_star = adi_step_masked(u, rhs_u, alpha, dx, dy, mask)
    v_star = adi_step_masked(v, rhs_v, alpha, dx, dy, mask)
    
    # Apply Inflow BCs to intermediate (Predictor should satisfy BCs approx)
    u_star = u_star.at[j_step:, 0].set(u_inlet)
    v_star = v_star.at[:, 0].set(0.0)
    
    # 3. Pressure Solve
    # Divergence
    div = jnp.zeros_like(p)
    div = div.at[1:-1, 1:-1].set(
        (u_star[1:-1, 2:] - u_star[1:-1, :-2]) / (2 * dx) +
        (v_star[2:, 1:-1] - v_star[:-2, 1:-1]) / (2 * dy)
    )
    b_term = (rho / dt) * div
    
    p_new = pressure_poisson_masked(p, b_term, dx, dy, nit, mask, i_step, j_step)
    
    # 4. Corrector
    u_new = u_star.copy()
    v_new = v_star.copy()
    
    dp_dx = (p_new[1:-1, 2:] - p_new[1:-1, :-2]) / (2 * dx)
    dp_dy = (p_new[2:, 1:-1] - p_new[:-2, 1:-1]) / (2 * dy)
    
    u_new = u_new.at[1:-1, 1:-1].add(-(dt/rho) * dp_dx)
    v_new = v_new.at[1:-1, 1:-1].add(-(dt/rho) * dp_dy)
    
    # Enforce Mask
    u_new = u_new * mask
    v_new = v_new * mask
    
    # Boundaries
    # Inlet
    u_new = u_new.at[j_step:, 0].set(u_inlet)
    v_new = v_new.at[:, 0].set(0.0)
    # Walls (Top/Bottom)
    u_new = u_new.at[-1, :].set(0.0); v_new = v_new.at[-1, :].set(0.0) # Top
    u_new = u_new.at[0, :].set(0.0); v_new = v_new.at[0, :].set(0.0) # Bottom (full row, masked parts 0 anyway)
    # Outlet (Neumann)
    u_new = u_new.at[:, -1].set(u_new[:, -2])
    v_new = v_new.at[:, -1].set(v_new[:, -2])
    
    # Calculate CFL
    cfl = dt * (jnp.max(jnp.abs(u_new))/dx + jnp.max(jnp.abs(v_new))/dy)
    
    return u_new, v_new, p_new, cfl

# ============================================================================
# Main Loop
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Implicit BFS Solver (JAX)")
    parser.add_argument("--re", type=float, default=100.0)
    parser.add_argument("--nx", type=int, default=201)
    parser.add_argument("--ny", type=int, default=51)
    parser.add_argument("--dt", type=float, default=0.005) # Larger dt for implicit
    parser.add_argument("--nt", type=int, default=2000)
    parser.add_argument("--nit", type=int, default=50) # Pressure iters
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    
    print("="*60)
    print(f"Implicit Backward-Facing Step (Re={args.re})")
    print("="*60)
    
    # Geometry
    geom = setup_geometry(args.nx, args.ny)
    mask = jnp.array(geom['mask'])
    dx, dy = geom['dx'], geom['dy']
    j_step, i_step = geom['j_step'], geom['i_step']
    
    # Physics
    u_max = 1.5
    h_step = geom['dims']['h']
    nu = calculate_nu_from_reynolds(args.re, u_max, h_step)
    rho = 1.0
    
    print(f"Grid: {args.nx}x{args.ny}, dt={args.dt}")
    print(f"Nu: {nu:.6f}")
    
    # Init Fields
    u = jnp.zeros((args.ny, args.nx))
    v = jnp.zeros((args.ny, args.nx))
    p = jnp.zeros((args.ny, args.nx))
    
    # Inlet Profile
    inlet_y = geom['y'][j_step:] - geom['y'][j_step]
    u_inlet_prof = jnp.array(get_inlet_profile(inlet_y, geom['dims']['h_inlet'], u_max))
    
    # Compilation
    print("Compiling JAX functions...")
    geom_params = (j_step, i_step)
    _ = time_step_implicit(u, v, p, args.dt, dx, dy, rho, nu, args.nit, mask, u_inlet_prof, geom_params)
    print("Compilation complete.")
    
    # Loop
    start = time.time()
    for n in tqdm(range(args.nt), disable=args.verbose):
        u, v, p, cfl = time_step_implicit(u, v, p, args.dt, dx, dy, rho, nu, args.nit, mask, u_inlet_prof, geom_params)
        
        if args.verbose and n % 100 == 0:
            cfl_val = float(cfl)
            print(f"Step {n}: CFL={cfl_val:.4f}, Max U={float(jnp.max(u)):.4f}")
            if jnp.isnan(cfl):
                 print("NaN detected!")
                 break
                 
    duration = time.time() - start
    print(f"\nCompleted in {duration:.2f}s ({args.nt/duration:.1f} iter/s)")
    
    # Save
    os.makedirs("simulations/backward_facing_step_implicit/outputs", exist_ok=True)
    np.savez("simulations/backward_facing_step_implicit/outputs/solution_implicit.npz", 
             u=u, v=v, p=p, x=geom['x'], y=geom['y'], mask=geom['mask'])
    print("Saved solution.")

    # Try plotting
    try:
        plot_results(np.array(u), np.array(v), np.array(p), geom, 
                     "simulations/backward_facing_step_implicit", args.re, "jax_implicit")
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    main()
