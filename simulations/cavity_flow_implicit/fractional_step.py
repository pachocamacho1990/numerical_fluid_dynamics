"""
Fractional Step (Projection) Method components for implicit Navier-Stokes.

This module implements the three steps of the Chorin-Temam projection method:
1. Predictor: Compute intermediate velocity (implicit diffusion, explicit advection)
2. Pressure: Solve Poisson equation to enforce incompressibility
3. Corrector: Project velocity onto divergence-free space

The predictor uses Crank-Nicolson for diffusion (unconditionally stable).
"""

import numpy as np
from adi_solver import adi_helmholtz_2d, laplacian_2d


def compute_advection(u, v, dx, dy):
    """
    Compute advection terms using backward differences (upwind).
    
    Advection terms:
        adv_u = u * du/dx + v * du/dy
        adv_v = u * dv/dx + v * dv/dy
    
    Args:
        u, v: Velocity components (ny, nx)
        dx, dy: Grid spacing
    
    Returns:
        adv_u, adv_v: Advection terms (ny, nx)
    """
    ny, nx = u.shape
    
    adv_u = np.zeros_like(u)
    adv_v = np.zeros_like(v)
    
    # Interior points using backward differences
    # du/dx ≈ (u[i,j] - u[i,j-1]) / dx
    # du/dy ≈ (u[i,j] - u[i-1,j]) / dy
    
    adv_u[1:, 1:] = (
        u[1:, 1:] * (u[1:, 1:] - u[1:, :-1]) / dx +
        v[1:, 1:] * (u[1:, 1:] - u[:-1, 1:]) / dy
    )
    
    adv_v[1:, 1:] = (
        u[1:, 1:] * (v[1:, 1:] - v[1:, :-1]) / dx +
        v[1:, 1:] * (v[1:, 1:] - v[:-1, 1:]) / dy
    )
    
    return adv_u, adv_v


def predictor_step(u, v, dt, dx, dy, nu):
    """
    Predictor step: Compute intermediate velocity with implicit diffusion.
    
    Solves:
        (u* - u^n) / dt = -advection(u^n) + nu/2 * (Laplacian(u*) + Laplacian(u^n))
    
    This is Crank-Nicolson for diffusion (unconditionally stable).
    
    Args:
        u, v: Current velocity (ny, nx)
        dt: Time step
        dx, dy: Grid spacing
        nu: Kinematic viscosity
    
    Returns:
        u_star, v_star: Intermediate velocity (ny, nx)
    """
    # Compute advection (explicit)
    adv_u, adv_v = compute_advection(u, v, dx, dy)
    
    # Crank-Nicolson coefficient
    alpha = nu * dt / 2.0
    
    # Build RHS for Helmholtz equation
    # RHS = u + (nu*dt/2) * Laplacian(u) - dt * advection
    lap_u = laplacian_2d(u, dx, dy)
    lap_v = laplacian_2d(v, dx, dy)
    
    rhs_u = u + alpha * lap_u - dt * adv_u
    rhs_v = v + alpha * lap_v - dt * adv_v
    
    # Solve Helmholtz: (I - alpha * Laplacian) u* = rhs
    u_star = adi_helmholtz_2d(u, rhs_u, alpha, dx, dy, bc_type='dirichlet')
    v_star = adi_helmholtz_2d(v, rhs_v, alpha, dx, dy, bc_type='dirichlet')
    
    return u_star, v_star


def pressure_poisson(p, u_star, v_star, dt, dx, dy, rho, nit=50):
    """
    Solve pressure Poisson equation to enforce incompressibility.
    
    Solves:
        Laplacian(phi) = (rho / dt) * divergence(u*)
    
    where phi is the pressure correction.
    
    Args:
        p: Current pressure field (ny, nx)
        u_star, v_star: Intermediate velocity (ny, nx)
        dt: Time step
        dx, dy: Grid spacing
        rho: Density
        nit: Number of iterations
    
    Returns:
        p_new: Updated pressure field (ny, nx)
    """
    ny, nx = p.shape
    p_new = p.copy()
    
    # Compute divergence of u*
    # div = du*/dx + dv*/dy
    div = np.zeros_like(p)
    div[1:-1, 1:-1] = (
        (u_star[1:-1, 2:] - u_star[1:-1, :-2]) / (2 * dx) +
        (v_star[2:, 1:-1] - v_star[:-2, 1:-1]) / (2 * dy)
    )
    
    # RHS for Poisson equation
    b = (rho / dt) * div
    
    # Iterative Poisson solver (Jacobi iteration)
    for it in range(nit):
        p_old = p_new.copy()
        
        # Interior points
        p_new[1:-1, 1:-1] = (
            (p_old[1:-1, 2:] + p_old[1:-1, :-2]) * dy**2 +
            (p_old[2:, 1:-1] + p_old[:-2, 1:-1]) * dx**2 -
            b[1:-1, 1:-1] * dx**2 * dy**2
        ) / (2 * (dx**2 + dy**2))
        
        # Boundary conditions
        # Neumann (zero gradient) at walls
        p_new[0, :] = p_new[1, :]      # Bottom
        p_new[:, 0] = p_new[:, 1]      # Left
        p_new[:, -1] = p_new[:, -2]    # Right
        
        # Dirichlet at top (lid)
        p_new[-1, :] = 0.0
    
    return p_new


def corrector_step(u_star, v_star, p, dt, dx, dy, rho):
    """
    Corrector step: Project velocity onto divergence-free space.
    
    Computes:
        u^{n+1} = u* - (dt / rho) * dp/dx
        v^{n+1} = v* - (dt / rho) * dp/dy
    
    Args:
        u_star, v_star: Intermediate velocity (ny, nx)
        p: Pressure field (ny, nx)
        dt: Time step
        dx, dy: Grid spacing
        rho: Density
    
    Returns:
        u_new, v_new: Divergence-free velocity (ny, nx)
    """
    u_new = u_star.copy()
    v_new = v_star.copy()
    
    # Compute pressure gradients (central differences)
    # dp/dx
    u_new[1:-1, 1:-1] -= (dt / rho) * (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * dx)
    
    # dp/dy
    v_new[1:-1, 1:-1] -= (dt / rho) * (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * dy)
    
    return u_new, v_new


def apply_boundary_conditions(u, v, u_lid=1.0):
    """
    Apply boundary conditions for lid-driven cavity.
    
    - Walls (bottom, left, right): u = v = 0 (no-slip)
    - Lid (top): u = u_lid, v = 0 (moving lid)
    
    Args:
        u, v: Velocity fields (ny, nx)
        u_lid: Lid velocity (default: 1.0)
    
    Returns:
        u, v: Velocity with BCs applied
    """
    # Bottom wall (y=0)
    u[0, :] = 0.0
    v[0, :] = 0.0
    
    # Top wall / lid (y=L)
    u[-1, :] = u_lid
    v[-1, :] = 0.0
    
    # Left wall (x=0)
    u[:, 0] = 0.0
    v[:, 0] = 0.0
    
    # Right wall (x=L)
    u[:, -1] = 0.0
    v[:, -1] = 0.0
    
    return u, v


def compute_divergence(u, v, dx, dy):
    """
    Compute divergence of velocity field.
    
    div = du/dx + dv/dy
    
    Args:
        u, v: Velocity components (ny, nx)
        dx, dy: Grid spacing
    
    Returns:
        div: Divergence (ny, nx)
    """
    div = np.zeros_like(u)
    
    # Central differences in interior
    div[1:-1, 1:-1] = (
        (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx) +
        (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)
    )
    
    return div


# Test function
def test_fractional_step():
    """Test the complete fractional step."""
    print("Testing Fractional Step components...")
    
    # Grid
    nx, ny = 21, 21
    Lx, Ly = 1.0, 1.0
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    
    # Parameters
    dt = 0.01
    nu = 0.1
    rho = 1.0
    
    # Initialize fields
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.zeros((ny, nx))
    
    # Apply initial BCs
    u, v = apply_boundary_conditions(u, v, u_lid=1.0)
    
    # Run multiple timesteps to let the method converge
    print("  Running 10 timesteps...")
    for step in range(10):
        u_star, v_star = predictor_step(u, v, dt, dx, dy, nu)
        p = pressure_poisson(p, u_star, v_star, dt, dx, dy, rho, nit=100)
        u, v = corrector_step(u_star, v_star, p, dt, dx, dy, rho)
        u, v = apply_boundary_conditions(u, v, u_lid=1.0)
    
    # Check divergence after convergence
    div = compute_divergence(u, v, dx, dy)
    max_div = np.max(np.abs(div[1:-1, 1:-1]))
    
    print(f"\n  Max divergence after 10 steps: {max_div:.2e}")
    # After several steps, divergence should be small
    assert max_div < 0.05, f"Divergence too large: {max_div}"
    
    # Check that lid velocity is maintained
    assert np.allclose(u[-1, 1:-1], 1.0), "Lid velocity not maintained"
    
    print("  ✓ Fractional step test passed")


if __name__ == "__main__":
    print("Testing Fractional Step Method Components...")
    print("=" * 60)
    
    test_fractional_step()
    
    print("=" * 60)
    print("All tests passed! ✓")
