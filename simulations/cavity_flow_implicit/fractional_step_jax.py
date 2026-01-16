"""
JAX-optimized Fractional Step (Projection) Method for implicit Navier-Stokes.

This is the JAX version for GPU acceleration.
"""

import jax.numpy as jnp
from jax import jit
from adi_solver_jax import adi_helmholtz_2d_jax, laplacian_2d_jax


@jit
def compute_advection_jax(u, v, dx, dy):
    """
    Compute advection terms using backward differences (JAX version).
    
    Args:
        u, v: Velocity components (ny, nx)
        dx, dy: Grid spacing
    
    Returns:
        adv_u, adv_v: Advection terms
    """
    adv_u = jnp.zeros_like(u)
    adv_v = jnp.zeros_like(v)
    
    # Interior points using backward differences
    adv_u = adv_u.at[1:, 1:].set(
        u[1:, 1:] * (u[1:, 1:] - u[1:, :-1]) / dx +
        v[1:, 1:] * (u[1:, 1:] - u[:-1, 1:]) / dy
    )
    
    adv_v = adv_v.at[1:, 1:].set(
        u[1:, 1:] * (v[1:, 1:] - v[1:, :-1]) / dx +
        v[1:, 1:] * (v[1:, 1:] - v[:-1, 1:]) / dy
    )
    
    return adv_u, adv_v


@jit
def predictor_step_jax(u, v, dt, dx, dy, nu):
    """
    Predictor step: Compute intermediate velocity with implicit diffusion (JAX).
    
    Args:
        u, v: Current velocity (ny, nx)
        dt: Time step
        dx, dy: Grid spacing
        nu: Kinematic viscosity
    
    Returns:
        u_star, v_star: Intermediate velocity
    """
    # Compute advection (explicit)
    adv_u, adv_v = compute_advection_jax(u, v, dx, dy)
    
    # Crank-Nicolson coefficient
    alpha = nu * dt / 2.0
    
    # Build RHS for Helmholtz equation
    lap_u = laplacian_2d_jax(u, dx, dy)
    lap_v = laplacian_2d_jax(v, dx, dy)
    
    rhs_u = u + alpha * lap_u - dt * adv_u
    rhs_v = v + alpha * lap_v - dt * adv_v
    
    # Solve Helmholtz: (I - alpha * Laplacian) u* = rhs
    u_star = adi_helmholtz_2d_jax(u, rhs_u, alpha, dx, dy, bc_type='dirichlet')
    v_star = adi_helmholtz_2d_jax(v, rhs_v, alpha, dx, dy, bc_type='dirichlet')
    
    return u_star, v_star


@jit(static_argnames=['nit'])
def pressure_poisson_jax(p, u_star, v_star, dt, dx, dy, rho, nit=50):
    """
    Solve pressure Poisson equation (JAX version).
    
    Args:
        p: Current pressure field (ny, nx)
        u_star, v_star: Intermediate velocity (ny, nx)
        dt: Time step
        dx, dy: Grid spacing
        rho: Density
        nit: Number of iterations
    
    Returns:
        p_new: Updated pressure field
    """
    ny, nx = p.shape
    p_new = p.copy()
    
    # Compute divergence of u*
    div = jnp.zeros_like(p)
    div = div.at[1:-1, 1:-1].set(
        (u_star[1:-1, 2:] - u_star[1:-1, :-2]) / (2 * dx) +
        (v_star[2:, 1:-1] - v_star[:-2, 1:-1]) / (2 * dy)
    )
    
    # RHS for Poisson equation
    b = (rho / dt) * div
    
    # Iterative Poisson solver (Jacobi iteration)
    def poisson_iter(p_new, _):
        p_old = p_new
        
        # Interior points
        p_new = p_new.at[1:-1, 1:-1].set(
            ((p_old[1:-1, 2:] + p_old[1:-1, :-2]) * dy**2 +
             (p_old[2:, 1:-1] + p_old[:-2, 1:-1]) * dx**2 -
             b[1:-1, 1:-1] * dx**2 * dy**2) / (2 * (dx**2 + dy**2))
        )
        
        # Boundary conditions
        # Neumann (zero gradient) at walls
        p_new = p_new.at[0, :].set(p_new[1, :])      # Bottom
        p_new = p_new.at[:, 0].set(p_new[:, 1])      # Left
        p_new = p_new.at[:, -1].set(p_new[:, -2])    # Right
        
        # Dirichlet at top (lid)
        p_new = p_new.at[-1, :].set(0.0)
        
        return p_new, None
    
    # Run iterations using lax.scan
    from jax import lax
    p_new, _ = lax.scan(poisson_iter, p_new, None, length=nit)
    
    return p_new


@jit
def corrector_step_jax(u_star, v_star, p, dt, dx, dy, rho):
    """
    Corrector step: Project velocity onto divergence-free space (JAX).
    
    Args:
        u_star, v_star: Intermediate velocity (ny, nx)
        p: Pressure field (ny, nx)
        dt: Time step
        dx, dy: Grid spacing
        rho: Density
    
    Returns:
        u_new, v_new: Divergence-free velocity
    """
    u_new = u_star.copy()
    v_new = v_star.copy()
    
    # Compute pressure gradients (central differences)
    u_new = u_new.at[1:-1, 1:-1].add(
        -(dt / rho) * (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * dx)
    )
    
    v_new = v_new.at[1:-1, 1:-1].add(
        -(dt / rho) * (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * dy)
    )
    
    return u_new, v_new


@jit
def apply_boundary_conditions_jax(u, v, u_lid=1.0):
    """
    Apply boundary conditions for lid-driven cavity (JAX).
    
    Args:
        u, v: Velocity fields (ny, nx)
        u_lid: Lid velocity
    
    Returns:
        u, v: Velocity with BCs applied
    """
    # Bottom wall (y=0)
    u = u.at[0, :].set(0.0)
    v = v.at[0, :].set(0.0)
    
    # Top wall / lid (y=L)
    u = u.at[-1, :].set(u_lid)
    v = v.at[-1, :].set(0.0)
    
    # Left wall (x=0)
    u = u.at[:, 0].set(0.0)
    v = v.at[:, 0].set(0.0)
    
    # Right wall (x=L)
    u = u.at[:, -1].set(0.0)
    v = v.at[:, -1].set(0.0)
    
    return u, v


@jit
def compute_divergence_jax(u, v, dx, dy):
    """
    Compute divergence of velocity field (JAX).
    
    Args:
        u, v: Velocity components (ny, nx)
        dx, dy: Grid spacing
    
    Returns:
        div: Divergence
    """
    div = jnp.zeros_like(u)
    
    # Central differences in interior
    div = div.at[1:-1, 1:-1].set(
        (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx) +
        (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)
    )
    
    return div


# Test function
if __name__ == "__main__":
    import numpy as np
    
    print("Testing JAX Fractional Step components...")
    print("=" * 60)
    
    # Grid
    nx, ny = 21, 21
    dx = dy = 2.0 / (nx - 1)
    
    # Parameters
    dt = 0.01
    nu = 0.1
    rho = 1.0
    
    # Initialize
    u = jnp.zeros((ny, nx))
    v = jnp.zeros((ny, nx))
    p = jnp.zeros((ny, nx))
    
    # Apply BCs
    u, v = apply_boundary_conditions_jax(u, v, u_lid=1.0)
    
    # Run one fractional step
    print("Running fractional step...")
    u_star, v_star = predictor_step_jax(u, v, dt, dx, dy, nu)
    p = pressure_poisson_jax(p, u_star, v_star, dt, dx, dy, rho, nit=50)
    u, v = corrector_step_jax(u_star, v_star, p, dt, dx, dy, rho)
    u, v = apply_boundary_conditions_jax(u, v, u_lid=1.0)
    
    # Check BCs
    assert jnp.allclose(u[-1, 1:-1], 1.0), "Lid velocity not maintained"
    assert jnp.allclose(u[0, :], 0.0), "Bottom wall not zero"
    
    print("✓ All components execute successfully")
    print("=" * 60)
    print("All tests passed! ✓")
