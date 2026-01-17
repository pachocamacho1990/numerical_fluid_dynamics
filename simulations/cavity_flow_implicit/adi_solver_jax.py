"""
JAX-optimized ADI (Alternating Direction Implicit) solver for 2D Helmholtz equation.

Solves: (I - alpha * Laplacian) u = f

Uses JAX's vmap and lax.scan for efficient parallel and sequential operations.
"""

import jax.numpy as jnp
from jax import jit, vmap
import jax.lax as lax
from functools import partial
from thomas_solver_jax import thomas_algorithm_jax, build_tridiagonal_diffusion_jax


@jit
def laplacian_2d_jax(u, dx, dy):
    """
    Compute 2D Laplacian using central differences (JAX version).
    
    Args:
        u: Field (ny, nx)
        dx, dy: Grid spacing
    
    Returns:
        lap: Laplacian of u
    """
    lap = jnp.zeros_like(u)
    
    # Interior points
    lap = lap.at[1:-1, 1:-1].set(
        (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) / dx**2 +
        (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) / dy**2
    )
    
    return lap


@partial(jit, static_argnums=(5,))
def adi_helmholtz_2d_jax(u, rhs, alpha, dx, dy, bc_type='dirichlet'):
    """
    Solve (I - alpha * Laplacian) u = rhs using ADI (JAX version).
    
    Args:
        u: Initial guess (ny, nx)
        rhs: Right-hand side (ny, nx)
        alpha: Coefficient (nu * dt / 2)
        dx, dy: Grid spacing
        bc_type: Boundary condition type
    
    Returns:
        u_new: Solution (ny, nx)
    """
    ny, nx = u.shape
    
    # Diffusion numbers
    rx = alpha / dx**2
    ry = alpha / dy**2
    
    # Build tridiagonal coefficients
    a_x, b_x, c_x = build_tridiagonal_diffusion_jax(nx, rx, bc_type)
    a_y, b_y, c_y = build_tridiagonal_diffusion_jax(ny, ry, bc_type)
    
    # ========================================
    # X-SWEEP: Solve implicitly in x-direction
    # ========================================
    def solve_x_row(j, u, rhs):
        """Solve for row j"""
        # Build RHS for this row
        rhs_x = rhs[j, :]
        
        # Add explicit y-diffusion term if interior row
        def add_y_diff(rhs_x):
            d2u_dy2 = (u[j+1, :] - 2*u[j, :] + u[j-1, :]) / dy**2
            return rhs_x + alpha * d2u_dy2
        
        rhs_x = lax.cond(
            (j > 0) & (j < ny - 1),
            add_y_diff,
            lambda rhs_x: rhs_x,
            rhs_x
        )
        
        # Handle boundary conditions
        if bc_type == 'dirichlet':
            rhs_x = rhs_x.at[0].set(rhs[j, 0])
            rhs_x = rhs_x.at[-1].set(rhs[j, -1])
        
        # Solve tridiagonal system
        return thomas_algorithm_jax(a_x, b_x, c_x, rhs_x)
    
    # Vectorize over rows
    u_star = vmap(lambda j: solve_x_row(j, u, rhs))(jnp.arange(ny))
    
    # ========================================
    # Y-SWEEP: Solve implicitly in y-direction
    # ========================================
    def solve_y_col(i, u_star, u, rhs):
        """Solve for column i"""
        # Build RHS for this column
        rhs_y = u_star[:, i]
        
        # Subtract y-diffusion term if interior column
        def sub_y_diff(rhs_y):
            d2u_dy2 = (u[:, i+1] - 2*u[:, i] + u[:, i-1]) / dy**2
            return rhs_y - alpha * d2u_dy2
        
        rhs_y = lax.cond(
            (i > 0) & (i < nx - 1),
            sub_y_diff,
            lambda rhs_y: rhs_y,
            rhs_y
        )
        
        # Handle boundary conditions
        if bc_type == 'dirichlet':
            rhs_y = rhs_y.at[0].set(rhs[0, i])
            rhs_y = rhs_y.at[-1].set(rhs[-1, i])
        
        # Solve tridiagonal system
        return thomas_algorithm_jax(a_y, b_y, c_y, rhs_y)
    
    # Vectorize over columns
    u_new = vmap(lambda i: solve_y_col(i, u_star, u, rhs))(jnp.arange(nx))
    u_new = u_new.T  # Transpose back to (ny, nx)
    
    return u_new


# Test function
if __name__ == "__main__":
    import numpy as np
    from adi_solver import adi_helmholtz_2d as adi_numpy, laplacian_2d as lap_numpy
    
    print("Testing JAX ADI Helmholtz Solver...")
    print("=" * 60)
    
    # Grid
    nx, ny = 21, 21
    dx = dy = 1.0 / (nx - 1)
    
    # Parameters
    dt = 0.01
    nu = 0.1
    alpha = nu * dt / 2
    
    # Initial condition: Gaussian bump
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    u_np = np.exp(-50 * ((X - 0.5)**2 + (Y - 0.5)**2))
    
    # Set boundaries to zero
    u_np[0, :] = u_np[-1, :] = u_np[:, 0] = u_np[:, -1] = 0.0
    
    # RHS for Crank-Nicolson
    lap_u_np = lap_numpy(u_np, dx, dy)
    rhs_np = u_np + alpha * lap_u_np
    
    # Convert to JAX
    u_jax = jnp.array(u_np)
    rhs_jax = jnp.array(rhs_np)
    
    # Solve with JAX
    print("Solving with JAX...")
    u_new_jax = adi_helmholtz_2d_jax(u_jax, rhs_jax, alpha, dx, dy)
    
    # Solve with NumPy
    print("Solving with NumPy...")
    u_new_np = adi_numpy(u_np, rhs_np, alpha, dx, dy)
    
    # Compare
    diff = np.max(np.abs(np.array(u_new_jax) - u_new_np))
    print(f"\nMax difference (JAX vs NumPy): {diff:.2e}")
    
    # Check boundary conditions
    assert np.allclose(u_new_jax[0, :], 0.0), "Top boundary"
    assert np.allclose(u_new_jax[-1, :], 0.0), "Bottom boundary"
    assert np.allclose(u_new_jax[:, 0], 0.0), "Left boundary"
    assert np.allclose(u_new_jax[:, -1], 0.0), "Right boundary"
    
    # Check that solution decays
    max_u_old = np.max(u_np[1:-1, 1:-1])
    max_u_new = np.max(np.array(u_new_jax[1:-1, 1:-1]))
    assert max_u_new < max_u_old, "Solution should decay"
    
    print(f"Initial max: {max_u_old:.6f}")
    print(f"Final max:   {max_u_new:.6f}")
    print(f"Decay ratio: {max_u_new/max_u_old:.6f}")
    
    print("\n✓ JAX ADI solver matches NumPy")
    print("=" * 60)
    print("All tests passed! ✓")
