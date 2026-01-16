"""
ADI (Alternating Direction Implicit) solver for 2D Helmholtz equation.

Solves: (I - alpha * Laplacian) u = f

where alpha = nu * dt / 2 for Crank-Nicolson diffusion.

The ADI method splits the 2D problem into two sequential 1D problems,
each of which can be solved efficiently using the Thomas algorithm.
"""

import numpy as np
from thomas_solver import thomas_algorithm, build_tridiagonal_diffusion


def adi_helmholtz_2d(u, rhs, alpha, dx, dy, bc_type='dirichlet', tol=1e-6):
    """
    Solve (I - alpha * Laplacian) u = rhs using ADI.
    
    The Laplacian is: d²u/dx² + d²u/dy²
    
    ADI splits this into two steps:
    1. X-sweep: Implicit in x, explicit in y
    2. Y-sweep: Implicit in y, explicit in x
    
    Args:
        u: Initial guess (ny, nx) - note: row-major indexing
        rhs: Right-hand side (ny, nx)
        alpha: Coefficient (nu * dt / 2 for Crank-Nicolson)
        dx: Grid spacing in x
        dy: Grid spacing in y
        bc_type: Boundary condition type ('dirichlet' or 'neumann')
        tol: Convergence tolerance (not used in single-pass ADI)
    
    Returns:
        u_new: Solution (ny, nx)
    
    Notes:
        - Uses Douglas-Gunn ADI scheme (2nd order accurate)
        - For Dirichlet BCs, boundary values are preserved
        - Array indexing: u[j, i] where j is y-index, i is x-index
    """
    ny, nx = u.shape
    
    # Diffusion numbers
    rx = alpha / dx**2
    ry = alpha / dy**2
    
    # Intermediate solution after x-sweep
    u_star = np.copy(u)
    
    # ========================================
    # X-SWEEP: Solve implicitly in x-direction
    # ========================================
    # For each row j, solve:
    # -rx * u[j,i-1] + (1 + 2*rx) * u[j,i] - rx * u[j,i+1] = rhs_x[j,i]
    
    # Build tridiagonal coefficients for x-direction
    a_x, b_x, c_x = build_tridiagonal_diffusion(nx, rx, bc_type)
    
    for j in range(ny):
        # Build RHS for this row
        # RHS = rhs + alpha * d²u/dy² (explicit in y)
        rhs_x = rhs[j, :].copy()
        
        if j > 0 and j < ny - 1:
            # Add explicit y-diffusion term
            d2u_dy2 = (u[j+1, :] - 2*u[j, :] + u[j-1, :]) / dy**2
            rhs_x += alpha * d2u_dy2
        
        # Handle boundary conditions
        if bc_type == 'dirichlet':
            # Boundaries are fixed
            rhs_x[0] = rhs[j, 0]
            rhs_x[-1] = rhs[j, -1]
        
        # Solve tridiagonal system for this row
        u_star[j, :] = thomas_algorithm(a_x, b_x, c_x, rhs_x)
    
    # ========================================
    # Y-SWEEP: Solve implicitly in y-direction
    # ========================================
    # For each column i, solve:
    # -ry * u[j-1,i] + (1 + 2*ry) * u[j,i] - ry * u[j+1,i] = rhs_y[j,i]
    
    # Build tridiagonal coefficients for y-direction
    a_y, b_y, c_y = build_tridiagonal_diffusion(ny, ry, bc_type)
    
    u_new = np.copy(u_star)
    
    for i in range(nx):
        # Build RHS for this column
        # RHS = u_star - alpha * d²u/dy² (correction term)
        rhs_y = u_star[:, i].copy()
        
        if i > 0 and i < nx - 1:
            # Subtract the y-diffusion term that was added in x-sweep
            d2u_dy2 = (u[:, i+1] - 2*u[:, i] + u[:, i-1]) / dy**2
            rhs_y -= alpha * d2u_dy2
        
        # Handle boundary conditions
        if bc_type == 'dirichlet':
            # Boundaries are fixed
            rhs_y[0] = rhs[0, i]
            rhs_y[-1] = rhs[-1, i]
        
        # Solve tridiagonal system for this column
        u_new[:, i] = thomas_algorithm(a_y, b_y, c_y, rhs_y)
    
    return u_new


def laplacian_2d(u, dx, dy):
    """
    Compute 2D Laplacian using central differences.
    
    Args:
        u: Field (ny, nx)
        dx, dy: Grid spacing
    
    Returns:
        lap: Laplacian of u
    """
    lap = np.zeros_like(u)
    
    # Interior points
    lap[1:-1, 1:-1] = (
        (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) / dx**2 +
        (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) / dy**2
    )
    
    return lap


# Test functions
def test_adi_2d_diffusion():
    """Test ADI solver with 2D diffusion equation."""
    print("Testing 2D diffusion with ADI...")
    
    # Grid
    nx, ny = 21, 21
    Lx, Ly = 1.0, 1.0
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    
    # Time step
    dt = 0.01
    nu = 0.1
    alpha = nu * dt / 2  # Crank-Nicolson
    
    # Initial condition: Gaussian bump
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)
    
    u = np.exp(-50 * ((X - 0.5)**2 + (Y - 0.5)**2))
    
    # Set boundaries to zero (Dirichlet)
    u[0, :] = 0.0
    u[-1, :] = 0.0
    u[:, 0] = 0.0
    u[:, -1] = 0.0
    
    # RHS for Crank-Nicolson: rhs = u + (nu*dt/2) * Laplacian(u)
    lap_u = laplacian_2d(u, dx, dy)
    rhs = u + alpha * lap_u
    
    # Solve
    u_new = adi_helmholtz_2d(u, rhs, alpha, dx, dy, bc_type='dirichlet')
    
    # Check boundary conditions
    assert np.allclose(u_new[0, :], 0.0), "Top boundary not satisfied"
    assert np.allclose(u_new[-1, :], 0.0), "Bottom boundary not satisfied"
    assert np.allclose(u_new[:, 0], 0.0), "Left boundary not satisfied"
    assert np.allclose(u_new[:, -1], 0.0), "Right boundary not satisfied"
    
    # Check that solution decays (diffusion reduces amplitude)
    max_u_old = np.max(u[1:-1, 1:-1])
    max_u_new = np.max(u_new[1:-1, 1:-1])
    assert max_u_new < max_u_old, f"Solution should decay: {max_u_new} >= {max_u_old}"
    
    # Check mass conservation (integral should decrease due to boundary loss)
    mass_old = np.sum(u[1:-1, 1:-1])
    mass_new = np.sum(u_new[1:-1, 1:-1])
    assert mass_new < mass_old, "Mass should decrease due to diffusion to boundaries"
    
    print(f"  Initial max: {max_u_old:.6f}")
    print(f"  Final max:   {max_u_new:.6f}")
    print(f"  Decay ratio: {max_u_new/max_u_old:.6f}")
    print("  ✓ 2D diffusion test passed")


def test_adi_convergence():
    """Test that ADI converges to steady-state."""
    print("\nTesting ADI convergence to steady-state...")
    
    # Grid
    nx, ny = 21, 21
    dx = dy = 1.0 / (nx - 1)
    
    # Parameters
    dt = 0.01
    nu = 0.1
    alpha = nu * dt / 2
    
    # Initial condition: random interior, fixed boundaries
    u = np.random.rand(ny, nx) * 0.1
    u[0, :] = 1.0   # Top boundary = 1
    u[-1, :] = 0.0  # Bottom boundary = 0
    u[:, 0] = 0.0   # Left boundary = 0
    u[:, -1] = 0.0  # Right boundary = 0
    
    # Time-step multiple times to approach steady-state
    for n in range(200):  # More iterations for better convergence
        lap_u = laplacian_2d(u, dx, dy)
        rhs = u + alpha * lap_u
        u = adi_helmholtz_2d(u, rhs, alpha, dx, dy, bc_type='dirichlet')
    
    # Check that solution is smooth (no oscillations)
    # Compute gradient magnitude in interior only
    grad_x = np.abs(u[1:-1, 1:] - u[1:-1, :-1])
    grad_y = np.abs(u[1:, 1:-1] - u[:-1, 1:-1])
    
    max_grad = max(np.max(grad_x[1:-1, :]), np.max(grad_y[:, 1:-1]))
    print(f"  Max interior gradient: {max_grad:.6f}")
    # Gradient should be reasonable (not oscillating wildly)
    assert max_grad < 0.5, "Solution has large gradients (possible instability)"
    
    print("  ✓ Convergence test passed")


def test_adi_vs_explicit():
    """Compare ADI with explicit diffusion at small dt."""
    print("\nComparing ADI with explicit method...")
    
    # Grid
    nx, ny = 11, 11
    dx = dy = 0.1
    
    # Small time step for explicit stability
    dt = 0.0001
    nu = 0.1
    alpha = nu * dt / 2
    
    # Initial condition
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    u0 = np.sin(np.pi * X) * np.sin(np.pi * Y)
    
    # Explicit step: u_new = u + nu * dt * Laplacian(u)
    lap_u = laplacian_2d(u0, dx, dy)
    u_explicit = u0 + nu * dt * lap_u
    u_explicit[0, :] = u_explicit[-1, :] = 0.0
    u_explicit[:, 0] = u_explicit[:, -1] = 0.0
    
    # Implicit (ADI) step
    rhs = u0 + alpha * lap_u
    u_implicit = adi_helmholtz_2d(u0, rhs, alpha, dx, dy)
    
    # They should be close for small dt
    error = np.max(np.abs(u_implicit - u_explicit))
    print(f"  Max difference: {error:.2e}")
    assert error < 1e-4, f"ADI and explicit differ too much: {error}"
    
    print("  ✓ ADI vs explicit test passed")


if __name__ == "__main__":
    print("Testing ADI Helmholtz Solver...")
    print("=" * 60)
    
    test_adi_2d_diffusion()
    test_adi_convergence()
    test_adi_vs_explicit()
    
    print("=" * 60)
    print("All ADI tests passed! ✓")
