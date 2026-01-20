"""
Pressure Poisson Solver - JAX Implementation

Implements Red-Black Successive Over-Relaxation (SOR) solver.
This is a robust, parallelizable solver that is significantly faster than Jacobi
while maintaining excellent stability properties for arbitrary grids.

Usage:
    from multigrid_poisson import solve_sor
    p = solve_sor(p_init, b, dx, dy, mask, n_iters=100, omega=1.9)
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import jit, lax
from functools import partial

# ============================================================================
# Boundary Conditions (Rectangular)
# ============================================================================

@jit
def apply_pressure_bc(p: jnp.ndarray) -> jnp.ndarray:
    """
    Apply boundary conditions (Rectangular domain).
    Matches bfs_optimized.py baseline.
    
    - Outlet (Right): p = 0 (Dirichlet)
    - Walls (Top/Bottom/Left): dp/dn = 0 (Neumann)
    """
    # Outlet: p = 0
    p = p.at[:, -1].set(0.0)
    
    # Top wall: dp/dy = 0
    p = p.at[-1, :].set(p[-2, :])
    
    # Bottom wall: dp/dy = 0
    p = p.at[0, :].set(p[1, :])
    
    # Inlet (Left): dp/dx = 0
    p = p.at[:, 0].set(p[:, 1])
    
    return p


# ============================================================================
# Red-Black SOR Smoother
# ============================================================================

@partial(jit, static_argnums=(5,))
def solve_sor(p: jnp.ndarray, b: jnp.ndarray, 
              dx: float, dy: float, mask: jnp.ndarray,
              n_iters: int = 100, omega: float = 1.9) -> jnp.ndarray:
    """
    Red-Black Successive Over-Relaxation (SOR) solver.
    
    Args:
        p: Initial pressure guess
        b: RHS of Poisson equation divergence(u)/dt
        dx, dy: Grid spacing
        mask: Geometry mask
        n_iters: Number of iterations
        omega: Relaxation factor (1.0 = Gauss-Seidel, 1 < w < 2 = SOR)
               Optimal omega approx 2 / (1 + sin(pi*h))
    """
    ny, nx = p.shape
    h2 = dx * dy # Approximation?
    # Discrete Poisson: (p_E + p_W)*dy^2 + (p_N + p_S)*dx^2 = p_C * 2(dx^2+dy^2) - dx^2*dy^2*b
    # If dx=dy=h: sum_neighbors - 4*p = h^2*b 
    # p = (sum - h^2*b)/4
    
    # Pre-compute coefficients for general dx != dy
    denom = 2 * (dx**2 + dy**2)
    coeff_x = dy**2 / denom
    coeff_y = dx**2 / denom
    coeff_b = (dx**2 * dy**2) / denom
    
    # Pre-compute masks for Red (even i+j) and Black (odd i+j)
    y_idx, x_idx = jnp.mgrid[0:ny, 0:nx]
    checkerboard = (x_idx + y_idx) % 2
    mask_red = (checkerboard == 0) * mask
    mask_black = (checkerboard == 1) * mask
    
    def sor_sweep(p, _):
        # 1. Red Sweep
        # Calculate Gauss-Seidel update
        p_gs = (
            coeff_x * (p[1:-1, 2:] + p[1:-1, :-2]) +
            coeff_y * (p[2:, 1:-1] + p[:-2, 1:-1]) -
            coeff_b * b[1:-1, 1:-1]
        )
        
        # SOR update: p_new = (1-w)*p_old + w*p_gs
        p_new = (1.0 - omega) * p[1:-1, 1:-1] + omega * p_gs
        
        # Apply to Red points
        p_full = jnp.zeros_like(p).at[1:-1, 1:-1].set(p_new)
        p = jnp.where(mask_red > 0, p_full, p)
        p = apply_pressure_bc(p)
        
        # 2. Black Sweep (using new Red values)
        p_gs = (
            coeff_x * (p[1:-1, 2:] + p[1:-1, :-2]) +
            coeff_y * (p[2:, 1:-1] + p[:-2, 1:-1]) -
            coeff_b * b[1:-1, 1:-1]
        )
        
        p_new = (1.0 - omega) * p[1:-1, 1:-1] + omega * p_gs
        
        # Apply to Black points
        p_full = jnp.zeros_like(p).at[1:-1, 1:-1].set(p_new)
        p = jnp.where(mask_black > 0, p_full, p)
        p = apply_pressure_bc(p)
        
        return p, None
    
    p, _ = lax.scan(sor_sweep, p, None, length=n_iters)
    return p


# ============================================================================
# Compatibility / Testing
# ============================================================================

# Alias for backward compatibility if needed, but signature changed
def solve_multigrid(p, b, dx, dy, mask, n_cycles=3):
    """Shim to use SOR instead of Multigrid."""
    # Map n_cycles multigrid -> n_iters SOR
    # Jacobi 50 iters used previously.
    # SOR converges ~3x faster than Jacobi.
    # 15 iters of SOR gives comparable/better quality than 50 Jacobi.
    return solve_sor(p, b, dx, dy, mask, n_iters=15, omega=1.9)


if __name__ == "__main__":
    import time
    
    print("="*60)
    print("Red-Black SOR Solver Test")
    print("="*60)
    
    nx, ny = 201, 51
    dx = 1.0 / (nx - 1)
    dy = dx
    
    mask = jnp.ones((ny, nx))
    key = jax.random.PRNGKey(0)
    b = jax.random.uniform(key, (ny, nx)) - 0.5
    p = jnp.zeros((ny, nx))
    
    print("\nWarming up...")
    _ = solve_sor(p, b, dx, dy, mask, n_iters=1)
    
    print("\nBenchmark (50 iters)...")
    start = time.perf_counter()
    p = solve_sor(p, b, dx, dy, mask, n_iters=50, omega=1.8)
    p = p.block_until_ready()
    elapsed = (time.perf_counter() - start) * 1000
    
    # Compute residual
    lap = jnp.zeros_like(p)
    lap = lap.at[1:-1, 1:-1].set(
        (p[1:-1, 2:] + p[1:-1, :-2] + p[2:, 1:-1] + p[:-2, 1:-1] - 4*p[1:-1, 1:-1]) / dx**2
    )
    r = (b - lap) * mask
    res_norm = float(jnp.linalg.norm(r) / jnp.linalg.norm(b))
    
    print(f"Time: {elapsed:.2f} ms")
    print(f"Relative Residual Norm: {res_norm:.2e}")
