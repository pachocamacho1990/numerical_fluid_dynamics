"""
JAX-optimized Thomas Algorithm for solving tridiagonal systems.

Uses JAX's lax.scan for efficient sequential operations that can be JIT-compiled.
"""

import jax.numpy as jnp
from jax import jit
import jax.lax as lax


@jit
def thomas_algorithm_jax(a, b, c, d):
    """
    Solve a tridiagonal system using the Thomas algorithm (JAX version).
    
    The system is of the form:
        a[i] * x[i-1] + b[i] * x[i] + c[i] * x[i+1] = d[i]
    
    Args:
        a: Lower diagonal coefficients (length n)
        b: Main diagonal coefficients (length n)
        c: Upper diagonal coefficients (length n)
        d: Right-hand side vector (length n)
    
    Returns:
        x: Solution vector (length n)
    
    Notes:
        - Uses lax.scan for efficient sequential computation
        - Fully JIT-compilable for GPU acceleration
    """
    n = len(d)
    
    # Forward elimination using scan
    def forward_step(carry, i):
        b_mod, d_mod = carry
        
        # Compute modification factor
        factor = a[i] / b_mod[i-1]
        
        # Update diagonal and RHS
        b_new = b[i] - factor * c[i-1]
        d_new = d[i] - factor * d_mod[i-1]
        
        # Update arrays
        b_mod = b_mod.at[i].set(b_new)
        d_mod = d_mod.at[i].set(d_new)
        
        return (b_mod, d_mod), None
    
    # Initialize
    b_mod = b.copy()
    d_mod = d.copy()
    
    # Run forward elimination
    (b_mod, d_mod), _ = lax.scan(forward_step, (b_mod, d_mod), jnp.arange(1, n))
    
    # Back substitution using scan (reverse order)
    def backward_step(carry, i):
        x = carry
        
        # Solve for x[i]
        x_i = (d_mod[i] - c[i] * x[i+1]) / b_mod[i]
        x = x.at[i].set(x_i)
        
        return x, None
    
    # Initialize x with last element
    x = jnp.zeros(n)
    x = x.at[-1].set(d_mod[-1] / b_mod[-1])
    
    # Run back substitution
    x, _ = lax.scan(backward_step, x, jnp.arange(n-2, -1, -1))
    
    return x


def build_tridiagonal_diffusion_jax(n, r, bc_type='dirichlet'):
    """
    Build tridiagonal system for 1D implicit diffusion (JAX version).
    
    Solves: (I - r * d²/dx²) u = rhs
    
    Args:
        n: Number of grid points
        r: Diffusion number (nu * dt / dx²)
        bc_type: 'dirichlet' or 'neumann'
    
    Returns:
        a, b, c: Tridiagonal matrix coefficients
    """
    a = jnp.full(n, -r)
    b = jnp.full(n, 1.0 + 2.0 * r)
    c = jnp.full(n, -r)
    
    # Boundary conditions
    if bc_type == 'dirichlet':
        b = b.at[0].set(1.0)
        c = c.at[0].set(0.0)
        a = a.at[-1].set(0.0)
        b = b.at[-1].set(1.0)
    elif bc_type == 'neumann':
        b = b.at[0].set(1.0)
        c = c.at[0].set(-1.0)
        a = a.at[-1].set(-1.0)
        b = b.at[-1].set(1.0)
    
    return a, b, c


# Test function
if __name__ == "__main__":
    import numpy as np
    from thomas_solver import thomas_algorithm as thomas_numpy
    
    print("Testing JAX Thomas Algorithm...")
    print("-" * 50)
    
    # Test 1: Simple system
    n = 100
    a = jnp.array(np.random.rand(n))
    b = jnp.array(np.random.rand(n) + 2.0)  # Ensure diagonal dominance
    c = jnp.array(np.random.rand(n))
    d = jnp.array(np.random.rand(n))
    
    # Solve with JAX
    x_jax = thomas_algorithm_jax(a, b, c, d)
    
    # Solve with NumPy
    x_numpy = thomas_numpy(np.array(a), np.array(b), np.array(c), np.array(d))
    
    # Compare
    diff = np.max(np.abs(np.array(x_jax) - x_numpy))
    print(f"Max difference (JAX vs NumPy): {diff:.2e}")
    assert diff < 1e-6, f"Too large difference: {diff}"
    
    print("✓ JAX Thomas algorithm matches NumPy")
    print("-" * 50)
    print("All tests passed! ✓")
