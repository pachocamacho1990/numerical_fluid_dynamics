
import jax
import jax.numpy as jnp
import numpy as np
from simulations.les.variable_viscosity_adi import (
    build_tridiagonal_variable_nu,
    thomas_algorithm_variable
)

def test_build_tridiagonal_constant_viscosity():
    """Test that variable viscosity coefficients match constant viscosity coefficients"""
    nz, ny, nx = 5, 5, 5
    dt = 0.1
    dx = 0.1
    nu_val = 0.01
    nu_field = jnp.full((nz, ny, nx), nu_val)
    
    # Axis 2 (x)
    a, b, c = build_tridiagonal_variable_nu(nu_field, dt, dx, axis=2)
    
    alpha = dt / dx**2
    # For constant nu:
    # a = -alpha * nu
    # b = 1 + 2 * alpha * nu
    # c = -alpha * nu
    
    expected_a = -alpha * nu_val
    expected_c = -alpha * nu_val
    expected_b = 1.0 + 2.0 * alpha * nu_val
    
    # Check internal points (avoid boundary logic diffs)
    assert jnp.allclose(a[..., 1:-1], expected_a)
    assert jnp.allclose(c[..., 1:-1], expected_c)
    assert jnp.allclose(b[..., 1:-1], expected_b)

def test_thomas_algorithm_variable_identity():
    """Test solving Identity system: b=1, a=0, c=0 => x=d"""
    nz, ny, nx = 4, 4, 4
    N = nx
    
    a = jnp.zeros((nz, ny, N))
    b = jnp.ones((nz, ny, N))
    c = jnp.zeros((nz, ny, N))
    d = jnp.ones((nz, ny, N)) * 5.0
    
    x = thomas_algorithm_variable(a, b, c, d)
    
    assert jnp.allclose(x, 5.0)

def test_thomas_algorithm_variable_simple_1d():
    """Test solving simple 1D diffusion-like system"""
    # -u_{i-1} + 2u_i - u_{i+1} = 0
    # BC: u_0 = 0, u_N = 1  (Fixed via boundary modification or RHS)
    # Actually simpler:
    # 2*u_i = d_i (diagonal matrix)
    
    nz, ny, nx = 2, 2, 4
    a = jnp.zeros((nz, ny, nx))
    b = jnp.full((nz, ny, nx), 2.0)
    c = jnp.zeros((nz, ny, nx))
    d = jnp.full((nz, ny, nx), 4.0)
    
    x = thomas_algorithm_variable(a, b, c, d)
    
    assert jnp.allclose(x, 2.0)

