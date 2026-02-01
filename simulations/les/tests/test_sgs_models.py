
import jax.numpy as jnp
import numpy as np
from simulations.les.sgs_models import (
    compute_strain_rate_tensor_3d,
    compute_strain_magnitude,
    compute_smagorinsky_viscosity,
    compute_filter_width
)

def test_strain_rate_tensor_simple_shear():
    """Test strain rate calculation for simple shear flow: u = y, v = 0, w = 0"""
    nx, ny, nz = 10, 10, 10
    dx, dy, dz = 0.1, 0.1, 0.1
    
    # Grid
    y = jnp.linspace(0, (ny-1)*dy, ny)
    # Broadcast to 3D: (nz, ny, nx)
    Y = jnp.broadcast_to(y[None, :, None], (nz, ny, nx))
    
    u = Y  # u = y
    v = jnp.zeros((nz, ny, nx))
    w = jnp.zeros((nz, ny, nx))
    
    S11, S22, S33, S12, S13, S23 = compute_strain_rate_tensor_3d(u, v, w, dx, dy, dz)
    
    # Expected: 
    # du/dy = 1.0
    # S12 = 0.5 * (du/dy + dv/dx) = 0.5 * (1.0 + 0) = 0.5
    # All others should be 0
    
    # Check internal points to avoid boundary artifacts
    assert jnp.allclose(S11[1:-1, 1:-1, 1:-1], 0.0)
    assert jnp.allclose(S22[1:-1, 1:-1, 1:-1], 0.0)
    assert jnp.allclose(S33[1:-1, 1:-1, 1:-1], 0.0)
    assert jnp.allclose(S12[1:-1, 1:-1, 1:-1], 0.5)
    assert jnp.allclose(S13[1:-1, 1:-1, 1:-1], 0.0)
    assert jnp.allclose(S23[1:-1, 1:-1, 1:-1], 0.0)

def test_strain_magnitude_simple_shear():
    """Test strain magnitude for S12 = 0.5"""
    nz, ny, nx = 5, 5, 5
    S12 = jnp.full((nz, ny, nx), 0.5)
    S11 = S22 = S33 = S13 = S23 = jnp.zeros_like(S12)
    
    S_mag = compute_strain_magnitude(S11, S22, S33, S12, S13, S23)
    
    # |S| = sqrt(2 * (2 * S12^2)) = sqrt(4 * 0.25) = sqrt(1) = 1.0
    assert jnp.allclose(S_mag, 1.0)

def test_smagorinsky_viscosity_zero_velocity():
    """Test that zero velocity results in zero eddy viscosity"""
    nx, ny, nz = 10, 10, 10
    dx, dy, dz = 0.1, 0.1, 0.1
    u = jnp.zeros((nz, ny, nx))
    v = jnp.zeros((nz, ny, nx))
    w = jnp.zeros((nz, ny, nx))
    
    S_comps = compute_strain_rate_tensor_3d(u, v, w, dx, dy, dz)
    nu_t = compute_smagorinsky_viscosity(*S_comps, cs=0.1, delta=0.1)
    
    assert jnp.allclose(nu_t, 0.0)

def test_filter_width():
    dx, dy, dz = 0.1, 0.2, 0.4
    delta = compute_filter_width(dx, dy, dz)
    expected = (0.1 * 0.2 * 0.4)**(1/3)
    assert jnp.isclose(delta, expected)
