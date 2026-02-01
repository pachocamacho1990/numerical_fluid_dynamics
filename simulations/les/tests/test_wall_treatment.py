
import jax.numpy as jnp
from simulations.les.wall_treatment import (
    compute_wall_distance_3d,
    van_driest_damping
)

def test_wall_distance_channel():
    """Test wall distance for channel flow (walls at y=0, y=1)"""
    nx, ny, nz = 5, 11, 5
    dx, dy, dz = 0.1, 0.1, 0.1
    # Ly = (11-1)*0.1 = 1.0
    
    mask = jnp.ones((nz, ny, nx)) # All fluid
    dist = compute_wall_distance_3d(mask, dx, dy, dz)
    
    # y coordinates: 0, 0.1, ... 0.5 ... 1.0
    # Expected dist: 0, 0.1, ... 0.5 ... 0.0
    
    # Check center (y=0.5, index 5)
    assert jnp.isclose(dist[0, 5, 0], 0.5)
    
    # Check near wall (y=0.1, index 1)
    assert jnp.isclose(dist[0, 1, 0], 0.1)
    
    # Check near top wall (y=0.9, index 9) -> dist should be 0.1
    assert jnp.isclose(dist[0, 9, 0], 0.1)

def test_van_driest_damping():
    """Test Van Driest damping function"""
    nu_t = 1.0
    
    # y+ very large -> damping -> 1
    d_far = van_driest_damping(nu_t, y_plus=1000.0)
    assert jnp.isclose(d_far, 1.0, atol=1e-3)
    
    # y+ = 0 -> damping -> 0
    d_wall = van_driest_damping(nu_t, y_plus=0.0)
    assert jnp.isclose(d_wall, 0.0)
    
    # y+ = 25 (A_plus) -> damping should be (1 - exp(-1))^2
    expected = (1.0 - jnp.exp(-1.0))**2
    d_inter = van_driest_damping(nu_t, y_plus=25.0)
    assert jnp.isclose(d_inter, expected)
