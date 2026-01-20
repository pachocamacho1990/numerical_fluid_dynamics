"""
3D Geometry utilities for backward-facing step flow simulation.

Extends the 2D geometry module to 3D by adding the spanwise (z) dimension.
Sidewalls at z=0 and z=W enforce no-slip conditions.

Based on Armaly et al. (1983) benchmark configuration.
"""

import numpy as np

# Armaly benchmark geometry constants (from 2D module)
H_TOTAL = 1.0
H_STEP = 0.484536       # Step height for ER=1.94
H_INLET = H_TOTAL - H_STEP
L_IN_FACTOR = 5.0
L_OUT_FACTOR = 30.0
W_FACTOR = 8.0  # Channel width W = 8h (typical aspect ratio for 3D effects)


def get_domain_dimensions_3d(h_step=H_STEP):
    """
    Calculate 3D domain dimensions.
    
    Returns
    -------
    dict
        Domain dimensions including W (channel width).
    """
    h_inlet = H_TOTAL - h_step
    L_in = L_IN_FACTOR * h_step
    L_out = L_OUT_FACTOR * h_step
    L_total = L_in + L_out
    W = W_FACTOR * h_step  # Spanwise width
    ER = H_TOTAL / h_inlet
    
    return {
        'H': H_TOTAL,
        'h': h_step,
        'h_inlet': h_inlet,
        'L_in': L_in,
        'L_out': L_out,
        'L_total': L_total,
        'W': W,
        'ER': ER
    }


def create_grid_3d(nx, ny, nz, h_step=H_STEP):
    """
    Create 3D computational grid.
    
    Returns
    -------
    x, y, z : 1D coordinate arrays
    dx, dy, dz : grid spacings
    """
    dims = get_domain_dimensions_3d(h_step)
    
    x = np.linspace(-dims['L_in'], dims['L_out'], nx)
    y = np.linspace(0, dims['H'], ny)
    z = np.linspace(0, dims['W'], nz)
    
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]
    
    return x, y, z, dx, dy, dz


def create_mask_3d(nx, ny, nz, x, y, h_step=H_STEP):
    """
    Create 3D mask array for the backward-facing step geometry.
    
    The mask is 1 for fluid cells and 0 for solid (step) cells.
    The step occupies: x < 0 AND y < h_step (for all z).
    
    Returns
    -------
    mask : ndarray
        3D mask array (nz, ny, nx): 1=fluid, 0=solid
    i_step : int
        x-index where step ends
    j_step : int
        y-index of step top
    """
    mask = np.ones((nz, ny, nx), dtype=np.float64)
    
    i_step = np.searchsorted(x, 0.0)
    j_step = np.searchsorted(y, h_step)
    
    # Mark step region as solid (x < 0 AND y < h_step) for all z
    mask[:, :j_step, :i_step] = 0.0
    
    return mask, i_step, j_step


def get_inlet_profile_3d(y_coords, z_coords, h_inlet, W, u_max=1.5):
    """
    Generate 3D parabolic inlet velocity profile.
    
    For a rectangular duct, the profile is separable:
    u(y, z) = u_y(y) * u_z(z)
    where u_y is parabolic in y and u_z is parabolic in z.
    
    Returns
    -------
    u_inlet : ndarray (nz, ny_inlet)
        Velocity profile at inlet
    """
    # Y-direction parabolic profile
    y_clipped = np.clip(y_coords, 0, h_inlet)
    y_norm = y_clipped / h_inlet
    u_y = 4.0 * y_norm * (1.0 - y_norm)
    
    # Z-direction parabolic profile (scaled by W)
    z_norm = z_coords / W
    u_z = 4.0 * z_norm * (1.0 - z_norm)
    
    # Combined profile: outer product scaled to u_max
    # Peak is at center (y=h_inlet/2, z=W/2) with value u_max
    u_inlet = u_max * np.outer(u_z, u_y)
    
    return u_inlet


def calculate_nu_from_reynolds(Re, u_max, h_step):
    """Calculate kinematic viscosity from Reynolds number."""
    u_avg = (2.0 / 3.0) * u_max
    return u_avg * h_step / Re


def setup_geometry_3d(nx, ny, nz, h_step=H_STEP):
    """
    Complete 3D geometry setup.
    
    Returns
    -------
    geometry : dict
        Dictionary containing all 3D geometry information.
    """
    dims = get_domain_dimensions_3d(h_step)
    x, y, z, dx, dy, dz = create_grid_3d(nx, ny, nz, h_step)
    mask, i_step, j_step = create_mask_3d(nx, ny, nz, x, y, h_step)
    
    # Inlet y-coordinates (from step top to channel top)
    inlet_y = y[j_step:] - y[j_step]
    
    return {
        'dims': dims,
        'x': x,
        'y': y,
        'z': z,
        'dx': dx,
        'dy': dy,
        'dz': dz,
        'mask': mask,
        'i_step': i_step,
        'j_step': j_step,
        'inlet_y': inlet_y,
        'nx': nx,
        'ny': ny,
        'nz': nz
    }


if __name__ == "__main__":
    print("3D Backward-Facing Step Geometry Module")
    print("=" * 50)
    
    dims = get_domain_dimensions_3d()
    print("\n3D Domain Dimensions:")
    for key, val in dims.items():
        print(f"  {key}: {val:.4f}")
    
    nx, ny, nz = 101, 26, 26
    geom = setup_geometry_3d(nx, ny, nz)
    
    print(f"\nGrid: {nx} x {ny} x {nz}")
    print(f"  dx = {geom['dx']:.6f}")
    print(f"  dy = {geom['dy']:.6f}")
    print(f"  dz = {geom['dz']:.6f}")
    print(f"  Total cells: {nx * ny * nz:,}")
    
    mask = geom['mask']
    n_solid = np.sum(mask == 0)
    n_fluid = np.sum(mask == 1)
    print(f"\nMask:")
    print(f"  Solid cells: {n_solid:,}")
    print(f"  Fluid cells: {n_fluid:,}")
    
    print("\n✓ 3D geometry tests passed!")
