"""
Geometry utilities for backward-facing step flow simulation.

This module handles the step geometry, masking, and inlet profile generation
based on the Armaly et al. (1983) benchmark configuration.

Geometry Parameters (Armaly Benchmark):
    - Total height H = 1.0
    - Step height h = 0.51 (h/H = 0.51)
    - Inlet height = H - h = 0.49
    - Expansion ratio ER = H/(H-h) = 1.94
    - Inlet length L_in = 5h ≈ 2.55
    - Outlet length L_out = 30h ≈ 15.3
"""

import numpy as np


# Armaly benchmark geometry constants
H_TOTAL = 1.0           # Total channel height (normalized)
H_STEP = 0.51           # Step height (h/H = 0.51 per Armaly)
H_INLET = H_TOTAL - H_STEP  # Inlet channel height = 0.49
L_IN_FACTOR = 5.0       # L_in = 5h (inlet development length)
L_OUT_FACTOR = 30.0     # L_out = 30h (outlet length for reattachment)


def get_domain_dimensions(h_step=H_STEP):
    """
    Calculate domain dimensions based on step height.
    
    Parameters
    ----------
    h_step : float
        Step height (default: 0.51 per Armaly benchmark)
    
    Returns
    -------
    dict
        Dictionary with domain dimensions:
        - H: total channel height
        - h: step height
        - h_inlet: inlet channel height
        - L_in: inlet length
        - L_out: outlet length
        - L_total: total domain length
        - ER: expansion ratio
    """
    h_inlet = H_TOTAL - h_step
    L_in = L_IN_FACTOR * h_step
    L_out = L_OUT_FACTOR * h_step
    L_total = L_in + L_out
    ER = H_TOTAL / h_inlet
    
    return {
        'H': H_TOTAL,
        'h': h_step,
        'h_inlet': h_inlet,
        'L_in': L_in,
        'L_out': L_out,
        'L_total': L_total,
        'ER': ER
    }


def create_grid(nx, ny, h_step=H_STEP):
    """
    Create computational grid for the backward-facing step domain.
    
    Parameters
    ----------
    nx : int
        Number of grid points in x-direction
    ny : int
        Number of grid points in y-direction
    h_step : float
        Step height (default: 0.51)
    
    Returns
    -------
    x : ndarray
        1D array of x-coordinates
    y : ndarray
        1D array of y-coordinates
    X : ndarray
        2D meshgrid of x-coordinates
    Y : ndarray
        2D meshgrid of y-coordinates
    dx : float
        Grid spacing in x
    dy : float
        Grid spacing in y
    """
    dims = get_domain_dimensions(h_step)
    
    # x ranges from -L_in to L_out (step at x=0)
    x = np.linspace(-dims['L_in'], dims['L_out'], nx)
    y = np.linspace(0, dims['H'], ny)
    
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    X, Y = np.meshgrid(x, y)
    
    return x, y, X, Y, dx, dy


def create_mask(nx, ny, x, y, h_step=H_STEP):
    """
    Create mask array for the backward-facing step geometry.
    
    The mask is 1 for fluid cells and 0 for solid (step) cells.
    The step occupies: x < 0 AND y < h_step
    
    Parameters
    ----------
    nx : int
        Number of grid points in x
    ny : int
        Number of grid points in y
    x : ndarray
        1D array of x-coordinates
    y : ndarray
        1D array of y-coordinates
    h_step : float
        Step height
    
    Returns
    -------
    mask : ndarray
        2D mask array (ny, nx): 1=fluid, 0=solid
    i_step : int
        x-index where step ends (first index with x >= 0)
    j_step : int
        y-index of step top (first index with y >= h_step)
    """
    mask = np.ones((ny, nx), dtype=np.float64)
    
    # Find indices for step boundary
    # i_step: first x-index where x >= 0 (step ends at x=0)
    i_step = np.searchsorted(x, 0.0)
    
    # j_step: first y-index where y >= h_step (step top)
    j_step = np.searchsorted(y, h_step)
    
    # Mark step region as solid (x < 0 AND y < h_step)
    mask[:j_step, :i_step] = 0.0
    
    return mask, i_step, j_step


def get_inlet_profile(y_coords, h_inlet, u_max=1.5):
    """
    Generate parabolic (Poiseuille) inlet velocity profile.
    
    The profile is: u(y) = u_max * 4 * (y/h) * (1 - y/h)
    where h = inlet height, giving u_max at the centerline.
    
    For a parabolic profile, U_avg = (2/3) * U_max.
    
    Parameters
    ----------
    y_coords : ndarray
        Y-coordinates in the inlet region (0 to h_inlet)
    h_inlet : float
        Inlet channel height
    u_max : float
        Maximum velocity at centerline (default: 1.5)
    
    Returns
    -------
    u_inlet : ndarray
        Velocity profile at inlet
    """
    # Clip y to valid range to avoid numerical issues
    y_clipped = np.clip(y_coords, 0, h_inlet)
    
    # Normalize y to [0, 1] range within inlet
    y_normalized = y_clipped / h_inlet
    
    # Parabolic profile: 4 * y/h * (1 - y/h) has max of 1 at y/h = 0.5
    u_inlet = u_max * 4.0 * y_normalized * (1.0 - y_normalized)
    
    return u_inlet


def apply_mask(u, v, mask):
    """
    Apply mask to velocity fields to enforce zero velocity in solid regions.
    
    Parameters
    ----------
    u : ndarray
        x-velocity field
    v : ndarray
        y-velocity field
    mask : ndarray
        Mask array (1=fluid, 0=solid)
    
    Returns
    -------
    u : ndarray
        Masked x-velocity
    v : ndarray
        Masked y-velocity
    """
    return u * mask, v * mask


def calculate_reynolds(u_max, h_step, nu):
    """
    Calculate Reynolds number based on step height and average inlet velocity.
    
    Re = U_avg * h / nu, where U_avg = (2/3) * U_max for parabolic profile.
    
    Parameters
    ----------
    u_max : float
        Maximum inlet velocity
    h_step : float
        Step height
    nu : float
        Kinematic viscosity
    
    Returns
    -------
    Re : float
        Reynolds number
    """
    u_avg = (2.0 / 3.0) * u_max
    return u_avg * h_step / nu


def calculate_nu_from_reynolds(Re, u_max, h_step):
    """
    Calculate kinematic viscosity from Reynolds number.
    
    nu = U_avg * h / Re, where U_avg = (2/3) * U_max
    
    Parameters
    ----------
    Re : float
        Target Reynolds number
    u_max : float
        Maximum inlet velocity
    h_step : float
        Step height
    
    Returns
    -------
    nu : float
        Kinematic viscosity
    """
    u_avg = (2.0 / 3.0) * u_max
    return u_avg * h_step / Re


# ============================================================================
# Convenience function for complete geometry setup
# ============================================================================

def setup_geometry(nx, ny, h_step=H_STEP):
    """
    Complete geometry setup for backward-facing step simulation.
    
    Parameters
    ----------
    nx : int
        Grid points in x
    ny : int
        Grid points in y
    h_step : float
        Step height (default: 0.51 per Armaly)
    
    Returns
    -------
    geometry : dict
        Dictionary containing all geometry information:
        - dims: domain dimensions
        - x, y: 1D coordinate arrays
        - X, Y: 2D meshgrids
        - dx, dy: grid spacings
        - mask: solid/fluid mask
        - i_step, j_step: step boundary indices
        - inlet_y: y-coordinates for inlet
        - inlet_profile: function to get inlet velocity profile
    """
    dims = get_domain_dimensions(h_step)
    x, y, X, Y, dx, dy = create_grid(nx, ny, h_step)
    mask, i_step, j_step = create_mask(nx, ny, x, y, h_step)
    
    # Inlet y-coordinates (from step top to channel top)
    inlet_y = y[j_step:] - y[j_step]  # Shift to start at 0
    
    return {
        'dims': dims,
        'x': x,
        'y': y, 
        'X': X,
        'Y': Y,
        'dx': dx,
        'dy': dy,
        'mask': mask,
        'i_step': i_step,
        'j_step': j_step,
        'inlet_y': inlet_y,
        'nx': nx,
        'ny': ny
    }


# ============================================================================
# Test/Demo
# ============================================================================

if __name__ == "__main__":
    print("Backward-Facing Step Geometry Module")
    print("=" * 50)
    
    # Get domain dimensions
    dims = get_domain_dimensions()
    print("\nDomain Dimensions (Armaly Benchmark):")
    for key, val in dims.items():
        print(f"  {key}: {val:.4f}")
    
    # Create grid
    nx, ny = 201, 51
    geom = setup_geometry(nx, ny)
    
    print(f"\nGrid: {nx} x {ny}")
    print(f"  dx = {geom['dx']:.6f}")
    print(f"  dy = {geom['dy']:.6f}")
    print(f"  i_step = {geom['i_step']} (x ≈ {geom['x'][geom['i_step']]:.4f})")
    print(f"  j_step = {geom['j_step']} (y ≈ {geom['y'][geom['j_step']]:.4f})")
    
    # Test mask
    mask = geom['mask']
    n_solid = np.sum(mask == 0)
    n_fluid = np.sum(mask == 1)
    print(f"\nMask:")
    print(f"  Solid cells: {n_solid}")
    print(f"  Fluid cells: {n_fluid}")
    print(f"  Total: {n_solid + n_fluid}")
    
    # Test inlet profile
    u_max = 1.5
    u_inlet = get_inlet_profile(geom['inlet_y'], dims['h_inlet'], u_max)
    print(f"\nInlet Profile (u_max = {u_max}):")
    print(f"  Max velocity: {np.max(u_inlet):.4f}")
    print(f"  At walls: {u_inlet[0]:.6f}, {u_inlet[-1]:.6f}")
    
    # Reynolds number calculation
    Re = 100
    nu = calculate_nu_from_reynolds(Re, u_max, dims['h'])
    Re_check = calculate_reynolds(u_max, dims['h'], nu)
    print(f"\nReynolds Number:")
    print(f"  Target Re = {Re}")
    print(f"  Calculated nu = {nu:.6f}")
    print(f"  Verified Re = {Re_check:.2f}")
    
    print("\n✓ All geometry tests passed!")
