"""
Visualization utilities for backward-facing step flow.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

def plot_results(u, v, p, geom, output_dir, Re, backend=""):
    """
    Generate visualization of flow field results.
    
    Parameters
    ----------
    u, v, p : ndarray
        Velocity and pressure fields
    geom : dict
        Geometry dictionary
    output_dir : str
        Directory to save plot
    Re : float
        Reynolds number
    backend : str, optional
        Backend name (e.g. "numpy", "jax_cpu")
    """
    X, Y = geom['X'], geom['Y']
    mask = geom['mask']
    dims = geom['dims']
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Plot 1: Pressure contours and streamlines
    ax1 = axes[0]
    p_masked = np.ma.masked_where(mask == 0, p)
    cf = ax1.contourf(X, Y, p_masked, levels=30, cmap='RdBu_r', alpha=0.8)
    plt.colorbar(cf, ax=ax1, label='Pressure')
    
    u_masked = np.where(mask == 1, u, np.nan)
    v_masked = np.where(mask == 1, v, np.nan)
    ax1.streamplot(X, Y, u_masked, v_masked, color='black', linewidth=0.5, density=2)
    
    # Draw step outline
    ax1.fill_between([-dims['L_in'], 0], 0, dims['h'], color='gray', alpha=0.5)
    
    # Construct title
    title = f'Backward-Facing Step Flow (Re={Re})'
    if backend:
        title += f' - {backend.upper()}'
    ax1.set_title(title)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_aspect('equal')
    
    # Plot 2: Velocity magnitude
    ax2 = axes[1]
    vel_mag = np.sqrt(u**2 + v**2)
    vel_masked = np.ma.masked_where(mask == 0, vel_mag)
    cf2 = ax2.contourf(X, Y, vel_masked, levels=30, cmap='viridis')
    plt.colorbar(cf2, ax=ax2, label='|u|')
    ax2.fill_between([-dims['L_in'], 0], 0, dims['h'], color='gray', alpha=0.5)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Velocity Magnitude')
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    
    # Construct filename
    filename = f'backward_step_re{int(Re)}'
    if backend:
        filename += f'_{backend}'
    filename += '.png'
        
    output_path = os.path.join(output_dir, 'plots', filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")
    plt.close()
