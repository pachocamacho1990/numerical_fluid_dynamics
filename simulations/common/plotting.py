
import matplotlib.pyplot as plt
import numpy as np
import os
from simulations.common.solvers import SimulationResult, SimulationConfig

def plot_velocity_magnitude(result: SimulationResult, config: SimulationConfig, output_path: str):
    """
    Plots the velocity magnitude and streamlines for a single result.
    Annotates with Re, t, dt, grid size.
    """
    nx = config.nx
    ny = config.nx
    L = 2.0
    x = np.linspace(0, L, nx)
    y = np.linspace(0, L, ny)
    X, Y = np.meshgrid(x, y)
    
    # Compute magnitude
    vel_mag = np.sqrt(result.u**2 + result.v**2)
    
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    
    # Contours
    levels = np.linspace(0, np.max(vel_mag), 25)
    cf = ax.contourf(X, Y, vel_mag, levels=levels, cmap='viridis')
    plt.colorbar(cf, ax=ax, label='Velocity Magnitude |u|')
    
    # Streamlines
    # Use lighter color for streamlines on top of viridis
    ax.streamplot(X, Y, result.u, result.v, color='white', linewidth=0.6, density=1.5, arrowsize=0.8)
    
    # Annotations
    title = f"Cavity Flow ({result.meta.get('backend', 'Unknown')})"
    subtitle = (f"Re={config.re} | Grid: {nx}x{ny}\n"
                f"t={result.t_end:.4f} | dt={config.dt:.5f} | steps={result.steps}")
    
    ax.set_title(title, fontsize=14, loc='left')
    ax.set_title(subtitle, fontsize=10, loc='right', color='gray')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    # Ensure directory exists if path is nested
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.close()

def plot_comparison_centerline(results: list[SimulationResult], labels: list[str], config: SimulationConfig, output_path: str):
    """
    Plots u-velocity along the vertical centerline (x=L/2) for multiple results.
    Useful for Ghia benchmark comparison.
    """
    nx = config.nx
    L = 2.0
    y = np.linspace(0, L, nx)
    center_idx = int(nx / 2)
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    
    for res, label in zip(results, labels):
        # Extract vertical line at middle of x
        u_center = res.u[:, center_idx]
        ax.plot(y, u_center, label=label, linewidth=2)
        
    ax.set_xlabel("y (Height)")
    ax.set_ylabel("u (Horizontal Velocity)")
    ax.set_title(f"Centerline Velocity Comparison (Re={config.re})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Ghia et al. validation range helper (approximate visual guide)
    ax.axvline(x=0.0, color='k', linestyle='-', alpha=0.1)
    ax.axvline(x=2.0, color='k', linestyle='-', alpha=0.1)
    
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path)
    print(f"Comparison plot saved to {output_path}")
    plt.close()
