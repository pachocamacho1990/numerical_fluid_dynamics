"""
3D Flow Visualization for Backward-Facing Step

Provides multiple visualization options:
1. XY mid-plane contours (z = W/2) - streamwise view
2. YZ cross-section contours - spanwise variation at different x locations
3. (Future) PyVista isosurfaces for 3D visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


import sys
# Look two levels up for geometry_3d (simulations/backward_facing_step_3d/)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from geometry_3d import setup_geometry_3d

def load_solution(filepath="checkpoint_120000.npz"):
    """Load 3D simulation results."""
    data = np.load(filepath)
    
    u = data['u']
    v = data['v']
    w = data['w']
    p = data['p']
    
    if 'x' in data:
        x, y, z = data['x'], data['y'], data['z']
    else:
        print("Reconstructing geometry from shape...")
        nz, ny, nx = u.shape
        geom = setup_geometry_3d(nx, ny, nz)
        x, y, z = geom['x'], geom['y'], geom['z']
        
    mask = data['mask'] if 'mask' in data else geom['mask']

    return {
        'u': u,
        'v': v,
        'w': w,
        'p': p,
        'x': x,
        'y': y,
        'z': z,
        'mask': mask
    }


def plot_xy_midplane(data, output_dir, h_step=0.4845):
    """
    Plot XY mid-plane contours (z = W/2).
    
    Shows streamwise velocity with the recirculation zone visible.
    """
    u, x, y, z = data['u'], data['x'], data['y'], data['z']
    nz = u.shape[0]
    k_mid = nz // 2
    
    u_mid = u[k_mid, :, :]
    
    fig, ax = plt.subplots(figsize=(14, 4))
    
    levels = np.linspace(-0.3, 1.5, 37)
    cf = ax.contourf(x, y, u_mid, levels=levels, cmap='RdBu_r', extend='both')
    ax.contour(x, y, u_mid, levels=[0.0], colors='k', linewidths=1.5)
    
    plt.colorbar(cf, ax=ax, label='u velocity', shrink=0.8)
    
    # Mark step
    ax.fill_between(x[x < 0], 0, h_step, color='gray', alpha=0.8)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'XY Mid-Plane (z = {z[k_mid]:.2f}) - Streamwise Velocity')
    ax.set_aspect('equal')
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0, y.max())
    
    plt.tight_layout()
    outpath = os.path.join(output_dir, 'xy_midplane.png')
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"Saved: {outpath}")
    return outpath


def plot_yz_crosssections(data, output_dir, h_step=0.4845):
    """
    Plot YZ cross-section contours at multiple x locations.
    
    Shows spanwise variation and sidewall boundary layer development.
    """
    u, x, y, z = data['u'], data['x'], data['y'], data['z']
    
    # Find reattachment length for reference
    nz = u.shape[0]
    k_mid = nz // 2
    i_step = np.searchsorted(x, 0.0)
    u_wall = u[k_mid, 1, i_step:]
    x_down = x[i_step:]
    sign_change = np.where(np.diff(np.sign(u_wall)) > 0)[0]
    if len(sign_change) > 0:
        Xr = x_down[sign_change[0]]
    else:
        Xr = 5 * h_step  # fallback
    
    # Cross-section locations
    x_locs = [0.0, Xr/2, Xr, 1.5*Xr]
    x_labels = ['x=0 (step)', f'x=Xr/2={Xr/2:.2f}', f'x=Xr={Xr:.2f}', f'x=1.5Xr={1.5*Xr:.2f}']
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    
    levels = np.linspace(-0.3, 1.2, 31)
    
    for ax, x_loc, label in zip(axes, x_locs, x_labels):
        i_loc = np.argmin(np.abs(x - x_loc))
        u_slice = u[:, :, i_loc].T  # (ny, nz) for contourf(z, y, ...)
        
        cf = ax.contourf(z, y, u_slice, levels=levels, cmap='RdBu_r', extend='both')
        ax.contour(z, y, u_slice, levels=[0.0], colors='k', linewidths=1.0)
        
        # Mark step region if x < 0
        if x_loc < 0:
            ax.axhline(h_step, color='gray', linestyle='--', linewidth=1)
            ax.fill_between(z, 0, h_step, color='gray', alpha=0.3)
        
        ax.set_xlabel('z (spanwise)')
        ax.set_title(label)
        ax.set_aspect('equal')
    
    axes[0].set_ylabel('y')
    
    # Colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    fig.colorbar(cf, cax=cbar_ax, label='u velocity')
    
    fig.suptitle('YZ Cross-Sections: Spanwise Velocity Variation', y=1.02)
    plt.tight_layout()
    
    outpath = os.path.join(output_dir, 'yz_crosssections.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {outpath}")
    return outpath


def plot_combined_view(data, output_dir, h_step=0.4845):
    """
    Combined view: XY midplane + YZ at reattachment.
    """
    u, x, y, z = data['u'], data['x'], data['y'], data['z']
    nz = u.shape[0]
    k_mid = nz // 2
    
    # Find Xr
    i_step = np.searchsorted(x, 0.0)
    u_wall = u[k_mid, 1, i_step:]
    x_down = x[i_step:]
    sign_change = np.where(np.diff(np.sign(u_wall)) > 0)[0]
    Xr = x_down[sign_change[0]] if len(sign_change) > 0 else 5 * h_step
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Top: XY midplane
    ax1 = axes[0]
    u_mid = u[k_mid, :, :]
    levels = np.linspace(-0.3, 1.5, 37)
    cf1 = ax1.contourf(x, y, u_mid, levels=levels, cmap='RdBu_r', extend='both')
    ax1.contour(x, y, u_mid, levels=[0.0], colors='k', linewidths=1.5)
    ax1.fill_between(x[x < 0], 0, h_step, color='gray', alpha=0.8)
    ax1.axvline(Xr, color='lime', linestyle='--', linewidth=2, label=f'Xr = {Xr:.2f}')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title(f'XY Mid-Plane (z = {z[k_mid]:.2f})')
    ax1.set_aspect('equal')
    ax1.legend(loc='upper right')
    plt.colorbar(cf1, ax=ax1, label='u', shrink=0.6)
    
    # Bottom: YZ at Xr
    ax2 = axes[1]
    i_xr = np.argmin(np.abs(x - Xr))
    u_xr = u[:, :, i_xr].T
    cf2 = ax2.contourf(z, y, u_xr, levels=levels, cmap='RdBu_r', extend='both')
    ax2.contour(z, y, u_xr, levels=[0.0], colors='k', linewidths=1.0)
    ax2.set_xlabel('z (spanwise)')
    ax2.set_ylabel('y')
    ax2.set_title(f'YZ Cross-Section at x = Xr = {Xr:.2f}')
    ax2.set_aspect('equal')
    plt.colorbar(cf2, ax=ax2, label='u', shrink=0.6)
    
    plt.tight_layout()
    outpath = os.path.join(output_dir, 'combined_view.png')
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"Saved: {outpath}")
    return outpath


def main():
    parser = argparse.ArgumentParser(description="3D BFS Visualization")
    parser.add_argument("--input", default="simulations/backward_facing_step_3d/outputs/solution_3d.npz",
                        help="Path to solution file")
    parser.add_argument("--output-dir", default="simulations/backward_facing_step_3d/outputs",
                        help="Output directory for plots")
    parser.add_argument("--all", action="store_true", help="Generate all plots")
    parser.add_argument("--xy", action="store_true", help="Generate XY midplane plot")
    parser.add_argument("--yz", action="store_true", help="Generate YZ cross-sections")
    parser.add_argument("--combined", action="store_true", help="Generate combined view")
    args = parser.parse_args()
    
    print("=" * 60)
    print("3D Backward-Facing Step Visualization")
    print("=" * 60)
    
    # Load data
    print(f"Loading: {args.input}")
    data = load_solution(args.input)
    print(f"Grid: {data['u'].shape[2]} x {data['u'].shape[1]} x {data['u'].shape[0]} (nx, ny, nz)")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Default to all if nothing specified
    if not (args.xy or args.yz or args.combined or args.all):
        args.all = True
    
    if args.xy or args.all:
        plot_xy_midplane(data, args.output_dir)
    
    if args.yz or args.all:
        plot_yz_crosssections(data, args.output_dir)
    
    if args.combined or args.all:
        plot_combined_view(data, args.output_dir)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
