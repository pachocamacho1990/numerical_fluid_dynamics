"""
Validate 3D solver by measuring reattachment length and comparing with Armaly data.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def find_reattachment_length_3d(u, x, mask, j_step, h_step):
    """
    Find reattachment length from 3D velocity field.
    
    For 3D, we can look at the mid-plane (z = W/2) or average over z.
    Reattachment is where u changes sign from negative to positive along bottom wall.
    """
    nz, ny, nx = u.shape
    
    # Take mid-plane slice (z = nz//2)
    k_mid = nz // 2
    u_mid = u[k_mid, :, :]  # (ny, nx)
    
    # Look at velocity just above the bottom wall in the downstream region
    # j=1 is just above y=0, starting from i_step (x=0)
    i_step = np.searchsorted(x, 0.0)
    
    u_wall = u_mid[1, i_step:]  # Velocity just above bottom wall, downstream of step
    x_downstream = x[i_step:]
    
    # Find reattachment: where u goes from negative to positive
    sign_change = np.where(np.diff(np.sign(u_wall)) > 0)[0]
    
    if len(sign_change) > 0:
        # Interpolate to find exact crossing
        i_cross = sign_change[0]
        x_r = x_downstream[i_cross] - u_wall[i_cross] * (x_downstream[i_cross+1] - x_downstream[i_cross]) / (u_wall[i_cross+1] - u_wall[i_cross])
        return x_r / h_step
    else:
        return np.nan


def main():
    # Load solution
    data = np.load("simulations/backward_facing_step_3d/outputs/solution_3d.npz")
    u = data['u']
    x = data['x']
    y = data['y']
    z = data['z']
    mask = data['mask']
    
    # Geometry
    h_step = 0.484536
    j_step = np.searchsorted(y, h_step)
    
    print("3D Reattachment Length Analysis")
    print("=" * 40)
    
    nz, ny, nx = u.shape
    print(f"Grid: {nx} x {ny} x {nz}")
    
    # Find Xr/h
    xr_h = find_reattachment_length_3d(u, x, mask, j_step, h_step)
    print(f"Xr/h (3D mid-plane): {xr_h:.2f}")
    print(f"Armaly Exp (Re=400): ~8.2")
    
    # Also check 2D result for comparison (average over z)
    u_avg = np.mean(u, axis=0)  # Average over z -> (ny, nx)
    
    # For z-averaged field
    i_step = np.searchsorted(x, 0.0)
    u_wall_avg = u_avg[1, i_step:]
    x_downstream = x[i_step:]
    
    sign_change = np.where(np.diff(np.sign(u_wall_avg)) > 0)[0]
    if len(sign_change) > 0:
        i_cross = sign_change[0]
        x_r_avg = x_downstream[i_cross] - u_wall_avg[i_cross] * (x_downstream[i_cross+1] - x_downstream[i_cross]) / (u_wall_avg[i_cross+1] - u_wall_avg[i_cross])
        xr_h_avg = x_r_avg / h_step
        print(f"Xr/h (z-averaged): {xr_h_avg:.2f}")
    
    # Plot mid-plane velocity
    k_mid = nz // 2
    u_mid = u[k_mid, :, :]
    
    fig, ax = plt.subplots(figsize=(12, 4))
    levels = np.linspace(-0.3, 1.5, 19)
    cf = ax.contourf(x, y, u_mid, levels=levels, cmap='RdBu_r', extend='both')
    plt.colorbar(cf, ax=ax, label='u velocity')
    ax.set_xlabel('x/h')
    ax.set_ylabel('y/h')
    ax.set_title(f'3D BFS Re=400 Mid-plane (z=W/2), Xr/h = {xr_h:.2f}')
    ax.set_aspect('equal')
    
    # Mark reattachment
    if not np.isnan(xr_h):
        ax.axvline(xr_h * h_step, color='k', linestyle='--', label=f'Xr = {xr_h:.2f}h')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig("simulations/backward_facing_step_3d/outputs/validation_3d.png", dpi=150)
    print("Saved plot to simulations/backward_facing_step_3d/outputs/validation_3d.png")


if __name__ == "__main__":
    main()
