"""
Reattachment Length Validation for Implicit Backward-Facing Step Solver

This script runs a series of simulations at different Reynolds numbers (100, 200, 300, 400)
and calculates the primary reattachment length (Xr).
It then plots the results against experimental data from Armaly et al. (1983).

Usage:
    JAX_PLATFORMS=cpu venv/bin/python experiments/validate_reattachment.py
"""

import sys
import os
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add solver directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
# root is up two levels: simulations/experiments -> simulations -> root
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(os.path.join(root_dir, "simulations", "backward_facing_step_implicit"))
sys.path.append(os.path.join(root_dir, "simulations", "backward_facing_step"))

import backward_facing_step_implicit as solver
from geometry import setup_geometry, get_inlet_profile, calculate_nu_from_reynolds

def find_reattachment_length(u, x, y, j_wall=0):
    """
    Finds Xr where wall shear stress changes sign (u becomes positive) on bottom wall.
    Scans along the first fluid node line near the bottom wall.
    """
    # Scan u at the first y-index above the wall (j=1)
    # The wall is at j=0.
    u_near_wall = u[1, :] 
    
    # We look for the zero crossing in the region x > 0
    # Xr is the x-coordinate where u goes from negative to positive
    
    # Find indices where x > 0
    start_idx = np.searchsorted(x, 0.0)
    
    # Scan forward from the step
    for i in range(start_idx + 1, len(x) - 1):
        if u_near_wall[i] < 0 and u_near_wall[i+1] > 0:
            # Linear interpolation for zero crossing
            u1 = u_near_wall[i]
            u2 = u_near_wall[i+1]
            x1 = x[i]
            x2 = x[i+1]
            xr = x1 - u1 * (x2 - x1) / (u2 - u1)
            return xr
            
    return None # No reattachment found

def run_simulation(re, nx, ny, nt):
    print(f"Running Re={re}...")
    
    # Setup
    dt = 0.01
    geom = setup_geometry(nx, ny)
    mask = jnp.array(geom['mask'])
    dx, dy = geom['dx'], geom['dy']
    j_step, i_step = geom['j_step'], geom['i_step']
    
    # Physics
    u_max = 1.5
    h_step = geom['dims']['h']
    nu = calculate_nu_from_reynolds(re, u_max, h_step)
    rho = 1.0
    
    # Initialize
    u = jnp.zeros((ny, nx))
    v = jnp.zeros((ny, nx))
    p = jnp.zeros((ny, nx))
    
    # Inlet
    inlet_y = geom['y'][j_step:] - geom['y'][j_step]
    u_inlet_prof = jnp.array(get_inlet_profile(inlet_y, geom['dims']['h_inlet'], u_max))
    
    geom_params = (j_step, i_step)
    
    # Run
    # Warning: We are JIT compiling for every Re, which is somewhat inefficient but fine for this script
    for n in range(nt):
        u, v, p, cfl = solver.time_step_implicit(u, v, p, dt, dx, dy, rho, nu, 20, mask, u_inlet_prof, geom_params)
        
    # Find Xr
    u_np = np.array(u)
    x_np = np.array(geom['x'])
    y_np = np.array(geom['y'])
    
    xr = find_reattachment_length(u_np, x_np, y_np)
    
    if xr:
        xr_normalized = xr / h_step
        print(f"  Re={re}: Xr/h = {xr_normalized:.3f}")
        return xr_normalized
    else:
        print(f"  Re={re}: No reattachment found.")
        return 0.0

def main():
    # Validation Points (Laminar Regime)
    # We avoid Re > 400 because 2D numerical results diverge from 3D experimental results (Armaly) due to 3D effects
    re_values = [100, 200, 300, 400]
    
    # Grid parameters
    nx = 401
    ny = 101
    nt = 10000 # Increased for convergence
    
    results_xr = []
    
    for re in re_values:
        xr = run_simulation(re, nx, ny, nt)
        results_xr.append(xr)
        
    # Armaly et al. (1983) Approx Experimental Data points
    armaly_re = [100, 200, 300, 400]
    armaly_xr = [3.0, 5.0, 6.5, 8.2] # Approximate readings from graph
    
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(armaly_re, armaly_xr, 'ko', label='Exp. (Armaly et al. 1983)')
    plt.plot(re_values, results_xr, 'db-', label='Implicit Solver (JAX)')
    
    plt.xlabel('Reynolds Number (Re)')
    plt.ylabel('Reattachment Length (Xr/h)')
    plt.title('Backward-Facing Step: Reattachment Length Validation')
    plt.legend()
    plt.grid(True)
    
    output_path = os.path.join(root_dir, "simulations", "experiments", "results", "reattachment_validation.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
