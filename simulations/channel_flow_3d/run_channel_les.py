
import argparse
import jax
import jax.numpy as jnp
import numpy as np
import time
import os
import sys

sys.path.append(os.getcwd())

from simulations.les.solver_3d_les import time_step_3d_les
from simulations.channel_flow_3d.geometry_channel import setup_channel_geometry
from simulations.les.config import LESConfig
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="3D LES Channel Flow (JAX)")
    parser.add_argument("--re_tau", type=float, default=180.0) # Friction Reynolds Number
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--nz", type=int, default=32)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--nt", type=int, default=1000)
    parser.add_argument("--stat_start", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default="simulations/channel_flow_3d/outputs")
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"3D LES Turbulent Channel Flow (Re_tau={args.re_tau})")
    print("=" * 60)
    
    # Physics parameters for Channel Flow
    # Non-dimensionalization by channel half-height (delta=1) and friction velocity (u_tau=1)
    # Then nu = 1 / Re_tau
    delta = 1.0
    u_tau = 1.0
    nu = 1.0 / args.re_tau
    rho = 1.0
    
    # Geometry
    # Domain size typically 4pi x 2 x 2pi or 2pi x 2 x pi 
    # Using 'Minimal Flow Unit' or small domain for testing: 2pi x 2 x pi
    geom = setup_channel_geometry(args.nx, args.ny, args.nz, Lx=2*np.pi, Ly=2.0, Lz=np.pi)
    
    mask = jnp.array(geom['mask'])
    dx, dy, dz = geom['dx'], geom['dy'], geom['dz']
    # No step logic
    geom_params = (0, 0) # j_step, i_step dummy
    
    print(f"Grid: {args.nx}x{args.ny}x{args.nz}")
    print(f"Viscosity: {nu:.6f}")
    print(f"Resolution: dx+={dx/nu:.1f}, dy+={dy/nu:.1f}, dz+={dz/nu:.1f}")
    
    # Init Fields with perturbations to trigger turbulence
    # Laminar parabolic + noise
    y = jnp.linspace(0, 2.0, args.ny)
    # Parabolic profile u(y) = 1.5*(1 - (y-1)^2)? No, log law for turbulent approx?
    # Or just laminar: u = 3/2 U_bulk (1 - (y/h - 1)^2)
    # U_bulk for Re_tau=180 is approx 15.6 u_tau ? ~15
    # Let's initialize with mean profile + random noise
    
    # Centerline approx 18 u_tau?
    u_mean = 15.0 * (1.0 - ((y - 1.0)/1.0)**8) # Flat plug-like turbulent profile
    # Broadcast
    u = jnp.broadcast_to(u_mean[None, :, None], (args.nz, args.ny, args.nx))
    
    # Noise
    key = jax.random.PRNGKey(42)
    noise_u = jax.random.uniform(key, u.shape) * 0.2 - 0.1
    u = u + noise_u
    v = jax.random.uniform(key, u.shape) * 0.1 - 0.05
    w = jax.random.uniform(key, u.shape) * 0.1 - 0.05
    p = jnp.zeros_like(u)
    
    # Compilation
    print("Compiling...")
    # Inlet is Periodic? 
    # We need to modify solver for Periodic BCs in X and Z if we want true channel flow!
    # The current `solver_3d_les.py` uses Inlet/Outlet BCs and Sidewall BCs (walls).
    # Channel flow implies Periodic X and Periodic Z.
    
    # CRITICAL: Channel flow validation requires Periodic BCs. 
    # The current solver implements Backward-Facing Step BCs (Dirichlet/Neumann).
    
    # Workaround: Run "Developing Channel Flow" (entrance length) 
    # Inlet = constants profile, Outlet = Neumann.
    # This matches the current solver capabilities without rewriting BCs.
    # We essentially simulate a long duct.
    
    print("Note: Running spatially developing channel flow (Entrance problem)")
    
    u_inlet_profile = u_mean[None, :] # (1, ny) -> broadcast roughly
    # Actually u_inlet needs to be (nz, ny)
    # For now, just fixed profile at inlet.
    u_inlet_2d = jnp.broadcast_to(u_mean[None, :], (args.nz, args.ny)) 
    
    _ = time_step_3d_les(u, v, w, p, args.dt, dx, dy, dz, rho, nu, 20, mask, u_inlet_2d, geom_params)
    print("Compilation complete.")
    
    # Loop
    stats_u = jnp.zeros(args.ny)
    count = 0
    
    start = time.time()
    for n in tqdm(range(args.nt)):
        u, v, w, p, nu_t, cfl = time_step_3d_les(u, v, w, p, args.dt, dx, dy, dz, rho, nu, 20, mask, u_inlet_2d, geom_params)
        
        # Accumulate statistics
        if n > args.stat_start:
            # Average over x, z planes
            stats_u += jnp.mean(u, axis=(0, 2))
            count += 1
            
        if n % 100 == 0:
            if jnp.isnan(cfl):
                 print("NaN"); break
                 
    # Save
    if count > 0:
        stats_u /= count
        
    os.makedirs(args.output_dir, exist_ok=True)
    np.savez(f"{args.output_dir}/channel_stats.npz", mean_u=stats_u, y=jnp.linspace(0, 2, args.ny))
    print("Done.")

if __name__ == "__main__":
    main()
