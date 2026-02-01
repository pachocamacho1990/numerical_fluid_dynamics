
import argparse
import jax
import jax.numpy as jnp
import numpy as np
import time
import os
import sys

# Ensure path is set up to find modules
sys.path.append(os.getcwd())

from simulations.les.solver_3d_les import time_step_3d_les
from simulations.les.config import LESConfig
from simulations.backward_facing_step_3d.geometry_3d import (
    setup_geometry_3d,
    get_inlet_profile_3d,
    calculate_nu_from_reynolds
)
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="3D LES BFS Solver (JAX)")
    parser.add_argument("--re", type=float, default=5000.0) # Higher Re for LES
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--ny", type=int, default=32)
    parser.add_argument("--nz", type=int, default=32)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--nt", type=int, default=1000)
    parser.add_argument("--nit", type=int, default=20)
    parser.add_argument("--cs", type=float, default=0.12, help="Smagorinsky constant")
    parser.add_argument("--output_dir", type=str, default="simulations/les/outputs")
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"3D LES Backward-Facing Step (Re={args.re})")
    print("=" * 60)
    
    # Geometry
    geom = setup_geometry_3d(args.nx, args.ny, args.nz)
    mask = jnp.array(geom['mask'])
    dx, dy, dz = geom['dx'], geom['dy'], geom['dz']
    j_step, i_step = geom['j_step'], geom['i_step']
    geom_params = (j_step, i_step)
    
    # Physics
    u_max = 1.0
    h_step = geom['dims']['h']
    W = geom['dims']['W']
    nu = calculate_nu_from_reynolds(args.re, u_max, h_step)
    rho = 1.0
    
    # LES Config
    les_config = LESConfig(cs=args.cs, wall_damping=True)
    
    print(f"Grid: {args.nx}x{args.ny}x{args.nz}")
    print(f"Nu (molecular): {nu:.8f}")
    print(f"LES Model: Smagorinsky (Cs={args.cs}) with Wall Damping")
    
    # Init Fields
    u = jnp.zeros((args.nz, args.ny, args.nx))
    v = jnp.zeros((args.nz, args.ny, args.nx))
    w = jnp.zeros((args.nz, args.ny, args.nx))
    p = jnp.zeros((args.nz, args.ny, args.nx))
    
    # Inlet Profile
    inlet_y = geom['inlet_y']
    u_inlet_2d = jnp.array(get_inlet_profile_3d(inlet_y, geom['z'], geom['dims']['h_inlet'], W, u_max))
    
    # Compilation
    print("Compiling JAX functions...")
    start_compile = time.time()
    # Call once to compile
    _ = time_step_3d_les(u, v, w, p, args.dt, dx, dy, dz, rho, nu, args.nit, mask, u_inlet_2d, geom_params)
    print(f"Compilation complete ({time.time() - start_compile:.2f}s)")
    
    # Simulation Loop
    start = time.time()
    for n in tqdm(range(args.nt)):
        u, v, w, p, nu_t, cfl = time_step_3d_les(u, v, w, p, args.dt, dx, dy, dz, rho, nu, args.nit, mask, u_inlet_2d, geom_params)
        
        if n % 100 == 0:
            u_max_val = float(jnp.max(u))
            nu_t_max = float(jnp.max(nu_t))
            cfl_val = float(cfl)
            tqdm.write(f"Step {n}: CFL={cfl_val:.4f}, U_max={u_max_val:.4f}, Nu_t_max={nu_t_max:.6f}")
            
            if jnp.isnan(cfl):
                print("NaN detected!")
                break
                
    duration = time.time() - start
    print(f"\nCompleted in {duration:.2f}s ({args.nt/duration:.1f} iter/s)")
    
    # Save Output
    os.makedirs(args.output_dir, exist_ok=True)
    np.savez(f"{args.output_dir}/solution_les_re{int(args.re)}.npz",
             u=u, v=v, w=w, p=p, nu_t=nu_t,
             x=geom['x'], y=geom['y'], z=geom['z'])
    print(f"Saved results to {args.output_dir}")

if __name__ == "__main__":
    main()
