#!/usr/bin/env python3
"""
DNS-LES Comparison Runner

Matches the DNS turbulent simulation parameters:
- Re = 2000
- Physical Time = 600 (120,000 steps × dt=0.005)
- Same geometry and boundary conditions  

Allows grid resolution variation to study LES coarsening effects.
"""

import jax
import jax.numpy as jnp
from jax import jit, lax
import numpy as np
import time
import os
import sys
import argparse
import glob
from functools import partial

# Enable F64 for consistency with DNS
jax.config.update("jax_enable_x64", True)

sys.path.append(os.getcwd())

from simulations.les.solver_3d_les import time_step_3d_les
from simulations.backward_facing_step_3d.geometry_3d import (
    setup_geometry_3d,
    get_inlet_profile_3d,
    calculate_nu_from_reynolds
)

def run_simulation(args):
    """Run LES simulation matching DNS parameters."""
    
    print("=" * 70)
    print(f"DNS-LES COMPARISON: LES Run (Re={args.re})")
    print("=" * 70)
    print(f"Grid: {args.nx} × {args.ny} × {args.nz}")
    print(f"Steps: {args.nt} (Checkpoint freq: {args.checkpoint_freq})")
    print(f"Physical Time: T = {args.nt * args.dt:.1f}")
    print(f"Backend: {jax.default_backend()}")
    print(f"Smagorinsky Cs: {args.cs}")
    
    # Setup
    output_dir = args.save_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Geometry (same as DNS)
    geom = setup_geometry_3d(args.nx, args.ny, args.nz)
    mask = jnp.array(geom['mask'])
    dx, dy, dz = geom['dx'], geom['dy'], geom['dz']
    j_step, i_step = geom['j_step'], geom['i_step']
    geom_params = (j_step, i_step)
    
    # Physics (match DNS exactly)
    u_max = 1.5  # Same as DNS
    h_step = geom['dims']['h']
    W = geom['dims']['W']
    nu = calculate_nu_from_reynolds(args.re, u_max, h_step)
    rho = 1.0
    
    print(f"Nu (molecular): {nu:.7f}")
    print(f"dt: {args.dt}")
    print(f"Grid spacing: dx={dx:.6f}, dy={dy:.6f}, dz={dz:.6f}")
    
    # Inlet Profile
    inlet_y = geom['inlet_y']
    u_inlet = jnp.array(get_inlet_profile_3d(inlet_y, geom['z'], geom['dims']['h_inlet'], W, u_max))
    
    # State initialization or restart
    start_step = 0
    checkpoints = sorted(glob.glob(os.path.join(output_dir, "checkpoint_*.npz")))
    
    if checkpoints and not args.force_restart:
        latest = checkpoints[-1]
        print(f"Resuming from: {latest}")
        data = np.load(latest)
        u = jnp.array(data['u'])
        v = jnp.array(data['v'])
        w = jnp.array(data['w'])
        p = jnp.array(data['p'])
        start_step = int(data['step'])
        print(f"Resuming at step {start_step}")
    else:
        print("Starting fresh LES simulation.")
        u = jnp.zeros((args.nz, args.ny, args.nx))
        v = jnp.zeros((args.nz, args.ny, args.nx))
        w = jnp.zeros((args.nz, args.ny, args.nx))
        p = jnp.zeros((args.nz, args.ny, args.nx))
    
    if start_step >= args.nt:
        print("Simulation already completed!")
        return
    
    # Compile step function
    print("\nCompiling LES solver...")
    
    @jit
    def multi_step_fn(carry_state, _):
        """Run one chunk of LES steps."""
        u_in, v_in, w_in, p_in = carry_state
        
        def step_fn(carry, _):
            u, v, w, p = carry
            u_new, v_new, w_new, p_new, nu_t, cfl = time_step_3d_les(
                u, v, w, p, args.dt, dx, dy, dz, rho, nu, args.nit,
                mask, u_inlet, geom_params
            )
            return (u_new, v_new, w_new, p_new), (cfl, jnp.max(nu_t))
        
        (u_out, v_out, w_out, p_out), (cfls, nu_ts) = lax.scan(
            step_fn, (u_in, v_in, w_in, p_in), None, length=args.checkpoint_freq
        )
        
        return (u_out, v_out, w_out, p_out), (cfls[-1], nu_ts[-1])
    
    print("Compilation triggered on first step...")
    
    # Main Loop
    steps_remaining = args.nt - start_step
    n_chunks = steps_remaining // args.checkpoint_freq
    
    start_time = time.time()
    
    for chunk in range(n_chunks):
        current_step_base = start_step + chunk * args.checkpoint_freq
        
        print(f"Steps {current_step_base} → {current_step_base + args.checkpoint_freq} ... ", end="", flush=True)
        chunk_start = time.time()
        
        # Execute
        (u, v, w, p), (last_cfl, last_nu_t) = multi_step_fn((u, v, w, p), None)
        u.block_until_ready()
        
        chunk_time = time.time() - chunk_start
        sps = args.checkpoint_freq / chunk_time
        
        print(f"Done ({chunk_time:.1f}s, {sps:.1f} it/s). CFL={float(last_cfl):.4f}, U_max={float(jnp.max(u)):.4f}, ν_t_max={float(last_nu_t):.6f}")
        
        # Checkpoint
        step_idx = current_step_base + args.checkpoint_freq
        save_path = os.path.join(output_dir, f"checkpoint_{step_idx:06d}.npz")
        np.savez(save_path, 
                 u=u, v=v, w=w, p=p, 
                 step=step_idx,
                 x=geom['x'], y=geom['y'], z=geom['z'],
                 dx=dx, dy=dy, dz=dz,
                 Re=args.re, dt=args.dt, cs=args.cs)
    
    total_duration = time.time() - start_time
    print(f"\nLES simulation finished in {total_duration:.2f}s ({total_duration/3600:.2f} hours)")
    print(f"Average: {(args.nt - start_step)/total_duration:.1f} steps/s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DNS-LES Comparison: LES Runner")
    
    # DNS-matched parameters
    parser.add_argument("--re", type=float, default=2000.0, help="Reynolds number (DNS=2000)")
    parser.add_argument("--dt", type=float, default=0.005, help="Time step (DNS=0.005)")
    parser.add_argument("--nt", type=int, default=120000, help="Total steps (DNS=120000)")
    
    # Grid (allow variation)
    parser.add_argument("--nx", type=int, default=128, help="Grid nx (DNS=301)")
    parser.add_argument("--ny", type=int, default=32, help="Grid ny (DNS=76)")
    parser.add_argument("--nz", type=int, default=32, help="Grid nz (DNS=76)")
    
    # LES parameters
    parser.add_argument("--cs", type=float, default=0.12, help="Smagorinsky constant")
    parser.add_argument("--nit", type=int, default=20, help="Pressure iterations")
    
    # Execution
    parser.add_argument("--checkpoint_freq", type=int, default=5000)
    parser.add_argument("--force_restart", action="store_true")
    parser.add_argument("--save_dir", default="simulations/les/outputs_dns_comparison")
    
    args = parser.parse_args()
    run_simulation(args)
