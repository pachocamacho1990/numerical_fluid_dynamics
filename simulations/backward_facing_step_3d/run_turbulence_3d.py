"""
3D Turbulent BFS Simulation Runner
==================================

Target: Re=2000 (Implicit LES regime)
Grid: 301 x 76 x 76
Technique: Implicit ADI + Pressure Poisson (JAX XLA compiled)

Features:
- Checkpointing (saves .npz every N steps)
- Restart capability
- lax.scan optimization loop
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

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

# Reuse existing solver logic
from solver_3d import time_step_3d
from geometry_3d import (
    setup_geometry_3d,
    get_inlet_profile_3d,
    calculate_nu_from_reynolds
)

# Enable F64
jax.config.update("jax_enable_x64", True)


def run_simulation(args):
    """Run the 3D Turbulent Simulation."""
    
    print("=" * 60)
    print(f"3D TURBULENT BFS (Re={args.re})")
    print("=" * 60)
    print(f"Grid: {args.nx} x {args.ny} x {args.nz}")
    print(f"Steps: {args.nt} (Chunk size: {args.checkpoint_freq})")
    print(f"Backend: {jax.default_backend()}")
    
    # -------------------------------------------------------------------------
    # 1. Setup
    # -------------------------------------------------------------------------
    output_dir = "simulations/backward_facing_step_3d/outputs_turbulence"
    os.makedirs(output_dir, exist_ok=True)
    
    # Geometry
    geom = setup_geometry_3d(args.nx, args.ny, args.nz)
    mask = jnp.array(geom['mask'])
    dx, dy, dz = geom['dx'], geom['dy'], geom['dz']
    j_step, i_step = geom['j_step'], geom['i_step']
    
    # Physics
    u_max = 1.5
    h_step = geom['dims']['h']
    W = geom['dims']['W']
    nu = calculate_nu_from_reynolds(args.re, u_max, h_step)
    rho = 1.0
    
    print(f"Nu: {nu:.7f}")
    print(f"dt: {args.dt}")
    
    # Inlet Profile
    inlet_y = geom['inlet_y']
    u_inlet = jnp.array(get_inlet_profile_3d(inlet_y, geom['z'], geom['dims']['h_inlet'], W, u_max))
    
    # Static Params for JIT
    geom_params = (j_step, i_step)
    
    # -------------------------------------------------------------------------
    # 2. State Initialization / Restart
    # -------------------------------------------------------------------------
    start_step = 0
    
    # Check for latest checkpoint
    checkpoints = sorted(glob.glob(os.path.join(output_dir, "checkpoint_*.npz")))
    if checkpoints and not args.force_restart:
        latest = checkpoints[-1]
        print(f"Resuming from checkpoint: {latest}")
        data = np.load(latest)
        u = jnp.array(data['u'])
        v = jnp.array(data['v'])
        w = jnp.array(data['w'])
        p = jnp.array(data['p'])
        start_step = int(data['step'])
        print(f"Resuming at step {start_step}")
    else:
        print("Starting fresh simulation.")
        u = jnp.zeros((args.nz, args.ny, args.nx))
        v = jnp.zeros((args.nz, args.ny, args.nx))
        w = jnp.zeros((args.nz, args.ny, args.nx))
        p = jnp.zeros((args.nz, args.ny, args.nx))
    
    if start_step >= args.nt:
        print("Simulation already completed!")
        return

    # -------------------------------------------------------------------------
    # 3. Compile Step Function (lax.scan body)
    # -------------------------------------------------------------------------
    print("\nCompiling XLA kernel...")
    
    # We define a scan body function that runs 'checkpoint_freq' steps at once
    # This keeps the python loop overhead minimal
    
    @jit
    def multi_step_fn(carry_state, _):
        """Runs one chunk of steps inside XLA."""
        u_in, v_in, w_in, p_in = carry_state
        
        def step_fn(carry, _):
            u, v, w, p = carry
            u_new, v_new, w_new, p_new, cfl = time_step_3d(
                u, v, w, p, args.dt, dx, dy, dz, rho, nu, args.nit, 
                mask, u_inlet, geom_params
            )
            return (u_new, v_new, w_new, p_new), cfl
            
        # Run loop for 'checkpoint_freq' steps
        # Checkpoint freq must be static for this to work as loop length
        (u_out, v_out, w_out, p_out), cfls = lax.scan(
            step_fn, (u_in, v_in, w_in, p_in), None, length=args.checkpoint_freq
        )
        
        return (u_out, v_out, w_out, p_out), cfls[-1] # Return last CFL

    # Trigger compilation with a dummy call if needed, or just let first run handle it
    print("Compilation triggered on first step...")
    
    # -------------------------------------------------------------------------
    # 4. Main Loop
    # -------------------------------------------------------------------------
    total_steps = args.nt
    steps_remaining = total_steps - start_step
    
    # Calculate number of chunks
    n_chunks = steps_remaining // args.checkpoint_freq
    remainder = steps_remaining % args.checkpoint_freq
    
    if remainder != 0:
        print(f"Warning: steps ({steps_remaining}) not divisible by frequency ({args.checkpoint_freq}). Last few steps might be skipped in this simplified runner.")
    
    start_time = time.time()
    
    for chunk in range(n_chunks):
        current_step_base = start_step + chunk * args.checkpoint_freq
        
        print(f"Running steps {current_step_base} -> {current_step_base + args.checkpoint_freq} ... ", end="", flush=True)
        chunk_start = time.time()
        
        # Execute Chunk
        (u, v, w, p), last_cfl = multi_step_fn((u, v, w, p), None)
        u.block_until_ready() # Force sync
        
        chunk_time = time.time() - chunk_start
        sps = args.checkpoint_freq / chunk_time
        
        print(f"Done ({chunk_time:.1f}s, {sps:.1f} steps/s). CFL={float(last_cfl):.4f}, Max U={float(jnp.max(u)):.4f}")
        
        # Checkpoint
        step_idx = current_step_base + args.checkpoint_freq
        save_path = os.path.join(output_dir, f"checkpoint_{step_idx:06d}.npz")
        np.savez(save_path, u=u, v=v, w=w, p=p, step=step_idx, 
                 dx=dx, dy=dy, dz=dz)
        # Also clean up old checkpoints? (Keep only last few? For now keep all to make movie)
    
    total_duration = time.time() - start_time
    print(f"\nSimulation finished in {total_duration:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Physics
    parser.add_argument("--re", type=float, default=2000.0)
    parser.add_argument("--nx", type=int, default=301)
    parser.add_argument("--ny", type=int, default=76)
    parser.add_argument("--nz", type=int, default=76)
    parser.add_argument("--dt", type=float, default=0.005)
    
    # Execution
    parser.add_argument("--nt", type=int, default=120000)
    parser.add_argument("--nit", type=int, default=30) # Pressure iters
    parser.add_argument("--checkpoint_freq", type=int, default=5000)
    parser.add_argument("--force_restart", action="store_true")
    
    args = parser.parse_args()
    
    run_simulation(args)
