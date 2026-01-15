import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os
import sys

# Ensure we can import from the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cavity_flow_jax import time_step

def main():
    # Parameters matches Barba Step 11 defaults
    # High-Res for Stability Test (5x original)
    nx = 201
    ny = 201
    nt = 5000  # More steps effectively, though physical time is less if dt is small
    nit = 50
    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)
    dt = 0.0001 # Reduced dt for fine grid
    rho = 1.0
    
    # Reynolds numbers to test (Testing limits with high res)
    re_values = [100, 1000, 2500, 5000, 7500]
    
    results = {}
    
    print(f"Starting Stability Analysis across Re: {re_values}")
    
    # JAX Warmup
    print("Warming up JAX...")
    u = jnp.zeros((ny, nx))
    v = jnp.zeros((ny, nx))
    p = jnp.zeros((ny, nx))
    ti = time.time()
    _ = time_step(u, v, p, dt, dx, dy, rho, 0.1, nit)
    u.block_until_ready()
    print(f"Warmup done ({time.time()-ti:.2f}s)")

    for Re in re_values:
        nu = 2.0 / Re
        print(f"Testing Re={Re} (nu={nu:.5f})...")
        
        # Reset fields
        u = jnp.zeros((ny, nx))
        v = jnp.zeros((ny, nx))
        p = jnp.zeros((ny, nx))
        
        vel_history = []
        cfl_history = []
        
        start_t = time.time()
        crashed = False
        
        for n in range(nt):
            u, v, p, cfl = time_step(u, v, p, dt, dx, dy, rho, nu, nit)
            
            # Helper to get scalar for checking
            # Note: pull to host is slow inside loop, but for 700 steps it's ok for analysis
            # To optimize we could keep on device and only check every N steps or just store and plot later.
            # But let's act conservatively to catch NaNs.
            
            # Actually, let's just append to list and convert later to avoid blocking every step
            # We can check for NaN at the end or if validation is critically needed.
            # But for "study", capturing the explosion is useful.
            
            # Let's verify every 50 steps
            if n % 50 == 0:
                max_u = float(jnp.max(jnp.abs(u)))
                if np.isnan(max_u) or max_u > 10.0:
                    print(f"  -> CRASHED at step {n}! Max U: {max_u}")
                    crashed = True
                    break
            
            # We need the values for plotting. 
            # Appending JAX arrays to list consumes memory but valid for small nt.
            # Better: Keep on device, stack at end? No, history is scalar.
            # cfl is scalar-ish (0-d array).
            
            cfl_val = float(cfl) # This forces sync, slows down but ensures we have data
            max_vel = float(jnp.max(jnp.abs(u)))
            
            vel_history.append(max_vel)
            cfl_history.append(cfl_val)
            
            if max_vel > 20.0 or np.isnan(max_vel):
                 print(f"  -> CRASHED at step {n}! Max U: {max_vel}")
                 crashed = True
                 break
        
        duration = time.time() - start_t
        print(f"  -> Finished {len(vel_history)} steps in {duration:.2f}s")
        
        results[Re] = {
            "vel": vel_history,
            "cfl": cfl_history,
            "crashed": crashed
        }

    # --- Plotting ---
    print("Generating plots...")
    
    # 1. Max Velocity vs Time
    plt.figure(figsize=(10, 6), dpi=100)
    for Re, data in results.items():
        ls = "-" if not data["crashed"] else "--"
        label = f"Re={Re}" + (" (Crashed)" if data["crashed"] else "")
        plt.plot(data["vel"], label=label, linestyle=ls, linewidth=2)
        
    plt.axhline(y=1.0, color='k', linestyle=':', alpha=0.5, label="Lid Velocity")
    plt.ylim(0, 5.0) # Limit y-axis to see useful part before explosion
    plt.xlabel("Time Step")
    plt.ylabel("Max Velocity (magnitude)")
    plt.title("Stability Analysis: Max Velocity vs Reynolds Number")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("simulations/cavity_flow/stability_velocity.png")
    
    # 2. CFL vs Time
    plt.figure(figsize=(10, 6), dpi=100)
    for Re, data in results.items():
        ls = "-" if not data["crashed"] else "--"
        plt.plot(data["cfl"], label=f"Re={Re}", linestyle=ls)
        
    plt.xlabel("Time Step")
    plt.ylabel("CFL Number")
    plt.title("CFL Evolution vs Reynolds Number")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("simulations/cavity_flow/stability_cfl.png")
    
    print("Done.")

if __name__ == "__main__":
    main()
