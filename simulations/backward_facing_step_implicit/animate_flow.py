"""
Animation script for Backward-Facing Step Flow (Implicit Method)
Run configuration: Re=800, 40k steps.

Strategy:
1. Run simulation and collect snapshots into a large NumPy array.
2. Save this history to disk (.npy).
3. Load the history and generate the GIF.
"""

import sys
import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
import time

# Ensure we can import the solver and geometry
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "..", "backward_facing_step"))

# Import solver components
import backward_facing_step_implicit as solver
from geometry import setup_geometry, get_inlet_profile, calculate_nu_from_reynolds

def run_simulation_and_save():
    # Parameters
    re = 800.0
    nx = 401
    ny = 101
    nt = 40000
    dt = 0.005
    save_interval = 100  # Save every 100 steps
    
    output_dir = os.path.join(current_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    history_path = os.path.join(output_dir, "history_re800.npz")
    
    print(f"Starting Simulation: Re={re}, Grid={nx}x{ny}, Steps={nt}")
    print(f"Traj saving to: {history_path}")
    
    # Setup Geometry
    geom = setup_geometry(nx, ny)
    mask = jnp.array(geom['mask'])
    dx, dy = geom['dx'], geom['dy']
    j_step, i_step = geom['j_step'], geom['i_step']
    
    # Physics
    u_max = 1.5
    h_step = geom['dims']['h']
    nu = calculate_nu_from_reynolds(re, u_max, h_step)
    rho = 1.0
    
    # Initialize fields
    u = jnp.zeros((ny, nx))
    v = jnp.zeros((ny, nx))
    p = jnp.zeros((ny, nx))
    
    # Inlet
    inlet_y = geom['y'][j_step:] - geom['y'][j_step]
    u_inlet_prof = jnp.array(get_inlet_profile(inlet_y, geom['dims']['h_inlet'], u_max))
    
    # Compile
    print("Compiling JAX functions...")
    geom_params = (j_step, i_step)
    solver.time_step_implicit(u, v, p, dt, dx, dy, rho, nu, 20, mask, u_inlet_prof, geom_params)
    print("Compiled.")
    
    # Pre-allocate history array (numpy)
    # Number of frames = nt // save_interval
    n_frames = nt // save_interval
    u_history = np.zeros((n_frames, ny, nx), dtype=np.float32)
    
    print("Running simulation loop...")
    start_time = time.time()
    
    # We iterate and fill the numpy array
    frame_idx = 0
    
    for n in tqdm(range(nt)):
        u, v, p, cfl = solver.time_step_implicit(u, v, p, dt, dx, dy, rho, nu, 20, mask, u_inlet_prof, geom_params)
        
        # Save frame
        if n % save_interval == 0:
            if frame_idx < n_frames:
                u_history[frame_idx] = np.array(u)
                frame_idx += 1
            
            if n % 2000 == 0:
                 cfl_val = float(cfl)
                 if np.isnan(cfl_val):
                     print("NaN detected! Stopping early.")
                     break
                     
    duration = time.time() - start_time
    print(f"Simulation finished in {duration:.2f}s")
    
    # Save to disk
    print("Saving history to disk...")
    np.savez_compressed(history_path, u=u_history, mask=np.array(mask), 
                        x=geom['x'], y=geom['y'], re=re, dt=dt, save_interval=save_interval)
    print("Save complete.")
    return history_path

def generate_animation_from_file(history_path):
    print(f"Loading history from {history_path}...")
    data = np.load(history_path)
    u_frames = data['u']
    mask = data['mask']
    x = data['x']
    y = data['y']
    re = data['re']
    
    print(f"Generating animation for {len(u_frames)} frames...")
    
    fig, ax = plt.subplots(figsize=(10, 3.5))
    
    # Mask solid regions
    mask_plot = np.where(mask == 0, np.nan, 1.0)
    
    # Initial plot
    u_plot = u_frames[0] * mask_plot
    img = ax.imshow(u_plot, extent=[x[0], x[-1], y[0], y[-1]], 
                    cmap='turbo', origin='lower', vmin=0, vmax=1.6) # Fixed scale for stability
    plt.colorbar(img, ax=ax, label='Velocity Magnitude (u)')
    title = ax.set_title(f"Re={re} Flow (Frame 0)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()
    
    def update(frame_idx):
        u_data = u_frames[frame_idx] * mask_plot
        img.set_data(u_data)
        title.set_text(f"Re={re} Flow (Frame {frame_idx})")
        return img, title
    
    anim = animation.FuncAnimation(fig, update, frames=len(u_frames), blit=True, interval=50)
    
    output_path = history_path.replace(".npz", ".gif")
    print(f"Saving GIF to {output_path}...")
    
    try:
        anim.save(output_path, writer='pillow', fps=30)
        print("Animation saved successfully!")
    except Exception as e:
        print(f"Failed to save animation: {e}")

if __name__ == "__main__":
    # Check if history exists to optionally skip sim (commented out for now, always run)
    path = run_simulation_and_save()
    generate_animation_from_file(path)
