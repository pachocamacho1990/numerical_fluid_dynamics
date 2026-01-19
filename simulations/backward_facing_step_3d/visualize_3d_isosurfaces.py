"""
3D Isosurface Visualization using PyVista (Requires Python 3.11+ venv)

This script generates 3D isosurfaces for the backward-facing step flow.
It is designed to run in a separate environment (`venv_vis`) to ensure
compatibility with VTK and PyVista.

Usage:
    ../venv_vis/bin/python visualize_3d_isosurfaces.py --interactive
    ../venv_vis/bin/python visualize_3d_isosurfaces.py --q-criterion --rotate
"""

import numpy as np
import pyvista as pv
import os
import argparse

def load_solution(filepath):
    print(f"Loading: {filepath}")
    data = np.load(filepath)
    return {
        'u': data['u'],
        'v': data['v'],
        'w': data['w'],
        'x': data['x'],
        'y': data['y'],
        'z': data['z'],
    }

def compute_q_criterion(grid):
    """
    Compute Q-criterion for vortex identification.
    Q = 0.5 * ( ||Omega||^2 - ||S||^2 )
    
    PyVista can compute derivatives (gradients) which we use to construct
    the velocity evaluation tensor.
    """
    print("Computing Q-criterion...")
    
    # PyVista's compute_derivative calculates the gradient of the vector field
    # Result is a (N, 9) array of gradients: du/dx, du/dy, du/dz, dv/dx...
    grid = grid.compute_derivative(scalars="velocity", gradient=True, faster=True)
    
    grad = grid["gradient"].reshape(-1, 3, 3) # (N, 3, 3)
    
    # Strain rate tensor S = 0.5 * (grad + grad.T)
    # Rotation tensor Omega = 0.5 * (grad - grad.T)
    
    grad_t = grad.transpose(0, 2, 1)
    S = 0.5 * (grad + grad_t)
    Omega = 0.5 * (grad - grad_t)
    
    # Frobenious norm squared
    norm_S2 = np.sum(S**2, axis=(1,2))
    norm_Omega2 = np.sum(Omega**2, axis=(1,2))
    
    Q = 0.5 * (norm_Omega2 - norm_S2)
    
    grid.point_data["Q"] = Q
    return grid

def create_grid(data):
    """Convert numpy data to PyVista grid with velocity vector."""
    u, v, w = data['u'], data['v'], data['w']
    x, y, z = data['x'], data['y'], data['z']
    
    # Create PyVista Grid
    grid = pv.RectilinearGrid(x, y, z)
    
    # Transpose to (nx, ny, nz) and flatten F-order for PyVista
    u_t = u.transpose(2, 1, 0).flatten(order='F')
    v_t = v.transpose(2, 1, 0).flatten(order='F')
    w_t = w.transpose(2, 1, 0).flatten(order='F')
    
    # Stack into velocity vector
    velocity = np.column_stack((u_t, v_t, w_t))
    grid.point_data['velocity'] = velocity
    grid.point_data['u'] = u_t # Keep scalar u for basic isosurfaces
    
    return grid

def visualize(data, output_dir, args, h_step=0.4845):
    grid = create_grid(data)
    
    # Setup Plotter
    off_screen = not args.interactive
    pl = pv.Plotter(off_screen=off_screen)
    pl.set_background('white')
    
    # Add Step Geometry
    x, z = data['x'], data['z']
    step_box = pv.Box(bounds=[x.min(), 0, 0, h_step, z.min(), z.max()])
    pl.add_mesh(step_box, color='lightgrey', opacity=1.0, show_edges=False)
    pl.add_mesh(grid.outline(), color="black")
    
    if args.q_criterion:
        # Vortex Detection
        grid = compute_q_criterion(grid)
        
        # Q > 0 indicates vortex regions. Value depends on scale.
        # Try a small positive value relative to max Q
        q_max = grid.point_data["Q"].max()
        iso_val = args.iso_val if args.iso_val else 0.01 * q_max
        
        print(f"Generating Q={iso_val:.4f} isosurface (Max Q={q_max:.4f})...")
        isosurf = grid.contour(isosurfaces=[iso_val], scalars='Q')
        
        pl.add_mesh(isosurf, color='orange', opacity=0.8, 
                    smooth_shading=True, show_edges=False,
                    label=f'Q-Criterion (Vortices)')
        save_name = "isosurface_Q"
    else:
        # Recirculation Bubble (u=0)
        print("Generating u=0 isosurface...")
        isosurf = grid.contour(isosurfaces=[0.0], scalars='u')
        
        pl.add_mesh(isosurf, color='dodgerblue', opacity=0.6, 
                    smooth_shading=True, show_edges=False,
                    label='Recirculation (u=0)')
        save_name = "isosurface_u0"

    # Camera Setup
    pl.camera_position = 'iso'
    pl.camera.azimuth = -45
    pl.camera.elevation = 30
    pl.add_axes()

    if args.interactive:
        print("\nInteractive Mode: Use mouse to rotate/zoom. Press 'q' to exit.")
        pl.show()
    elif args.rotate:
        print("\nGenerating rotation animation...")
        outpath = os.path.join(output_dir, f"{save_name}_rotation.gif")
        path = pl.open_gif(outpath)
        
        # Orbit animation
        pl.camera_position = 'iso'
        pl.camera.azimuth = -45
        pl.camera.elevation = 30
        
        n_frames = 36
        for i in range(n_frames):
            pl.camera.azimuth += 360 / n_frames
            pl.write_frame()
            
        pl.close()
        print(f"Saved animation: {outpath}")
    else:
        outpath = os.path.join(output_dir, f"{save_name}.png")
        pl.screenshot(outpath, transparent_background=False)
        pl.close()
        print(f"Saved snapshot: {outpath}")


def main():
    parser = argparse.ArgumentParser(description="PyVista 3D Visualization")
    parser.add_argument("--input", default="outputs/solution_3d.npz", help="Input .npz file")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--interactive", action="store_true", help="Open interactive window")
    parser.add_argument("--rotate", action="store_true", help="Generate rotation GIF (off-screen)")
    parser.add_argument("--q-criterion", action="store_true", help="Visualize Q-criterion for vortices")
    parser.add_argument("--iso-val", type=float, help="Isovalue for Q-criterion")
    
    args = parser.parse_args()
    
    # Handle paths (same relative logic as before)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = args.input
    if not os.path.isabs(input_path):
        input_path = os.path.join(script_dir, input_path)
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(script_dir, output_dir)
    
    if not os.path.exists(input_path):
        project_root = os.path.join(script_dir, "../../../")
        input_path = os.path.join(project_root, "simulations/backward_facing_step_3d/outputs/solution_3d.npz")
        output_dir = os.path.join(project_root, "simulations/backward_facing_step_3d/outputs")
    
    if not os.path.exists(input_path):
        print(f"Error: Solution file not found at {input_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    data = load_solution(input_path)
    visualize(data, output_dir, args)
    print("Done.")

if __name__ == "__main__":
    main()
