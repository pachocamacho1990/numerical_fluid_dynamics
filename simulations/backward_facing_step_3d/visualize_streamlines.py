"""
3D Flow Visualization with Streamlines (PyVista)

Creates high-quality 3D visualizations showing:
- Backward-facing step geometry
- Streamlines colored by velocity magnitude
- Q-criterion isosurfaces for vortex identification

Usage:
    ../venv_vis/bin/python visualize_streamlines.py
    ../venv_vis/bin/python visualize_streamlines.py --interactive
    ../venv_vis/bin/python visualize_streamlines.py --q-criterion
"""

import numpy as np
import pyvista as pv
import os
import argparse


def load_solution(filepath):
    """Load 3D simulation results."""
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
    
    # Velocity magnitude for coloring
    vel_mag = np.sqrt(u_t**2 + v_t**2 + w_t**2)
    grid.point_data['velocity_magnitude'] = vel_mag
    grid.point_data['u'] = u_t
    
    return grid


def compute_q_criterion(grid):
    """Compute Q-criterion for vortex identification."""
    print("Computing Q-criterion...")
    
    grid = grid.compute_derivative(scalars="velocity", gradient=True, faster=True)
    grad = grid["gradient"].reshape(-1, 3, 3)
    
    grad_t = grad.transpose(0, 2, 1)
    S = 0.5 * (grad + grad_t)
    Omega = 0.5 * (grad - grad_t)
    
    norm_S2 = np.sum(S**2, axis=(1, 2))
    norm_Omega2 = np.sum(Omega**2, axis=(1, 2))
    
    Q = 0.5 * (norm_Omega2 - norm_S2)
    grid.point_data["Q"] = Q
    return grid


def visualize_streamlines(data, output_dir, args, h_step=0.4845):
    """Create streamline visualization with optional Q-criterion."""
    grid = create_grid(data)
    
    x, y, z = data['x'], data['y'], data['z']
    
    # Setup Plotter
    off_screen = not args.interactive
    pl = pv.Plotter(off_screen=off_screen, window_size=(1920, 1080))
    pl.set_background('white')
    
    # Add Step Geometry
    step_box = pv.Box(bounds=[x.min(), 0, 0, h_step, z.min(), z.max()])
    pl.add_mesh(step_box, color='grey', opacity=1.0, show_edges=True)
    pl.add_mesh(grid.outline(), color="black")
    
    # Add Q-criterion isosurface if requested
    if args.q_criterion:
        grid = compute_q_criterion(grid)
        q_max = grid.point_data["Q"].max()
        iso_val = 0.9 * q_max
        print(f"Generating Q-criterion isosurface at Q = {iso_val:.4f} (max = {q_max:.4f})")
        isosurf = grid.contour(isosurfaces=[iso_val], scalars='Q')
        pl.add_mesh(isosurf, color="green", opacity=0.2, smooth_shading=True, label='vortices')
    
    # Generate Streamlines from multiple seed locations
    print("Generating streamlines...")
    
    # Seed 1: Near step corner (captures recirculation)
    seed_center_1 = (0.2, 0.4, z.max() - 0.1)
    streamlines_1 = grid.streamlines(
        vectors='velocity',
        source_radius=0.1,
        source_center=seed_center_1,
        n_points=200,
        max_time=14.0,
    )
    
    # Seed 2: Near sidewall (captures boundary layer flow)
    seed_center_2 = (0.2, 0.5, 0.1 + z.min())
    streamlines_2 = grid.streamlines(
        vectors='velocity',
        source_radius=0.1,
        source_center=seed_center_2,
        n_points=200,
        max_time=14.0,
    )
    
    # Add Streamlines as tubes colored by velocity
    if streamlines_1.n_points > 0:
        tubes_1 = streamlines_1.tube(radius=0.005)
        pl.add_mesh(tubes_1, cmap='coolwarm', scalars='velocity_magnitude',
                    show_scalar_bar=True, scalar_bar_args={'title': 'Velocity Magnitude'})
    
    if streamlines_2.n_points > 0:
        tubes_2 = streamlines_2.tube(radius=0.005)
        pl.add_mesh(tubes_2, cmap='coolwarm', scalars='velocity_magnitude',
                    show_scalar_bar=False)
    
    pl.add_axes()
    
    # Camera position from notebook (good view angle)
    camera_position = [
        (9.35, 1.56, -0.44),  # Camera location
        (6.06, 0.50, 1.94),   # Focal point
        (-0.13, 0.96, 0.25)   # View up
    ]
    
    if args.interactive:
        print("\nInteractive Mode: Use mouse to rotate/zoom. Press 'q' to exit.")
        pl.show()
    else:
        pl.camera_position = camera_position
        outpath = os.path.join(output_dir, "streamlines_3d.png")
        pl.screenshot(outpath, transparent_background=False, window_size=(2920, 1080))
        pl.close()
        print(f"Saved: {outpath}")
        return outpath


def main():
    parser = argparse.ArgumentParser(description="3D Streamline Visualization")
    parser.add_argument("--input", default="outputs/solution_3d.npz", help="Input .npz file")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--interactive", action="store_true", help="Open interactive window")
    parser.add_argument("--q-criterion", action="store_true", help="Add Q-criterion vortex isosurface")
    
    args = parser.parse_args()
    
    # Handle paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = args.input
    if not os.path.isabs(input_path):
        input_path = os.path.join(script_dir, input_path)
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(script_dir, output_dir)
    
    if not os.path.exists(input_path):
        print(f"Error: Solution file not found at {input_path}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    data = load_solution(input_path)
    visualize_streamlines(data, output_dir, args)
    print("Done.")


if __name__ == "__main__":
    main()
