
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

try:
    import pyvista as pv
    HAVE_PYVISTA = True
except ImportError:
    HAVE_PYVISTA = False
    print("PyVista not found. 3D visualization will be skipped.")

def main():
    parser = argparse.ArgumentParser(description="Visualize LES Results")
    parser.add_argument("--file", type=str, required=True, help="Path to solution_les_*.npz")
    parser.add_argument("--output_dir", type=str, default="simulations/les/outputs/viz")
    parser.add_argument("--show", action="store_true", help="Show interactive plot")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading {args.file}...")
    data = np.load(args.file)
    u = data['u']
    v = data['v']
    w = data['w']
    p = data['p']
    if 'nu_t' in data:
        nu_t = data['nu_t']
    else:
        print("Warning: nu_t not found in file. Using zeros.")
        nu_t = np.zeros_like(u)
        
    x = data['x']
    y = data['y']
    z = data['z']
    
    nz, ny, nx = u.shape
    print(f"Grid: {nx}x{ny}x{nz}")
    print(f"Max Velocity: {np.max(u):.4f}")
    print(f"Max Eddy Viscosity: {np.max(nu_t):.6f}")
    
    # 1. Matplotlib Slices (Centerline)
    plt.figure(figsize=(15, 10))
    
    # Mid-plane Z
    z_idx = nz // 2
    
    plt.subplot(3, 1, 1)
    plt.contourf(x, y, u[z_idx, :, :], levels=50, cmap='viridis')
    plt.colorbar(label='u velocity')
    plt.title(f'Velocity U (z={z[z_idx]:.2f})')
    plt.xlabel('x')
    plt.ylabel('y')
    
    plt.subplot(3, 1, 2)
    plt.contourf(x, y, nu_t[z_idx, :, :], levels=50, cmap='inferno')
    plt.colorbar(label='Eddy Viscosity nu_t')
    plt.title(f'Eddy Viscosity (z={z[z_idx]:.2f})')
    plt.xlabel('x')
    plt.ylabel('y')
    
    # Mid-plane Y (Top view)
    y_idx = ny // 2
    plt.subplot(3, 1, 3)
    plt.contourf(x, z, u[:, y_idx, :], levels=50, cmap='viridis')
    plt.colorbar(label='u velocity')
    plt.title(f'Velocity U (y={y[y_idx]:.2f})')
    plt.xlabel('x')
    plt.ylabel('z')
    
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/les_slices.png")
    print(f"Saved slices to {args.output_dir}/les_slices.png")
    
    # 2. PyVista 3D Viz
    if HAVE_PYVISTA:
        print("Generating 3D Visualization...")
        grid = pv.RectilinearGrid(x, y, z)
        
        # Add data (flatten in F order for PyVista/VTK default usually, but check)
        # PyVista/VTK use (nx, ny, nz) ordering usually if x,y,z passed.
        # But our arrays are (nz, ny, nx). 
        # Flattening 'F' (column-major) on transposed array?
        # PyVista format: (nx, ny, nz) points.
        
        # Robust way: ravel with correct order.
        # If we passed x,y,z arrays of length nx, ny, nz.
        # grid.points are ordered x changes fastest, then y, then z.
        # Our data u is (nz, ny, nx). 
        # So we need to flatten such that x changes fastest. 
        # That is simple u.flatten() IF u was (nz, ny, nx)? 
        # No, standard C flatten is last index fastest.
        # (nz, ny, nx) -> last index is x. 
        # So u.flatten() iterates x, then y, then z. This matches x-fastest!
        # Wait: (0,0,0), (0,0,1)... no.
        # u[k, j, i]. Flatten iterates i first?
        # C-order (default): last index changes fastest.
        # u[0,0,0], u[0,0,1] ... u[0,0,nx-1], u[0,1,0]...
        # So x changes fastest, then y, then z.
        # This matches VTK StructuredGrid ordering if built from x, y, z vectors.
        
        grid.point_data["u"] = u.flatten()
        grid.point_data["v"] = v.flatten()
        grid.point_data["w"] = w.flatten()
        grid.point_data["p"] = p.flatten()
        grid.point_data["nu_t"] = nu_t.flatten()
        
        # Compute magnitude
        grid.point_data["velocity_magnitude"] = np.sqrt(u.flatten()**2 + v.flatten()**2 + w.flatten()**2)
        grid.point_data["velocity"] = np.column_stack((u.flatten(), v.flatten(), w.flatten()))
        

        # 1. Q-Criterion Isosurfaces
        grid = grid.compute_derivative(scalars="velocity", gradient=True, faster=True)
        grad = grid["gradient"].reshape(-1, 3, 3)
        grad_t = grad.transpose(0, 2, 1)
        S = 0.5 * (grad + grad_t)
        Omega = 0.5 * (grad - grad_t)
        norm_S2 = np.sum(S**2, axis=(1, 2))
        norm_Omega2 = np.sum(Omega**2, axis=(1, 2))
        Q = 0.5 * (norm_Omega2 - norm_S2)
        grid.point_data["Q"] = Q
        
        q_max = np.max(Q)
        print(f"Max Q-criterion: {q_max:.2f}")
        
        # Isosurface at high Q (vortex cores)
        if q_max > 0.1:
            iso_val = 0.1 * q_max # Use 10% of max or specific value like previous script
            contours = grid.contour(isosurfaces=[iso_val], scalars="Q")
            
            p = pv.Plotter(off_screen=not args.show, window_size=(1920, 1080))
            p.set_background('white')
            
            # Step Geometry
            h_step = 0.4845
            dims_z_min, dims_z_max = np.min(z), np.max(z)
            step_box = pv.Box(bounds=[np.min(x), 0, 0, h_step, dims_z_min, dims_z_max])
            p.add_mesh(step_box, color='grey', opacity=1.0, show_edges=True)
            p.add_mesh(grid.outline(), color="black")
            
            p.add_mesh(contours, color="green", opacity=0.3, smooth_shading=True, label='Vortices (Q-crit)')
            
            # Streamlines (Matched to visualize_streamlines.py)
            # Seed 1: Near step corner
            seed_center_1 = (0.2, 0.4, dims_z_max - 0.1)
            streamlines_1 = grid.streamlines(
                vectors='velocity',
                source_radius=0.1,
                source_center=seed_center_1,
                n_points=100,
                max_time=20.0,
            )
            
            # Seed 2: Near sidewall
            seed_center_2 = (0.2, 0.5, 0.1 + dims_z_min)
            streamlines_2 = grid.streamlines(
                vectors='velocity',
                source_radius=0.1,
                source_center=seed_center_2,
                n_points=100,
                max_time=20.0,
            )
            
            if streamlines_1.n_points > 0:
                p.add_mesh(streamlines_1.tube(radius=0.005), cmap='coolwarm', scalars='velocity_magnitude')
            if streamlines_2.n_points > 0:
                p.add_mesh(streamlines_2.tube(radius=0.005), cmap='coolwarm', scalars='velocity_magnitude')

            # Camera Position (Matched)
            p.camera_position = [
                (9.35, 1.56, -0.44),  # Camera location
                (6.06, 0.50, 1.94),   # Focal point
                (-0.13, 0.96, 0.25)   # View up
            ]
            
            out_path = f"{args.output_dir}/les_matched_viz.png"
            p.show(screenshot=out_path)
            print(f"Saved matched visualization to {out_path}")


if __name__ == "__main__":
    main()
