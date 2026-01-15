import numpy as np
import matplotlib.pyplot as plt
import os
import glob

import glob
import argparse

def main():
    parser = argparse.ArgumentParser(description="Compare Solutions")
    parser.add_argument("--output", default="solutions_comparison.png", help="Output filename")
    args = parser.parse_args()

    base_dir = "simulations/cavity_flow"
    files = sorted(glob.glob(os.path.join(base_dir, "solution_*.npz")))
    
    if not files:
        print("No solution files found.")
        return

    n_files = len(files)
    fig, axes = plt.subplots(1, n_files, figsize=(5 * n_files, 5), dpi=100)
    
    if n_files == 1:
        axes = [axes]
    
    if n_files == 1:
        axes = [axes]
    
    for i, fpath in enumerate(files):
        filename = os.path.basename(fpath)
        # Extract label from filename (solution_numpy.npz -> numpy, solution_jax_metal.npz -> jax_metal)
        label = filename.replace("solution_", "").replace(".npz", "")
        
        print(f"Loading {filename}...")
        data = np.load(fpath)
        u = data['u']
        v = data['v']
        p = data['p']
        
        # Grid for plotting (Dynamic)
        ny, nx = p.shape
        x = np.linspace(0, 2, nx)
        y = np.linspace(0, 2, ny)
        X, Y = np.meshgrid(x, y)
        
        ax = axes[i]
        # Contour plot of pressure
        contour = ax.contourf(X, Y, p, alpha=0.5, cmap='viridis')
        # Streamplot
        ax.streamplot(X, Y, u, v, color='k', density=1.0, linewidth=0.5, arrowsize=0.8)
        
        ax.set_title(label)
        ax.set_xlabel('X')
        if i == 0:
            ax.set_ylabel('Y')
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.set_aspect('equal')

    plt.tight_layout()
    output_file = os.path.join(base_dir, args.output)
    plt.savefig(output_file)
    print(f"Combined plot saved to {output_file}")

if __name__ == "__main__":
    main()
