
import argparse
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# Ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
sim_dir = os.path.dirname(os.path.dirname(current_dir))
if sim_dir not in sys.path:
    sys.path.append(sim_dir)

from simulations.common.solvers import get_solver, SimulationConfig, SOLVER_REGISTRY

# --- Ghia et al. (1982) Benchmark Data ---
# u-velocity along vertical centerline (x=0.5)
# Format: (y, u_Re100, u_Re400, u_Re1000)
GHIA_DATA = np.array([
    [0.0000, 0.00000, 0.00000, 0.00000],
    [0.0547, -0.03717, -0.08186, -0.18109],
    [0.0625, -0.04192, -0.09266, -0.20196],
    [0.0703, -0.04775, -0.10338, -0.22223],
    [0.1016, -0.06434, -0.14612, -0.29730],
    [0.1719, -0.10150, -0.24299, -0.38289],
    [0.2813, -0.15662, -0.32726, -0.27805],
    [0.4531, -0.21090, -0.17119, -0.10649],
    [0.5000, -0.20581, -0.11477, -0.06080],
    [0.6172, -0.13641, 0.02135, 0.05702],
    [0.7344, 0.00332, 0.16256, 0.18719],
    [0.8516, 0.23151, 0.29093, 0.33399],
    [0.9531, 0.68717, 0.55892, 0.46604],
    [0.9609, 0.73722, 0.61756, 0.51117],
    [0.9688, 0.78871, 0.68439, 0.57492],
    [0.9766, 0.84123, 0.75837, 0.66016],
    [1.0000, 1.00000, 1.00000, 1.00000]
])

def run_benchmark(solver_name: str, re: float, nx: int, dt: float, t_end: float):
    print(f"\nRunning {solver_name} at Re={re}...")
    
    config = SimulationConfig(nx=nx, re=re, dt=dt, t_end=t_end)
    run_fn = get_solver(solver_name)
    
    start_t = time.time()
    result = run_fn(config)
    dur = time.time() - start_t
    print(f"  -> Finished in {dur:.2f}s ({result.steps} steps)")
    
    return result

def plot_ghia_comparison(results_map: dict, output_file: str):
    """
    Plots centerline velocities against Ghia data.
    results_map: { re_value: SimulationResult }
    """
    fig, ax = plt.subplots(figsize=(8, 10), dpi=150)
    
    # Plot Ghia Data (Points)
    # y is column 0
    y_ghia = GHIA_DATA[:, 0]
    # Re=100 is col 1, Re=400 is col 2, Re=1000 is col 3
    
    # Map Re to Ghia column index
    ghia_cols = {100: 1, 400: 2, 1000: 3}
    colors = {100: 'blue', 400: 'green', 1000: 'red'}
    
    for re, res in results_map.items():
        if re not in ghia_cols:
            continue
            
        color = colors[re]
        
        # Check for NaNs
        if np.isnan(res.u).any():
            print(f"WARNING: Simulation for Re={re} contains NaNs! Skipping plot.")
            # Plot a dummy line for legend indicating crash
            ax.plot([], [], '-', color=color, label=f"Sim Re={re} (CRASHED!)")
            continue
        
        # 1. Plot Simulation Data (Solid Line)
        nx = res.u.shape[1]
        center_x_idx = nx // 2
        
        # Extract vertical centerline profile
        # Note: Simulation domain is 0..2, but Ghia is normalized 0..1
        # We need to normalize our y-coordinates to 0..1 for plotting against Ghia
        # And usually u is normalized by lid velocity (which is 1.0 here)
        
        u_sim = res.u[:, center_x_idx]
        y_sim = np.linspace(0, 1.0, nx) # Normalized height 0..1
        
        ax.plot(u_sim, y_sim, '-', color=color, label=f"Sim Re={re}", linewidth=2, alpha=0.7)
        
        # 2. Plot Ghia Reference (Dots)
        u_ghia = GHIA_DATA[:, ghia_cols[re]]
        ax.plot(u_ghia, y_ghia, 'o', color=color, label=f"Ghia Re={re}", markersize=6, fillstyle='none', markeredgewidth=1.5)

    ax.set_title(f"Centerline Velocity Validation (Ghia 1982)")
    ax.set_xlabel("u-velocity (normalized)")
    ax.set_ylabel("y-position (normalized)")
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, 1.1)
    ax.set_ylim(0, 1.0)
    
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_file)
    print(f"Validation plot saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Ghia Benchmark Verification")
    parser.add_argument("--solver", type=str, default="jax_explicit", choices=list(SOLVER_REGISTRY.keys()))
    parser.add_argument("--re-list", type=str, default="100,400", help="Comma separated Re values to test")
    parser.add_argument("--nx", type=int, default=129, help="Grid size (recommended 129 for Re=1000)")
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--t-end", type=float, default=10.0, help="Run long enough to reach steady state")
    
    # Default output structured in results folder
    default_dir = os.path.join(sim_dir, "experiments/results/ghia_benchmark")
    default_file = os.path.join(default_dir, "ghia_validation.png")
    parser.add_argument("--output", type=str, default=default_file)
    
    args = parser.parse_args()
    
    re_values = [int(x) for x in args.re_list.split(",")]
    results = {}
    
    print(f"--- Ghia Benchmark: Solver={args.solver}, Grid={args.nx}x{args.nx} ---")
    
    for re in re_values:
        res = run_benchmark(args.solver, re, args.nx, args.dt, args.t_end)
        results[re] = res
        
    print("Generating validation plot...")
    plot_ghia_comparison(results, args.output)

if __name__ == "__main__":
    main()
