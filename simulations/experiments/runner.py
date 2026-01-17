
import argparse
import sys
import os
import time

# Ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
sim_dir = os.path.dirname(os.path.dirname(current_dir))
if sim_dir not in sys.path:
    sys.path.append(sim_dir)

from simulations.common.solvers import get_solver, SimulationConfig, SOLVER_REGISTRY
from simulations.common.plotting import plot_velocity_magnitude

def main():
    parser = argparse.ArgumentParser(description="Generic Cavity Flow Experiment Runner")
    parser.add_argument("--solver", type=str, required=True, choices=list(SOLVER_REGISTRY.keys()), 
                        help="Which solver implementation to use")
    parser.add_argument("--re", type=float, default=100.0, help="Reynolds Number")
    parser.add_argument("--nx", type=int, default=41, help="Grid points")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step")
    parser.add_argument("--t-end", type=float, default=2.0, help="Physical end time")
    
    # Default output structured in results folder
    default_dir = os.path.join(sim_dir, "experiments/results/single_runs")
    default_file = os.path.join(default_dir, "latest_run.png")
    parser.add_argument("--output", type=str, default=default_file, help="Output plot filename")
    parser.add_argument("--verbose", action="store_true", help="Print debug info")
    
    args = parser.parse_args()
    
    print(f"--- Experiment: Re={args.re}, Solver={args.solver} ---")
    
    # 1. Configure
    config = SimulationConfig(
        nx=args.nx,
        re=args.re,
        dt=args.dt,
        t_end=args.t_end,
        verbose=args.verbose
    )
    
    # 2. Get Solver
    run_solver = get_solver(args.solver)
    
    # 3. Execute
    print(f"Running simulation for {args.t_end} physical seconds...")
    start_t = time.time()
    result = run_solver(config)
    wall_time = time.time() - start_t
    
    print(f"Done! executed {result.steps} steps in {wall_time:.2f}s (Speed: {result.steps/wall_time:.1f} steps/s)")
    print(f"Physical time reached: {result.t_end:.4f}")
    
    # 4. Visualize
    plot_velocity_magnitude(result, config, args.output)

if __name__ == "__main__":
    main()
