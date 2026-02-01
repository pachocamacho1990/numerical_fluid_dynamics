
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Visualize Channel Flow Results")
    parser.add_argument("--file", type=str, required=True, help="Path to channel_stats.npz")
    parser.add_argument("--re_tau", type=float, default=180.0)
    parser.add_argument("--output_dir", type=str, default="simulations/channel_flow_3d/outputs/viz")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading {args.file}...")
    data = np.load(args.file)
    mean_u = data['mean_u']
    y = data['y']
    
    # Physics
    u_tau = 1.0 # Solved dimensionless
    delta = 1.0 # Half height
    nu = 1.0 / args.re_tau
    
    # Check if simulation u_tau matches target
    # Wall gradient approximation
    du_dy_wall = (mean_u[1] - mean_u[0]) / (y[1] - y[0])
    calc_u_tau = np.sqrt(nu * du_dy_wall)
    print(f"Target Re_tau: {args.re_tau}")
    print(f"Approx Calculated u_tau: {calc_u_tau:.4f}")
    
    # Normalize
    y_plus = y * u_tau / nu
    u_plus = mean_u / u_tau
    
    # Analytical Law of the Wall
    # Viscous sublayer: u+ = y+
    y_viscous = np.linspace(0.1, 10, 100)
    u_viscous = y_viscous
    
    # Log law: u+ = 1/k ln(y+) + B
    kappa = 0.41
    B = 5.2
    y_log = np.linspace(10, args.re_tau, 100)
    u_log = (1.0/kappa) * np.log(y_log) + B
    
    plt.figure(figsize=(10, 6))
    plt.semilogx(y_plus, u_plus, 'bo-', label='LES Simulation')
    plt.semilogx(y_viscous, u_viscous, 'k--', label='Viscous Sublayer (u+=y+)')
    plt.semilogx(y_log, u_log, 'r--', label=f'Log Law (1/{kappa} ln y+ + {B})')
    
    plt.xlabel('y+')
    plt.ylabel('u+')
    plt.title(f'Mean Velocity Profile (Target Re_tau={args.re_tau})')
    plt.grid(True, which="both", ls="-")
    plt.legend()
    plt.xlim(0.5, args.re_tau * 1.5)
    
    output_path = f"{args.output_dir}/profile_retau{int(args.re_tau)}.png"
    plt.savefig(output_path)
    print(f"Saved user profile plot to {output_path}")

if __name__ == "__main__":
    main()
