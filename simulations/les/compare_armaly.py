#!/usr/bin/env python3
"""
Armaly Benchmark Comparison for LES

Extracts reattachment length from LES solution and compares with experimental data.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# Armaly et al. (1983) experimental data (Re vs X_r/h)
ARMALY_DATA = {
    'Re': [100, 200, 300, 400, 500, 600, 700, 800],
    'Xr_over_h': [3.0, 5.0, 6.0, 6.8, 7.5, 8.0, 8.5, 9.0]  # Approximate values
}

def find_reattachment_length(x, y, u, h_step=0.4845):
    """
    Find reattachment length by detecting where u=0 at the lower wall.
    
    Parameters
    ----------
    x, y : ndarray
        Grid coordinates
    u : ndarray
        Velocity field (nz, ny, nx)
    h_step : float
        Step height
        
    Returns
    -------
    x_r : float
        Reattachment point (x-coordinate)
    x_r_over_h : float
        Normalized reattachment length
    """
    # Take mid-plane in z
    nz = u.shape[0]
    z_mid = nz // 2
    u_midplane = u[z_mid, :, :]  # (ny, nx)
    
    # Find wall index (just above step)
    j_wall = np.searchsorted(y, h_step)
    if j_wall >= len(y):
        j_wall = 0
    
    u_wall = u_midplane[j_wall, :]
    
    # Find where flow reattaches (u changes from negative to positive)
    # Look only downstream of step (x > 0)
    i_step = np.searchsorted(x, 0.0)
    u_wall_downstream = u_wall[i_step:]
    x_downstream = x[i_step:]
    
    # Find zero crossing
    sign_changes = np.where(np.diff(np.sign(u_wall_downstream)))[0]
    
    if len(sign_changes) == 0:
        print("Warning: No reattachment detected!")
        return np.nan, np.nan
    
    # First zero crossing is the reattachment point
    i_reattach = sign_changes[0]
    x_r = x_downstream[i_reattach]
    x_r_over_h = x_r / h_step
    
    return x_r, x_r_over_h

def main():
    parser = argparse.ArgumentParser(description="Armaly Benchmark Comparison")
    parser.add_argument("--file", type=str, required=True, help="Solution .npz file")
    parser.add_argument("--re", type=float, default=5000, help="Reynolds number")
    parser.add_argument("--output_dir", type=str, default="simulations/les/outputs_long")
    args = parser.parse_args()
    
    print("="*60)
    print("ARMALY BENCHMARK COMPARISON")
    print("="*60)
    
    # Load solution
    print(f"\nLoading: {args.file}")
    data = np.load(args.file)
    u = data['u']
    x = data['x']
    y = data['y']
    
    # Extract reattachment length
    x_r, x_r_over_h = find_reattachment_length(x, y, u)
    
    print(f"\nResults:")
    print(f"  Reynolds Number:      Re = {args.re}")
    print(f"  Reattachment Point:   X_r = {x_r:.4f}")
    print(f"  Normalized Length:    X_r/h = {x_r_over_h:.2f}")
    
    # Interpolate Armaly data
    armaly_re = np.array(ARMALY_DATA['Re'])
    armaly_xr = np.array(ARMALY_DATA['Xr_over_h'])
    
    if args.re <= max(armaly_re) and args.re >= min(armaly_re):
        expected_xr = np.interp(args.re, armaly_re, armaly_xr)
        error = abs(x_r_over_h - expected_xr) / expected_xr * 100
        print(f"  Armaly Expected:      X_r/h ≈ {expected_xr:.2f}")
        print(f"  Relative Error:       {error:.1f}%")
    else:
        print(f"  (Re={args.re} outside Armaly range {min(armaly_re)}-{max(armaly_re)})")
    
    # Generate comparison plot
    plt.figure(figsize=(10, 6))
    plt.plot(armaly_re, armaly_xr, 'ko-', label='Armaly et al. (1983)', markersize=8)
    plt.plot(args.re, x_r_over_h, 'r^', label=f'LES (Re={args.re})', markersize=12)
    
    plt.xlabel('Reynolds Number (Re)')
    plt.ylabel('Reattachment Length (X_r / h)')
    plt.title('Backward-Facing Step: Reattachment Length vs Reynolds Number')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0, max(max(armaly_re), args.re) * 1.1)
    plt.ylim(0, max(max(armaly_xr), x_r_over_h) * 1.2)
    
    os.makedirs(args.output_dir, exist_ok=True)
    outpath = os.path.join(args.output_dir, "armaly_comparison.png")
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot: {outpath}")
    
    # Save results
    results = {
        'Re': args.re,
        'X_r': x_r,
        'X_r_over_h': x_r_over_h
    }
    np.savez(os.path.join(args.output_dir, "benchmark_results.npz"), **results)
    print("Saved results: benchmark_results.npz")

if __name__ == "__main__":
    main()
