#!/usr/bin/env python3
"""
DNS vs LES Comparison Analysis

Compares DNS and LES results for the turbulent BFS case:
- Reattachment length
- Velocity profiles at key locations
- Visual comparisons
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path

def find_reattachment_length(x, y, u, h_step=0.4845):
    """
    Find reattachment length by detecting where u=0 at the lower wall.
    
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
    
    # Find reattachment (u changes from negative to positive)
    i_step = np.searchsorted(x, 0.0)
    u_wall_downstream = u_wall[i_step:]
    x_downstream = x[i_step:]
    
    # Find zero crossing
    sign_changes = np.where(np.diff(np.sign(u_wall_downstream)))[0]
    
    if len(sign_changes) == 0:
        print("Warning: No reattachment detected!")
        return np.nan, np.nan
    
    # First zero crossing
    i_reattach = sign_changes[0]
    
    # Linear interpolation for more accuracy
    u1 = u_wall_downstream[i_reattach]
    u2 = u_wall_downstream[i_reattach + 1]
    x1 = x_downstream[i_reattach]
    x2 = x_downstream[i_reattach + 1]
    
    x_r = x1 - u1 * (x2 - x1) / (u2 - u1)
    x_r_over_h = x_r / h_step
    
    return x_r, x_r_over_h

def extract_profile(x, y, u_field, x_location):
    """Extract velocity profile at given x location."""
    nz = u_field.shape[0]
    z_mid = nz // 2
    u_midplane = u_field[z_mid, :, :]
    
    # Find closest x index
    i_loc = np.argmin(np.abs(x - x_location))
    
    return y, u_midplane[:, i_loc]

def compare_dns_les(dns_file, les_file, output_dir):
    """Main comparison function."""
    
    print("="*70)
    print("DNS vs LES COMPARISON ANALYSIS")
    print("="*70)
    
    # Load data
    print(f"\nLoading DNS: {dns_file}")
    dns_data = np.load(dns_file)
    u_dns = dns_data['u']
    x_dns = dns_data['x'] if 'x' in dns_data else None
    y_dns = dns_data['y'] if 'y' in dns_data else None
    
    # For DNS, might need to reconstruct grid
    if x_dns is None:
        nx, ny, nz = 301, 76, 76
        dx = dns_data['dx']
        dy = dns_data['dy']
        x_dns = np.arange(nx) * dx
        y_dns = np.arange(ny) * dy
    
    print(f"Loading LES: {les_file}")
    les_data = np.load(les_file)
    u_les = les_data['u']
    x_les = les_data['x']
    y_les = les_data['y']
    
    print(f"DNS grid: {u_dns.shape}")
    print(f"LES grid: {u_les.shape}")
    
    # 1. Reattachment Length
    print("\n" + "-"*70)
    print("REATTACHMENT LENGTH COMPARISON")
    print("-"*70)
    
    x_r_dns, xr_h_dns = find_reattachment_length(x_dns, y_dns, u_dns)
    x_r_les, xr_h_les = find_reattachment_length(x_les, y_les, u_les)
    
    print(f"DNS: X_r = {x_r_dns:.4f}, X_r/h = {xr_h_dns:.2f}")
    print(f"LES: X_r = {x_r_les:.4f}, X_r/h = {xr_h_les:.2f}")
    
    if not np.isnan(xr_h_dns) and not np.isnan(xr_h_les):
        error = abs(xr_h_les - xr_h_dns) / xr_h_dns * 100
        print(f"Relative Error: {error:.1f}%")
    
    # 2. Velocity Profiles
    print("\n" + "-"*70)
    print("VELOCITY PROFILE COMPARISON")
    print("-"*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Compare profiles at x/h = 4, 7, 10
    h_step = 0.4845
    x_locations = [4*h_step, 7*h_step, 10*h_step]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, x_loc in enumerate(x_locations):
        ax = axes[idx]
        
        # Extract profiles
        y_dns_prof, u_dns_prof = extract_profile(x_dns, y_dns, u_dns, x_loc)
        y_les_prof, u_les_prof = extract_profile(x_les, y_les, u_les, x_loc)
        
        # Plot
        ax.plot(u_dns_prof, y_dns_prof, 'k-', linewidth=2, label='DNS')
        ax.plot(u_les_prof, y_les_prof, 'r--', linewidth=2, label='LES')
        
        ax.set_xlabel('u velocity')
        ax.set_ylabel('y')
        ax.set_title(f'x/h = {x_loc/h_step:.1f}')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'velocity_profiles_comparison.png'), dpi=150)
    print(f"Saved: velocity_profiles_comparison.png")
    
    # 3. Summary Report
    report_path = os.path.join(output_dir, 'comparison_report.txt')
    with open(report_path, 'w') as f:
        f.write("DNS vs LES Comparison Report\n")
        f.write("="*70 + "\n\n")
        f.write(f"DNS File: {dns_file}\n")
        f.write(f"LES File: {les_file}\n\n")
        f.write(f"DNS Grid: {u_dns.shape}\n")
        f.write(f"LES Grid: {u_les.shape}\n\n")
        f.write("Reattachment Length:\n")
        f.write(f"  DNS: X_r/h = {xr_h_dns:.2f}\n")
        f.write(f"  LES: X_r/h = {xr_h_les:.2f}\n")
        if not np.isnan(xr_h_dns) and not np.isnan(xr_h_les):
            f.write(f"  Error: {abs(xr_h_les - xr_h_dns)/xr_h_dns*100:.1f}%\n")
    
    print(f"Saved: comparison_report.txt")
    
    # 4. Velocity Magnitude Comparison
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # DNS
    nz_dns = u_dns.shape[0]
    u_dns_slice = u_dns[nz_dns//2, :, :]
    im1 = axes[0].imshow(u_dns_slice, extent=[x_dns.min(), x_dns.max(), y_dns.min(), y_dns.max()],
                         origin='lower', cmap='jet', aspect='auto')
    axes[0].set_title('DNS: Velocity Magnitude (z-midplane)')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0])
    
    # LES
    nz_les = u_les.shape[0]
    u_les_slice = u_les[nz_les//2, :, :]
    im2 = axes[1].imshow(u_les_slice, extent=[x_les.min(), x_les.max(), y_les.min(), y_les.max()],
                         origin='lower', cmap='jet', aspect='auto')
    axes[1].set_title('LES: Velocity Magnitude (z-midplane)')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'velocity_field_comparison.png'), dpi=150)
    print(f"Saved: velocity_field_comparison.png")
    
    print("\n" + "="*70)
    print("Comparison complete! Check output directory for results.")
    print("="*70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dns_file", required=True, help="DNS checkpoint .npz file")
    parser.add_argument("--les_file", required=True, help="LES checkpoint .npz file")
    parser.add_argument("--output_dir", default="simulations/les/comparison_results")
    
    args = parser.parse_args()
    compare_dns_les(args.dns_file, args.les_file, args.output_dir)
