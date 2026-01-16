"""
Compare explicit vs implicit cavity flow solutions.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Load solutions
explicit = np.load('simulations/cavity_flow/outputs/baseline/solution_numpy.npz')
implicit = np.load('simulations/cavity_flow_implicit/outputs/baseline/solution_implicit.npz')

u_exp = explicit['u']
v_exp = explicit['v']
p_exp = explicit['p']

u_imp = implicit['u']
v_imp = implicit['v']
p_imp = implicit['p']

nx = u_exp.shape[1]  # Get from array shape
L = 2.0  # Standard cavity size

x = np.linspace(0, L, nx)
y = np.linspace(0, L, nx)
X, Y = np.meshgrid(x, y)

# Create comparison plot
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Row 1: Explicit
ax = axes[0, 0]
vel_mag_exp = np.sqrt(u_exp**2 + v_exp**2)
cf = ax.contourf(X, Y, vel_mag_exp, levels=20, cmap=cm.viridis)
ax.streamplot(X, Y, u_exp, v_exp, color='white', linewidth=0.5, density=1.5)
ax.set_title('Explicit: Velocity Magnitude', fontsize=12, fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')
plt.colorbar(cf, ax=ax)

ax = axes[0, 1]
cf = ax.contourf(X, Y, p_exp, levels=20, cmap=cm.RdBu_r)
ax.set_title('Explicit: Pressure', fontsize=12, fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')
plt.colorbar(cf, ax=ax)

ax = axes[0, 2]
ax.plot(u_exp[:, nx//2], y, 'b-', label='u at x=L/2', linewidth=2)
ax.plot(v_exp[nx//2, :], x, 'r-', label='v at y=L/2', linewidth=2)
ax.set_title('Explicit: Centerline Velocities', fontsize=12, fontweight='bold')
ax.set_xlabel('Velocity')
ax.set_ylabel('Position')
ax.legend()
ax.grid(True, alpha=0.3)

# Row 2: Implicit
ax = axes[1, 0]
vel_mag_imp = np.sqrt(u_imp**2 + v_imp**2)
cf = ax.contourf(X, Y, vel_mag_imp, levels=20, cmap=cm.viridis)
ax.streamplot(X, Y, u_imp, v_imp, color='white', linewidth=0.5, density=1.5)
ax.set_title('Implicit: Velocity Magnitude', fontsize=12, fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')
plt.colorbar(cf, ax=ax)

ax = axes[1, 1]
cf = ax.contourf(X, Y, p_imp, levels=20, cmap=cm.RdBu_r)
ax.set_title('Implicit: Pressure', fontsize=12, fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')
plt.colorbar(cf, ax=ax)

ax = axes[1, 2]
ax.plot(u_imp[:, nx//2], y, 'b-', label='u at x=L/2', linewidth=2)
ax.plot(v_imp[nx//2, :], x, 'r-', label='v at y=L/2', linewidth=2)
ax.set_title('Implicit: Centerline Velocities', fontsize=12, fontweight='bold')
ax.set_xlabel('Velocity')
ax.set_ylabel('Position')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('simulations/cavity_flow_implicit/plots/explicit_vs_implicit_comparison.png', 
            dpi=150, bbox_inches='tight')
print("Comparison saved to simulations/cavity_flow_implicit/plots/explicit_vs_implicit_comparison.png")

# Compute differences
u_diff = np.abs(u_imp - u_exp)
v_diff = np.abs(v_imp - v_exp)
p_diff = np.abs(p_imp - p_exp)

print(f"\nDifferences (Implicit - Explicit):")
print(f"  Max |Δu|: {np.max(u_diff):.6f}")
print(f"  Max |Δv|: {np.max(v_diff):.6f}")
print(f"  Max |Δp|: {np.max(p_diff):.6f}")
print(f"  RMS |Δu|: {np.sqrt(np.mean(u_diff**2)):.6f}")
print(f"  RMS |Δv|: {np.sqrt(np.mean(v_diff**2)):.6f}")
print(f"  RMS |Δp|: {np.sqrt(np.mean(p_diff**2)):.6f}")
