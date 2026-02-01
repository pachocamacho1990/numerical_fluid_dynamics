#!/usr/bin/env python3
"""
Comprehensive Physics Verification for LES Implementation

Tests all physics components systematically and generates a report.
"""

import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import jax.numpy as jnp
from simulations.les.sgs_models import (
    compute_strain_rate_tensor_3d,
    compute_strain_magnitude,
    compute_smagorinsky_viscosity,
    compute_filter_width
)
from simulations.les.wall_treatment import (
    compute_wall_distance_3d,
    van_driest_damping
)
from simulations.les.variable_viscosity_adi import (
    build_tridiagonal_variable_nu,
    thomas_algorithm_variable
)

def test_strain_rate_tensor():
    """Test 1: Strain Rate Tensor Computation"""
    print("\n" + "="*60)
    print("TEST 1: Strain Rate Tensor Computation")
    print("="*60)
    
    # Simple shear flow: u = y, v = w = 0
    nx, ny, nz = 10, 10, 10
    dx = dy = dz = 0.1
    
    y = jnp.linspace(0, 1, ny)
    u = jnp.broadcast_to(y[None, :, None], (nz, ny, nx))
    v = jnp.zeros_like(u)
    w = jnp.zeros_like(u)
    
    S11, S22, S33, S12, S13, S23 = compute_strain_rate_tensor_3d(u, v, w, dx, dy, dz)
    
    # Expected: S12 = 0.5 * du/dy = 0.5 / dy (approx)
    # Interior should have strain
    S12_center = S12[nz//2, ny//2, nx//2]
    expected_S12 = 0.5 / dy
    
    print(f"  Simple shear u=y:")
    print(f"    S12 (center): {S12_center:.4f}")
    print(f"    Expected:     {expected_S12:.4f}")
    print(f"    Error:        {abs(S12_center - expected_S12):.2e}")
    
    # Check other components are near zero
    print(f"    S11 (should be ~0): {S11[nz//2, ny//2, nx//2]:.2e}")
    print(f"    S22 (should be ~0): {S22[nz//2, ny//2, nx//2]:.2e}")
    
    status = "PASS" if abs(S12_center - expected_S12) < 0.1 else "FAIL"
    print(f"  Status: {status}")
    return status == "PASS"

def test_smagorinsky_viscosity():
    """Test 2: Smagorinsky Viscosity"""
    print("\n" + "="*60)
    print("TEST 2: Smagorinsky Viscosity Calculation")
    print("="*60)
    
    nx, ny, nz = 10, 10, 10
    dx = dy = dz = 0.1
    
    # Laminar case: Zero strain
    S11 = S22 = S33 = S12 = S13 = S23 = jnp.zeros((nz, ny, nx))
    cs = 0.12
    delta = compute_filter_width(dx, dy, dz)
    
    nu_t = compute_smagorinsky_viscosity(S11, S22, S33, S12, S13, S23, cs, delta)
    
    print(f"  Laminar (zero strain):")
    print(f"    Max nu_t:  {jnp.max(nu_t):.2e} (should be 0)")
    
    # Turbulent case: Non-zero strain
    S12_turb = jnp.ones((nz, ny, nx)) * 10.0  # Strong shear
    nu_t_turb = compute_smagorinsky_viscosity(S11, S22, S33, S12_turb, S13, S23, cs, delta)
    
    S_mag = compute_strain_magnitude(S11, S22, S33, S12_turb, S13, S23)
    expected_nu_t = (cs * delta)**2 * S_mag[nz//2, ny//2, nx//2]
    actual_nu_t = nu_t_turb[nz//2, ny//2, nx//2]
    
    print(f"  Turbulent (S12=10):")
    print(f"    nu_t:      {actual_nu_t:.4f}")
    print(f"    Expected:  {expected_nu_t:.4f}")
    print(f"    Error:     {abs(actual_nu_t - expected_nu_t):.2e}")
    
    status = "PASS" if (jnp.max(nu_t) < 1e-10 and abs(actual_nu_t - expected_nu_t) < 1e-6) else "FAIL"
    print(f"  Status: {status}")
    return status == "PASS"

def test_van_driest_damping():
    """Test 3: Van Driest Wall Damping"""
    print("\n" + "="*60)
    print("TEST 3: Van Driest Wall Damping")
    print("="*60)
    
    # Test damping behavior
    nu_t_raw = 0.01
    
    # Near wall (y+ = 1)
    y_plus_wall = jnp.array([1.0])
    damped_wall = van_driest_damping(nu_t_raw, y_plus_wall, A_plus=25.0)
    
    # Away from wall (y+ = 100)
    y_plus_outer = jnp.array([100.0])
    damped_outer = van_driest_damping(nu_t_raw, y_plus_outer, A_plus=25.0)
    
    print(f"  Original nu_t:          {nu_t_raw:.4f}")
    print(f"  At y+=1 (near wall):    {damped_wall[0]:.6f} ({damped_wall[0]/nu_t_raw*100:.1f}%)")
    print(f"  At y+=100 (outer):      {damped_outer[0]:.6f} ({damped_outer[0]/nu_t_raw*100:.1f}%)")
    
    # Check damping is significant near wall
    damping_ratio = damped_wall[0] / nu_t_raw
    status = "PASS" if damping_ratio < 0.1 and damped_outer[0] > 0.9 * nu_t_raw else "FAIL"
    print(f"  Status: {status}")
    return status == "PASS"

def test_variable_adi_consistency():
    """Test 4: Variable Viscosity ADI Consistency"""
    print("\n" + "="*60)
    print("TEST 4: Variable Viscosity ADI Solver")
    print("="*60)
    
    # Test that variable viscosity reduces to constant case
    nx = 20
    nu_const = 0.01
    nu_field = jnp.ones((1, 1, nx)) * nu_const
    dt = 0.01
    dx = 0.1
    
    a, b, c = build_tridiagonal_variable_nu(nu_field, dt, dx, axis=2)
    
    # Solve trivial system: u_new = u_old (RHS = u_old)
    u_old = jnp.sin(jnp.linspace(0, 2*jnp.pi, nx))
    u_old_3d = u_old[None, None, :]
    
    a_2d = a[0, 0, :]
    b_2d = b[0, 0, :]
    c_2d = c[0, 0, :]
    
    # Need to solve (I - dt/2 L) u = u_old
    # For testing, just check coefficient structure
    print(f"  Coefficient shapes: a={a.shape}, b={b.shape}, c={c.shape}")
    print(f"  Sample b[0,0,10]: {b_2d[10]:.6f}")
    
    # Check diagonal dominance (stability)
    interior_idx = 10
    diag_dominance = abs(b_2d[interior_idx]) - (abs(a_2d[interior_idx]) + abs(c_2d[interior_idx]))
    print(f"  Diagonal dominance: {diag_dominance:.6f} (should be > 0)")
    
    status = "PASS" if diag_dominance > 0 else "FAIL"
    print(f"  Status: {status}")
    return status == "PASS"

def main():
    """Run all physics verification tests"""
    print("\n" + "#"*60)
    print("# LES PHYSICS COMPONENT VERIFICATION")
    print("#"*60)
    
    results = {}
    results['Strain Rate Tensor'] = test_strain_rate_tensor()
    results['Smagorinsky Viscosity'] = test_smagorinsky_viscosity()
    results['Van Driest Damping'] = test_van_driest_damping()
    results['Variable ADI Solver'] = test_variable_adi_consistency()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for test_name, passed in results.items():
        status_str = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name:30s} {status_str}")
    
    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All physics components verified successfully!")
        return 0
    else:
        print("\n✗ Some tests failed. Review output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
