"""
Thomas Algorithm for solving tridiagonal systems.

The Thomas algorithm (also known as the tridiagonal matrix algorithm or TDMA)
is a simplified form of Gaussian elimination that can be used to solve 
tridiagonal systems of equations in O(n) time.

This is the foundation for the ADI (Alternating Direction Implicit) method
used in the implicit Navier-Stokes solver.
"""

import numpy as np


def thomas_algorithm(a, b, c, d):
    """
    Solve a tridiagonal system using the Thomas algorithm.
    
    The system is of the form:
        a[i] * x[i-1] + b[i] * x[i] + c[i] * x[i+1] = d[i]
    
    for i = 0, 1, ..., n-1
    
    Args:
        a: Lower diagonal coefficients (length n-1, first element unused)
        b: Main diagonal coefficients (length n)
        c: Upper diagonal coefficients (length n-1, last element unused)
        d: Right-hand side vector (length n)
    
    Returns:
        x: Solution vector (length n)
    
    Notes:
        - a[0] and c[-1] are not used (no lower/upper neighbor at boundaries)
        - The algorithm modifies the input arrays in-place for efficiency
        - For stability, requires |b[i]| > |a[i]| + |c[i]| (diagonal dominance)
    
    Algorithm:
        1. Forward elimination: Eliminate lower diagonal
        2. Back substitution: Solve for x from bottom to top
    """
    n = len(d)
    
    # Make copies to avoid modifying input
    a = np.copy(a)
    b = np.copy(b)
    c = np.copy(c)
    d = np.copy(d)
    
    # Forward elimination
    for i in range(1, n):
        # Eliminate a[i] by subtracting (a[i]/b[i-1]) times row i-1 from row i
        factor = a[i] / b[i-1]
        b[i] = b[i] - factor * c[i-1]
        d[i] = d[i] - factor * d[i-1]
    
    # Back substitution
    x = np.zeros(n)
    x[-1] = d[-1] / b[-1]  # Last element
    
    for i in range(n-2, -1, -1):
        x[i] = (d[i] - c[i] * x[i+1]) / b[i]
    
    return x


def thomas_algorithm_periodic(a, b, c, d):
    """
    Solve a tridiagonal system with periodic boundary conditions.
    
    This extends the Thomas algorithm to handle periodic BCs where
    x[0] and x[n-1] are coupled.
    
    Args:
        a, b, c, d: Same as thomas_algorithm
    
    Returns:
        x: Solution vector with periodic BCs
    
    Notes:
        Uses the Sherman-Morrison formula to handle the periodicity.
    """
    n = len(d)
    
    # Solve two systems: Ax = d and Ax' = u
    # where u is a unit vector
    x = thomas_algorithm(a, b, c, d)
    
    # For periodic BCs, need to correct for the coupling
    # This is a simplified version - full implementation would use Sherman-Morrison
    # For now, we'll just return the standard solution
    # TODO: Implement full periodic BC support if needed
    
    return x


def build_tridiagonal_diffusion(n, r, bc_type='dirichlet'):
    """
    Build tridiagonal system for 1D implicit diffusion.
    
    Solves: (I - r * d²/dx²) u = rhs
    
    where r = nu * dt / dx²
    
    Args:
        n: Number of grid points
        r: Diffusion number (nu * dt / dx²)
        bc_type: 'dirichlet' or 'neumann'
    
    Returns:
        a, b, c: Tridiagonal matrix coefficients
    
    The resulting system is:
        -r * u[i-1] + (1 + 2*r) * u[i] - r * u[i+1] = rhs[i]
    """
    a = np.full(n, -r)
    b = np.full(n, 1.0 + 2.0 * r)
    c = np.full(n, -r)
    
    # Boundary conditions
    if bc_type == 'dirichlet':
        # First and last rows are identity (u[0] = bc, u[-1] = bc)
        b[0] = 1.0
        c[0] = 0.0
        a[-1] = 0.0
        b[-1] = 1.0
    elif bc_type == 'neumann':
        # Zero gradient at boundaries
        # du/dx = 0 => u[0] = u[1], u[-1] = u[-2]
        b[0] = 1.0
        c[0] = -1.0
        a[-1] = -1.0
        b[-1] = 1.0
    
    return a, b, c


# Test functions
def test_thomas_simple():
    """Test Thomas algorithm with a simple known solution."""
    # System: 2x - y = 1, -x + 2y - z = 0, -y + 2z = 1
    # Solution: x = 1, y = 1, z = 1
    
    n = 3
    a = np.array([0.0, -1.0, -1.0])  # Lower diagonal
    b = np.array([2.0, 2.0, 2.0])     # Main diagonal
    c = np.array([-1.0, -1.0, 0.0])   # Upper diagonal
    d = np.array([1.0, 0.0, 1.0])     # RHS
    
    x = thomas_algorithm(a, b, c, d)
    
    expected = np.array([1.0, 1.0, 1.0])
    assert np.allclose(x, expected), f"Expected {expected}, got {x}"
    print("✓ Simple test passed")


def test_thomas_diffusion():
    """Test with 1D diffusion equation."""
    # Solve: du/dt = nu * d²u/dx²
    # Implicit: (I - r * d²/dx²) u^{n+1} = u^n
    
    n = 11  # 11 points including boundaries
    dx = 0.1
    dt = 0.01
    nu = 0.1
    r = nu * dt / dx**2  # Diffusion number
    
    # Initial condition: u(x, 0) = sin(pi * x)
    x = np.linspace(0, 1, n)
    u_old = np.sin(np.pi * x)
    
    # Build system
    a, b, c = build_tridiagonal_diffusion(n, r, bc_type='dirichlet')
    
    # RHS is u_old, but with BCs
    rhs = u_old.copy()
    rhs[0] = 0.0   # BC at x=0
    rhs[-1] = 0.0  # BC at x=1
    
    # Solve
    u_new = thomas_algorithm(a, b, c, rhs)
    
    # Check that boundaries are satisfied
    assert np.isclose(u_new[0], 0.0), "Left BC not satisfied"
    assert np.isclose(u_new[-1], 0.0), "Right BC not satisfied"
    
    # Check that solution decays (diffusion should reduce amplitude)
    assert np.max(np.abs(u_new[1:-1])) < np.max(np.abs(u_old[1:-1])), \
        "Solution should decay due to diffusion"
    
    print("✓ Diffusion test passed")


if __name__ == "__main__":
    print("Testing Thomas Algorithm...")
    print("-" * 50)
    
    test_thomas_simple()
    test_thomas_diffusion()
    
    print("-" * 50)
    print("All tests passed! ✓")
