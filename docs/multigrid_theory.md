# Multigrid Methods for Pressure Poisson Equation

## 1. Introduction

### 1.1 The Pressure Poisson Problem

In the fractional step method for incompressible Navier-Stokes, we solve:

$$\nabla^2 p = \frac{\rho}{\Delta t} \nabla \cdot \mathbf{u}^*$$

Discretized on a uniform grid with spacing $h$:

$$\frac{p_{i+1,j} - 2p_{i,j} + p_{i-1,j}}{h^2} + \frac{p_{i,j+1} - 2p_{i,j} + p_{i,j-1}}{h^2} = b_{i,j}$$

This is a sparse linear system $Ap = b$ where $A$ is the discrete Laplacian matrix.

### 1.2 Why Jacobi/Gauss-Seidel Are Slow

Classical iterative methods update:

$$p_{i,j}^{(k+1)} = \frac{1}{4}\left(p_{i+1,j}^{(k)} + p_{i-1,j}^{(k)} + p_{i,j+1}^{(k)} + p_{i,j-1}^{(k)} - h^2 b_{i,j}\right)$$

**Error Analysis:** The error $e^{(k)} = p^{(k)} - p_{exact}$ can be decomposed into Fourier modes:

$$e^{(k)}(x,y) = \sum_{m,n} \hat{e}_{m,n}^{(k)} \sin\left(\frac{m\pi x}{L}\right)\sin\left(\frac{n\pi y}{L}\right)$$

The Jacobi amplification factor for mode $(m,n)$ is:

$$\lambda_{m,n} = \frac{1}{2}\left(\cos\frac{m\pi}{N} + \cos\frac{n\pi}{N}\right)$$

| Mode Type | Frequency | $\lambda$ | Convergence |
|-----------|-----------|-----------|-------------|
| High ($m,n \approx N$) | Oscillatory | $\approx 0$ | **Fast** |
| Low ($m,n \approx 1$) | Smooth | $\approx 1 - O(1/N^2)$ | **Very slow** |

**Problem:** Low-frequency (smooth) errors decay as $(1 - O(1/N^2))^k$, requiring $O(N^2)$ iterations.

---

## 2. The Multigrid Principle

### 2.1 Key Insight

A smooth error on a fine grid appears oscillatory on a coarser grid:

```
Fine Grid (h):     ∿∿∿∿∿∿∿∿  (low frequency → slow)
                        ↓
Coarse Grid (2h):  ⋀⋁⋀⋁⋀⋁⋀⋁  (appears high frequency → fast!)
```

By transferring the error to progressively coarser grids, we can eliminate all frequency components efficiently.

### 2.2 Two-Grid Algorithm

1. **Pre-smooth** on fine grid $\Omega^h$: Apply $\nu_1$ relaxation sweeps
   $$p^h \leftarrow S^{\nu_1}(p^h, b^h)$$

2. **Compute residual**:
   $$r^h = b^h - A^h p^h$$

3. **Restrict** residual to coarse grid $\Omega^{2h}$:
   $$r^{2h} = I_h^{2h} r^h$$

4. **Solve coarse problem** for error $e^{2h}$:
   $$A^{2h} e^{2h} = r^{2h}$$

5. **Prolong** correction to fine grid:
   $$e^h = I_{2h}^h e^{2h}$$

6. **Correct**:
   $$p^h \leftarrow p^h + e^h$$

7. **Post-smooth**: Apply $\nu_2$ relaxation sweeps
   $$p^h \leftarrow S^{\nu_2}(p^h, b^h)$$

### 2.3 V-Cycle (Recursive Extension)

Apply the two-grid algorithm recursively: at step 4, instead of solving exactly, apply the same procedure:

```
Level 0 (finest)  ●────────────────────●
                   ╲                  ╱
                    ╲ Restrict      ╱ Prolong + Correct
                     ╲            ╱
Level 1              ●──────────●
                      ╲        ╱
                       ╲      ╱
                        ╲    ╱
Level L (coarsest)       ●      ← Direct solve or many iterations
```

---

## 3. Transfer Operators

### 3.1 Restriction (Fine → Coarse): $I_h^{2h}$

**Full Weighting** (recommended):

$$r_{i,j}^{2h} = \frac{1}{16}\begin{bmatrix} 1 & 2 & 1 \\ 2 & 4 & 2 \\ 1 & 2 & 1 \end{bmatrix} * r^h \bigg|_{2i,2j}$$

In code:
```python
r_coarse[j, i] = (1 * r_fine[2j-1, 2i-1] + 2 * r_fine[2j-1, 2i] + 1 * r_fine[2j-1, 2i+1] +
                  2 * r_fine[2j,   2i-1] + 4 * r_fine[2j,   2i] + 2 * r_fine[2j,   2i+1] +
                  1 * r_fine[2j+1, 2i-1] + 2 * r_fine[2j+1, 2i] + 1 * r_fine[2j+1, 2i+1]) / 16
```

### 3.2 Prolongation (Coarse → Fine): $I_{2h}^h$

**Bilinear Interpolation**:

At coarse points (direct injection):
$$e_{2j,2i}^h = e_{j,i}^{2h}$$

At horizontal midpoints:
$$e_{2j,2i+1}^h = \frac{1}{2}(e_{j,i}^{2h} + e_{j,i+1}^{2h})$$

At vertical midpoints:
$$e_{2j+1,2i}^h = \frac{1}{2}(e_{j,i}^{2h} + e_{j+1,i}^{2h})$$

At cell centers:
$$e_{2j+1,2i+1}^h = \frac{1}{4}(e_{j,i}^{2h} + e_{j+1,i}^{2h} + e_{j,i+1}^{2h} + e_{j+1,i+1}^{2h})$$

---

## 4. Smoothers

### 4.1 Weighted Jacobi

$$p_{i,j}^{new} = (1-\omega)p_{i,j}^{old} + \frac{\omega}{4}(p_{i+1,j} + p_{i-1,j} + p_{i,j+1} + p_{i,j-1} - h^2 b_{i,j})$$

Optimal $\omega = 2/3$ for multigrid smoothing (not convergence!).

### 4.2 Red-Black Gauss-Seidel (Parallel-Friendly)

Checkerboard coloring allows parallel updates:

1. Update all "red" points (where $i+j$ is even)
2. Update all "black" points (where $i+j$ is odd)

Each phase is fully parallel and can be vectorized in JAX.

---

## 5. Coarse Grid Operator

### 5.1 Galerkin Condition

For theoretical optimality:
$$A^{2h} = I_h^{2h} A^h I_{2h}^h$$

### 5.2 Geometric Coarsening (Simpler)

Re-discretize the Laplacian directly on the coarse grid with spacing $2h$:
$$A^{2h} = \text{Laplacian}_{2h}$$

This is simpler to implement and works well for uniform grids.

---

## 6. Complexity Analysis

### 6.1 Work per Level

| Level $\ell$ | Grid Size | Work |
|--------------|-----------|------|
| 0 | $N^2$ | $O(N^2)$ |
| 1 | $N^2/4$ | $O(N^2/4)$ |
| 2 | $N^2/16$ | $O(N^2/16)$ |
| $\vdots$ | $\vdots$ | $\vdots$ |
| $L$ | $O(1)$ | $O(1)$ |

Total work per V-cycle:
$$W = O(N^2) \sum_{k=0}^{L} 4^{-k} = O(N^2) \cdot \frac{1}{1-1/4} = O(N^2)$$

### 6.2 Iterations Required

Multigrid achieves a **grid-independent convergence rate**:
$$\|e^{(k)}\| \leq \rho^k \|e^{(0)}\|$$

where $\rho \approx 0.1$ (reduces error by 10× per cycle).

| Tolerance | Jacobi Iterations | V-Cycles |
|-----------|-------------------|----------|
| $10^{-6}$ | $O(N^2)$ | **6-7** |
| $10^{-10}$ | $O(N^2)$ | **10-11** |

---

## 7. Handling Complex Geometry

### 7.1 The Step Problem

The backward-facing step has solid regions where $p$ is undefined:

```
┌──────────────────────────────┐
│ Fluid region                 │
│                              │
├────┐                         │
│████│  Step (solid)           │
│████│                         │
└────┴─────────────────────────┘
```

### 7.2 Masked Multigrid

**Approach 1: Skip solid cells**
- Mask out solid cells in all operations
- Boundary conditions at solid-fluid interface

**Approach 2: Ghost cells**
- Extend solution into solid with appropriate values
- Apply Neumann BC: $\partial p / \partial n = 0$ at walls

For restriction/prolongation, weight by valid fluid cells only.

---

## 8. JAX Implementation Considerations

### 8.1 Recursive V-Cycle → Iterative

JAX works best with fixed shapes and `lax.scan`. Unroll the V-cycle:

```python
def v_cycle_iterative(p, b, grids):
    # Descending: pre-smooth and restrict
    residuals = []
    for level in range(n_levels - 1):
        p[level] = smooth(p[level], b[level])
        r = compute_residual(p[level], b[level])
        residuals.append(r)
        b[level + 1] = restrict(r)
    
    # Coarsest level solve
    p[-1] = coarse_solve(b[-1])
    
    # Ascending: prolong and post-smooth
    for level in range(n_levels - 2, -1, -1):
        e = prolong(p[level + 1])
        p[level] = p[level] + e
        p[level] = smooth(p[level], b[level])
    
    return p[0]
```

### 8.2 Fixed Grid Hierarchy

Pre-compute grid sizes:
```python
grid_sizes = [(nx, ny)]
while min(grid_sizes[-1]) > 4:
    nx, ny = grid_sizes[-1]
    grid_sizes.append(((nx + 1) // 2, (ny + 1) // 2))
```

### 8.3 Vectorized Operations

All operations are stencil-based and fully vectorizable:
```python
@jit
def restrict(fine):
    # Full weighting using slicing
    coarse = (fine[0:-2:2, 0:-2:2] + 2*fine[0:-2:2, 1:-1:2] + fine[0:-2:2, 2::2] +
              2*fine[1:-1:2, 0:-2:2] + 4*fine[1:-1:2, 1:-1:2] + 2*fine[1:-1:2, 2::2] +
              fine[2::2, 0:-2:2] + 2*fine[2::2, 1:-1:2] + fine[2::2, 2::2]) / 16
    return coarse
```

---

## 9. Expected Performance

### 9.1 Comparison

| Method | Iterations | Time per Iter | Total Time |
|--------|------------|---------------|------------|
| Jacobi (50 iter) | 50 | 1.2 ms | 60 ms |
| V-Cycle (3 cycles) | 3 | 2-3 ms | 6-9 ms |

**Expected speedup: 7-10×** for the pressure solve.

### 9.2 When Multigrid Helps Most

- Large grids (gains increase with N)
- Tight tolerances required
- Many time steps (amortize setup cost)

---

## 10. References

1. Trottenberg, U., Oosterlee, C., & Schüller, A. (2001). *Multigrid*. Academic Press.
2. Briggs, W. L., Henson, V. E., & McCormick, S. F. (2000). *A Multigrid Tutorial*. SIAM.
3. Wesseling, P. (1992). *An Introduction to Multigrid Methods*. Wiley.
