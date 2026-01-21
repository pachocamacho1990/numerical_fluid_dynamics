# Theory: 3D Backward-Facing Step Flow

This document provides a comprehensive theoretical background for the 3D implicit solution of backward-facing step (BFS) flow. It extends the 2D formulation to account for sidewall effects and spanwise variations.

## 1. Why 3D Simulation?

### 1.1 Limitations of 2D Simulations

While 2D simulations are computationally efficient, they cannot capture:
- **Sidewall boundary layers** that develop along the channel walls
- **Secondary flows** in the $(y,z)$ plane induced by corner vortices
- **Spanwise variations** in reattachment length

### 1.2 Experimental Observations (Armaly et al.)

The Armaly et al. (1983) experiments revealed critical 3D effects:
- For $Re > 400$, 2D simulations **overpredict** the reattachment length
- The sidewall boundary layers thicken and interact with the primary recirculation
- At the channel centerline (mid-span), reattachment occurs earlier than 2D predictions

The discrepancy between 2D simulations and experiments grows with Reynolds number, making 3D simulation essential for accurate predictions at moderate-to-high Re.

---

## 2. Geometry & Domain

### 2.1 3D Domain Dimensions

We extend the 2D Armaly geometry by adding a spanwise dimension $z$:

| Parameter | Value | Description |
|-----------|-------|-------------|
| $H$ | 1.0 | Total channel height |
| $h$ | 0.4845 | Step height (ER = 1.94) |
| $h_{inlet}$ | $H - h = 0.5155$ | Inlet height |
| $L_{in}$ | $5h$ | Inlet length |
| $L_{out}$ | $30h$ | Outlet length |
| $W$ | $8h$ | **Spanwise width** (typical aspect ratio) |

### 2.2 Coordinate System

$$
\begin{aligned}
x &: \text{streamwise direction} \quad (x \in [-L_{in}, L_{out}]) \\
y &: \text{wall-normal direction} \quad (y \in [0, H]) \\
z &: \text{spanwise direction} \quad (z \in [0, W])
\end{aligned}
$$

### 2.3 Step Region (Solid)

The step occupies the region:
$$
\Omega_{solid} = \{ (x, y, z) : x < 0 \text{ and } y < h, \; \forall z \}
$$

We define a 3D mask:
$$
M(i,j,k) = \begin{cases} 
1 & \text{if } (x_i, y_j, z_k) \in \Omega_{fluid} \\
0 & \text{if } (x_i, y_j, z_k) \in \Omega_{solid}
\end{cases}
$$

---

## 3. Governing Equations

### 3.1 3D Incompressible Navier-Stokes

**Continuity:**
$$
\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} + \frac{\partial w}{\partial z} = 0
$$

**Momentum (u-component):**
$$
\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} + w \frac{\partial u}{\partial z} = -\frac{1}{\rho} \frac{\partial p}{\partial x} + \nu \nabla^2 u
$$

**Momentum (v-component):**
$$
\frac{\partial v}{\partial t} + u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} + w \frac{\partial v}{\partial z} = -\frac{1}{\rho} \frac{\partial p}{\partial y} + \nu \nabla^2 v
$$

**Momentum (w-component):**
$$
\frac{\partial w}{\partial t} + u \frac{\partial w}{\partial x} + v \frac{\partial w}{\partial y} + w \frac{\partial w}{\partial z} = -\frac{1}{\rho} \frac{\partial p}{\partial z} + \nu \nabla^2 w
$$

where the 3D Laplacian is:
$$
\nabla^2 \phi = \frac{\partial^2 \phi}{\partial x^2} + \frac{\partial^2 \phi}{\partial y^2} + \frac{\partial^2 \phi}{\partial z^2}
$$

### 3.2 Reynolds Number

Using the standard Armaly definition:
$$
Re_h = \frac{\bar{U}_{inlet} \cdot h}{\nu}
$$

where $\bar{U}_{inlet} = \frac{2}{3} U_{max}$ for a parabolic profile.

---

## 4. Numerical Method: 3D Implicit Fractional Step

We extend the 2D projection method to 3D, following Chorin's algorithm.

### 4.1 Algorithm Overview

```
For each time step n → n+1:

1. ADVECTION (Explicit Upwind)
   Compute advection terms for u, v, w using first-order upwind

2. PREDICTOR (Implicit Diffusion via 3D ADI)
   Solve: (I - α∇²) u* = RHS_u  using X→Y→Z sweeps
   Solve: (I - α∇²) v* = RHS_v  using X→Y→Z sweeps
   Solve: (I - α∇²) w* = RHS_w  using X→Y→Z sweeps

3. PRESSURE POISSON (3D Iterative)
   Solve: ∇²p = (ρ/Δt) ∇·u*

4. CORRECTOR (Projection)
   u^{n+1} = u* - (Δt/ρ) ∂p/∂x
   v^{n+1} = v* - (Δt/ρ) ∂p/∂y
   w^{n+1} = w* - (Δt/ρ) ∂p/∂z

5. BOUNDARY CONDITIONS
   Apply mask, walls, sidewalls, inlet, outlet
```

---

## 5. 3D ADI Method

### 5.1 Approximate Factorization

The 3D Helmholtz equation $(I - \alpha \nabla^2) \phi = f$ is factorized:
$$
(I - \alpha \partial_{xx})(I - \alpha \partial_{yy})(I - \alpha \partial_{zz}) \phi = f + O(\Delta t^2)
$$

The splitting error is $O(\alpha^2) = O(\Delta t^2)$, consistent with second-order Crank-Nicolson accuracy.

### 5.2 Three-Sweep Algorithm

**Step 1: X-Sweep**

Solve for intermediate $\phi^*$ with explicit Y and Z diffusion:
$$
(I - \alpha \partial_{xx}) \phi^* = f + \alpha (\partial_{yy} + \partial_{zz}) \phi^n
$$

For each $(j, k)$ pair, this gives a tridiagonal system in $i$:
$$
-r_x \phi^*_{i-1,j,k} + (1 + 2r_x) \phi^*_{i,j,k} - r_x \phi^*_{i+1,j,k} = \text{RHS}_{x}
$$

where $r_x = \alpha / \Delta x^2$.

**Step 2: Y-Sweep**

Remove explicit Y diffusion and apply implicit Y operator:
$$
(I - \alpha \partial_{yy}) \phi^{**} = \phi^* - \alpha \partial_{yy} \phi^n
$$

For each $(i, k)$ pair, tridiagonal in $j$:
$$
-r_y \phi^{**}_{i,j-1,k} + (1 + 2r_y) \phi^{**}_{i,j,k} - r_y \phi^{**}_{i,j+1,k} = \text{RHS}_{y}
$$

**Step 3: Z-Sweep**

Remove explicit Z diffusion and apply implicit Z operator:
$$
(I - \alpha \partial_{zz}) \phi^{n+1} = \phi^{**} - \alpha \partial_{zz} \phi^n
$$

For each $(i, j)$ pair, tridiagonal in $k$:
$$
-r_z \phi^{n+1}_{i,j,k-1} + (1 + 2r_z) \phi^{n+1}_{i,j,k} - r_z \phi^{n+1}_{i,j,k+1} = \text{RHS}_{z}
$$

### 5.3 Masked Tridiagonal Systems

For solid points ($M(i,j,k) = 0$), the tridiagonal system is modified:
$$
0 \cdot \phi_{-} + 1 \cdot \phi_{i,j,k} + 0 \cdot \phi_{+} = 0
$$

This enforces $\phi = 0$ inside the step (no-slip).

---

## 6. 3D Pressure Poisson Solver

### 6.1 Discrete Equation

The 3D Poisson equation uses a 7-point stencil:
$$
\frac{p_{i+1,j,k} - 2p_{i,j,k} + p_{i-1,j,k}}{\Delta x^2} + 
\frac{p_{i,j+1,k} - 2p_{i,j,k} + p_{i,j-1,k}}{\Delta y^2} +
\frac{p_{i,j,k+1} - 2p_{i,j,k} + p_{i,j,k-1}}{\Delta z^2} = b_{i,j,k}
$$

where $b = \frac{\rho}{\Delta t} \nabla \cdot \mathbf{u}^*$.

### 6.2 Jacobi Iteration

Rearranging for iterative solution:
$$
p_{i,j,k} = \frac{1}{2(\gamma_x + \gamma_y + \gamma_z)} \left[ 
    \gamma_x (p_{i+1} + p_{i-1}) + 
    \gamma_y (p_{j+1} + p_{j-1}) + 
    \gamma_z (p_{k+1} + p_{k-1}) - 
    \Delta x^2 \Delta y^2 \Delta z^2 \cdot b
\right]
$$

where $\gamma_x = \Delta y^2 \Delta z^2$, $\gamma_y = \Delta x^2 \Delta z^2$, $\gamma_z = \Delta x^2 \Delta y^2$.

---

## 7. Boundary Conditions

### 7.1 Velocity Boundaries

| Boundary | Location | Condition |
|----------|----------|-----------|
| **Top Wall** | $y = H$ | $u = v = w = 0$ |
| **Bottom Wall** | $y = 0$ | $u = v = w = 0$ |
| **Sidewall (left)** | $z = 0$ | $u = v = w = 0$ |
| **Sidewall (right)** | $z = W$ | $u = v = w = 0$ |
| **Step** | $\Omega_{solid}$ | $u = v = w = 0$ (via mask) |
| **Inlet** | $x = -L_{in}$ | $u = u_{profile}(y,z)$, $v = w = 0$ |
| **Outlet** | $x = L_{out}$ | $\partial u/\partial x = \partial v/\partial x = \partial w/\partial x = 0$ |

### 7.2 Pressure Boundaries

| Boundary | Condition |
|----------|-----------|
| **Inlet** | $\partial p / \partial x = 0$ |
| **Outlet** | $p = 0$ (reference) |
| **Top/Bottom** | $\partial p / \partial y = 0$ |
| **Sidewalls** | $\partial p / \partial z = 0$ |
| **Step Top** | $p_{j_{step}-1} = p_{j_{step}}$ |
| **Step Face** | $p_{i_{step}-1} = p_{i_{step}}$ |

### 7.3 3D Inlet Profile

For a rectangular duct, the inlet profile is separable:
$$
u(y, z) = u_y(y) \cdot u_z(z)
$$

where each component is parabolic:
$$
u_y(y) = 4 \frac{y}{h_{inlet}} \left(1 - \frac{y}{h_{inlet}}\right), \quad
u_z(z) = 4 \frac{z}{W} \left(1 - \frac{z}{W}\right)
$$

The combined profile peaks at $(y = h_{inlet}/2, z = W/2)$ with value $U_{max}$.

---

## 8. Validation & Expected Results

### 8.1 Reattachment Length

The primary validation metric is $X_r/h$, measured at the mid-plane ($z = W/2$).

**Expected Behavior:**

| Reynolds | 2D Xr/h | 3D Mid-plane Xr/h | Armaly Exp. |
|----------|---------|-------------------|-------------|
| 100 | ~3.0 | ~3.0 | ~3.0 |
| 200 | ~5.0 | ~5.0 | ~5.0 |
| 400 | ~8.0 | ~7.0-8.0 | ~8.2 |
| 800 | ~10+ (2D) | ~7-8 | ~7.5 |

### 8.2 3D Effects

Key observations:
1. At low Re (<300), 2D and 3D results agree well
2. At moderate Re (400-800), 3D sidewall effects reduce Xr/h
3. Spanwise variation: Xr is largest at mid-plane, shortest near sidewalls

### 8.3 Computational Notes

- Grid: $(N_x, N_y, N_z)$ - typical $(101, 26, 26)$ for testing
- The 3D simulation is $\sim 26\times$ more expensive than 2D (one factor of $N_z$)
- JAX `vmap` enables efficient parallelization of tridiagonal solves
 
### 8.4 High-Reynolds Number Physics (Re=2000)
 
 At $Re_h = 2000$, the flow transitions from laminar steady state to **unsteady turbulent** behavior.
 
 - **Numerical Approach**: The simulation acts as an **Implicit Large Eddy Simulation (ILES)** or under-resolved DNS. The numerical dissipation of the discrete operators stabilizes the solution without an explicit subgrid-scale turbulence model.
 - **Unsteady Dynamics**: Unlike the steady low-Re cases, the flow at Re=2000 does not converge to a static solution ($d/dt = 0$). Instead, it exhibits:
     - Kelvin-Helmholtz instabilities in the shear layer.
     - Vortex shedding and breakup downstream of the step.
     - Fluctuating reattachment points.
 - **Validation Criterion**: For unsteady flow, statistical averages (mean velocity, Reynolds stresses) should be compared with experiments, rather than instantaneous snapshots.
 
---

## 9. References

1. Armaly, B.F., Durst, F., Pereira, J.C.F., & Schönung, B. (1983). "Experimental and theoretical investigation of backward-facing step flow." *Journal of Fluid Mechanics*, 127, 473-496.

2. Ghia, K.N., Osswald, G.A., & Ghia, U. (1989). "Analysis of incompressible massively separated viscous flows using unsteady Navier-Stokes equations." *International Journal for Numerical Methods in Fluids*, 9(8), 1025-1050.

3. Williams, P.T., & Baker, A.J. (1997). "Numerical simulations of laminar flow over a 3D backward-facing step." *International Journal for Numerical Methods in Fluids*, 24(11), 1159-1183.

4. Douglas, J., & Gunn, J.E. (1964). "A general formulation of alternating direction methods." *Numerische Mathematik*, 6(1), 428-453.
