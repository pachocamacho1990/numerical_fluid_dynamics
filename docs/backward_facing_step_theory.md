# Backward-Facing Step Flow - Numerical Scheme

Based on the **Armaly et al. (1983)** benchmark geometry and using standard finite difference methods.

## 1. Physical Problem & Geometry

The problem consists of flow in a channel with a sudden expansion (step). This geometry induces flow separation and a recirculation zone (separation bubble) downstream of the step. The length of this reattachment region ($X_r$) is a primary validation metric.

### Armaly Benchmark Dimensions
*   **Step Height ($h$)**: $0.51$
*   **Channel Height after step ($H$)**: $1.0$
*   **Inlet Height ($h_{inlet}$)**: $H - h = 0.49$
*   **Expansion Ratio (ER)**: $H / h_{inlet} = 1.94$
*   **Inlet Length ($L_{in}$)**: $5h \approx 2.55$
*   **Outlet Length ($L_{out}$)**: $30h \approx 15.3$

## 2. Governing Equations

We solve the **2D Incompressible Navier-Stokes Equations**:

**Continuity:**
$$
\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0
$$

**Momentum:**
$$
\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} = -\frac{1}{\rho} \frac{\partial p}{\partial x} + \nu \nabla^2 u
$$

$$
\frac{\partial v}{\partial t} + u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} = -\frac{1}{\rho} \frac{\partial p}{\partial y} + \nu \nabla^2 v
$$

## 3. Numerical Method

We use a **Finite Difference Method** on a staggered or collocated grid (collocated in this implementation with careful boundary handling).

### Discretization Schemes
1.  **Time**: Explicit Euler (Forward Difference).
2.  **Diffusion**: Central Difference (2nd order).
3.  **Advection**: **Conditional Upwinding**.
    *   Unlike cavity flow, the backward-facing step has a large recirculation zone where $u < 0$.
    *   Standard backward differencing is unstable here.
    *   We use a conditional scheme:
        *   If $u > 0$: Backward Difference ($\frac{u_{i} - u_{i-1}}{\Delta x}$)
        *   If $u < 0$: Forward Difference ($\frac{u_{i+1} - u_{i}}{\Delta x}$)

### Pressure Correction
We use the **Pressure Poisson Equation (PPE)** to enforce continuity:
$$
\nabla^2 p = \frac{\rho}{\Delta t} \nabla \cdot \mathbf{u}^*
$$
Solved iteratively using Jacobi or SOR.

## 4. Boundary Conditions

| Boundary | Location | Type | Condition |
|----------|----------|------|-----------|
| **Inlet** | $x = -L_{in}$ | Dirichlet | Parabolic Profile: $u(y) = U_{max} \frac{4y}{h_{inlet}}(1 - \frac{y}{h_{inlet}})$ |
| **Outlet** | $x = L_{out}$ | Neumann | $\frac{\partial u}{\partial x} = 0, \frac{\partial v}{\partial x} = 0, p = 0$ |
| **Walls** | Top/Bottom | Dirichlet | No-slip: $u=0, v=0$ |
| **Step** | Vertical Face | Dirichlet | No-slip: $u=0, v=0$ |

## 5. Stability & Validation

### Reynolds Number Definition
$$
Re_h = \frac{U_{avg} \cdot h}{\nu} = \frac{\frac{2}{3} U_{max} \cdot h}{\nu}
$$
*Note: Some definitions use hydraulic diameter or channel height. We strictly follow the Armaly definition based on step height.*

### Reattachment Length
The primary validation metric is the reattachment length $X_r/h$, defined as the distance from the step to the point where wall shear stress ($\tau_w \propto \frac{\partial u}{\partial y}|_{wall}$) changes sign from negative to positive.

**Expected Results (Armaly et al.):**
*   $Re = 100 \implies X_r/h \approx 3.0 - 3.5$
*   $Re = 200 \implies X_r/h \approx 5.0$
*   $Re = 400 \implies X_r/h \approx 8.0$
