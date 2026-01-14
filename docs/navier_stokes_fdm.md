# The Mathematics of Incompressible Fluid Dynamics (FDM)

This document provides a rigorous derivation of the numerical methods used in the **Lid-Driven Cavity Flow** simulation. It moves beyond simply listing equations to explaining *why* we solve them this way.

## 1. The Governing Physics

### 1.1 The Navier-Stokes Equations
The motion of an incompressible, viscous Newtonian fluid is governed by two fundamental conservation laws:

1.  **Conservation of Mass (Continuity Equation)**:
    $$
    \nabla \cdot \mathbf{u} = 0
    $$
    *Physical Interpretation*: What goes in must come out. Since density is constant ($\rho = \text{const}$), the velocity field $\mathbf{u}$ must be divergence-free (solenoidal).

2.  **Conservation of Momentum**:
    $$
    \frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla) \mathbf{u} = -\frac{1}{\rho} \nabla p + \nu \nabla^2 \mathbf{u}
    $$
    *   **$\frac{\partial \mathbf{u}}{\partial t}$**: Unsteady acceleration.
    *   **$(\mathbf{u} \cdot \nabla) \mathbf{u}$**: Convection (Advection). Nonlinear transport of momentum.
    *   **$-\frac{1}{\rho} \nabla p$**: Pressure gradient force. Drives flow from high to low pressure.
    *   **$\nu \nabla^2 \mathbf{u}$**: Viscous diffusion. Effect of fluid friction (Kinematic Viscosity $\nu$).

---

## 2. Chorin's Projection Method (1968)

Directly solving these coupled equations is difficult because there is no explicit equation for pressure $p$. Pressure acts as a **Lagrange Multiplier** to enforce the incompressibility constraint ($\nabla \cdot \mathbf{u} = 0$).

To solve this numerically, we use the **Fractional Step Method** (or Projection Method), which splits the problem into two parts:

### Step 2.1: The Tentative Velocity Step ($u^*$)
First, we solve the momentum equation *ignoring the pressure gradient*. We calculate a temporary, intermediate velocity field $\mathbf{u}^*$ that is **not divergence-free**.

$$
\frac{\mathbf{u}^* - \mathbf{u}^n}{\Delta t} = -(\mathbf{u}^n \cdot \nabla) \mathbf{u}^n + \nu \nabla^2 \mathbf{u}^n
$$

Solving for $\mathbf{u}^*$:
$$
\mathbf{u}^* = \mathbf{u}^n + \Delta t \left[ -(\mathbf{u}^n \cdot \nabla) \mathbf{u}^n + \nu \nabla^2 \mathbf{u}^n \right]
$$

### Step 2.2: The Pressure Correction
Now we need to correct $\mathbf{u}^*$ by adding the pressure term we skipped, such that the final velocity $\mathbf{u}^{n+1}$ satisfies $\nabla \cdot \mathbf{u}^{n+1} = 0$.

The correction equation is:
$$
\mathbf{u}^{n+1} = \mathbf{u}^* - \frac{\Delta t}{\rho} \nabla p^{n+1}
$$

Taking the divergence of both sides:
$$
\nabla \cdot \mathbf{u}^{n+1} = \nabla \cdot \mathbf{u}^* - \frac{\Delta t}{\rho} \nabla^2 p^{n+1}
$$

We demand that $\nabla \cdot \mathbf{u}^{n+1} = 0$ (Continuity). This simplifies the equation to the **Pressure Poisson Equation (PPE)**:

$$
\nabla^2 p^{n+1} = \frac{\rho}{\Delta t} \nabla \cdot \mathbf{u}^*
$$

This is purely elliptic: "What pressure distribution $p$ is required to counteract the divergence present in $\mathbf{u}^*$?"

### Step 2.3: Velocity Update
Once $p^{n+1}$ is computed (by relaxing the Poisson equation), we perform the final update:

$$
\mathbf{u}^{n+1} = \mathbf{u}^* - \frac{\Delta t}{\rho} \nabla p^{n+1}
$$

---

## 3. Finite Difference Discretization
We use a standard MAC (Marker-and-Cell) influenced approach on a collocated grid.

*   $i$ index for $x$ direction.
*   $j$ index for $y$ direction.

### 3.1 The "b" Term (Source of PPE)
In the code, you will see a function `build_up_b`. This calculates the Right Hand Side (RHS) of the Poisson equation.

$$
b_{i,j} = \frac{\rho}{\Delta t} \left( \frac{\partial u^*}{\partial x} + \frac{\partial v^*}{\partial y} \right)
$$
*(Note: In the actual implementation, we approximate $\nabla \cdot \mathbf{u}^*$ using terms derived from the explicit discretization of the momentum terms. See code comments for local expansion.)*

### 3.2 Solving the PPE
The Poisson equation $\nabla^2 p = b$ is discretized using central differences:

$$
\frac{p_{i+1,j} - 2p_{i,j} + p_{i-1,j}}{\Delta x^2} + \frac{p_{i,j+1} - 2p_{i,j} + p_{i,j-1}}{\Delta y^2} = b_{i,j}
$$

Rearranging for $p_{i,j}$ gives the iterative update rule used in the **Jacobi / Gauss-Seidel solver**:

$$
p_{i,j}^{new} = \frac{(p_{i+1,j} + p_{i-1,j})\Delta y^2 + (p_{i,j+1} + p_{i,j-1})\Delta x^2 - b_{i,j}\Delta x^2 \Delta y^2}{2(\Delta x^2 + \Delta y^2)}
$$

---

## 4. Boundary Conditions

### 4.1 Velocity BCs (No-Slip)
At stationary walls, fluid sticks to the surface:
*   $u = 0, v = 0$ at $x=0, x=2, y=0$.
At the moving lid ($y=2$):
*   $u = 1, v = 0$.

### 4.2 Pressure BCs (Neumann)
This is subtle. Why do we set $\frac{\partial p}{\partial n} = 0$?
If we look at the momentum equation at a stationary wall (where $\mathbf{u}=0$):

$$
0 = -\frac{1}{\rho}\nabla p + \nu \nabla^2 \mathbf{u}
$$

For high Reynolds number flows, the boundary layer approximation effectively implies that the pressure gradient normal to the wall is zero:
$$
\frac{\partial p}{\partial n} \approx 0
$$

**In Code:**
*   Right Wall ($x=L$): `p[:,-1] = p[:,-2]` implies $\frac{\partial p}{\partial x} = 0$.
*   Bottom Wall ($y=0$): `p[0,:] = p[1,:]` implies $\frac{\partial p}{\partial y} = 0$.
