# Cavity Flow Simulation - Numerical Scheme
Based on "Step 11" of Lorena Barba's *12 Steps to Navier-Stokes*.

## 1. Governing Equations

The flow is governed by the incompressible Navier-Stokes equations in 2D. 

**Conservation of Mass (Continuity Equation):**
$$
\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0
$$

**Conservation of Momentum:**
$$
\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} = -\frac{1}{\rho} \frac{\partial p}{\partial x} + \nu \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right)
$$

$$
\frac{\partial v}{\partial t} + u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} = -\frac{1}{\rho} \frac{\partial p}{\partial y} + \nu \left( \frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2} \right)
$$

Where:
*   $u, v$: Velocity components in $x$ and $y$ directions.
*   $p$: Pressure.
*   $\rho$: Density.
*   $\nu$: Kinematic viscosity.

## 2. Discretization

We use a **Finite Difference Method** with the following schemes:
*   **Time**: Forward Difference (Explicit).
*   **Space**: Central Difference (for both diffusion and pressure gradients).
*   **Non-linear Advection**: Backward Difference (Upwind) is often used for stability, but Step 11 in its simplest form often uses Central or Backward. *In our code, we used Backward Difference for the advection terms to ensure stability.*

### Discretized Momentum Equations

Solving for the velocity at the next time step ($u_{i,j}^{n+1}, v_{i,j}^{n+1}$):

$$
\begin{align}
u_{i,j}^{n+1} = u_{i,j}^n & - u_{i,j}^n \frac{\Delta t}{\Delta x} (u_{i,j}^n - u_{i-1,j}^n) - v_{i,j}^n \frac{\Delta t}{\Delta y} (u_{i,j}^n - u_{i,j-1}^n) \\
& - \frac{\Delta t}{2\rho \Delta x} (p_{i+1,j}^n - p_{i-1,j}^n) \\
& + \nu \left[ \frac{\Delta t}{\Delta x^2} (u_{i+1,j}^n - 2u_{i,j}^n + u_{i-1,j}^n) + \frac{\Delta t}{\Delta y^2} (u_{i,j+1}^n - 2u_{i,j}^n + u_{i,j-1}^n) \right]
\end{align}
$$

$$
\begin{align}
v_{i,j}^{n+1} = v_{i,j}^n & - u_{i,j}^n \frac{\Delta t}{\Delta x} (v_{i,j}^n - v_{i-1,j}^n) - v_{i,j}^n \frac{\Delta t}{\Delta y} (v_{i,j}^n - v_{i,j-1}^n) \\
& - \frac{\Delta t}{2\rho \Delta y} (p_{i,j+1}^n - p_{i,j-1}^n) \\
& + \nu \left[ \frac{\Delta t}{\Delta x^2} (v_{i+1,j}^n - 2v_{i,j}^n + v_{i-1,j}^n) + \frac{\Delta t}{\Delta y^2} (v_{i,j+1}^n - 2v_{i,j}^n + v_{i,j-1}^n) \right]
\end{align}
$$

*(Note: The advection terms above use backward differences spatial discretization)*

## 3. Pressure Poisson Equation

By taking the divergence of the discretized momentum equations and enforcing the continuity constraint ($\nabla \cdot \mathbf{u}^{n+1} = 0$), we derive a Poisson equation for pressure:

$$
\frac{\partial^2 p}{\partial x^2} + \frac{\partial^2 p}{\partial y^2} = -\rho \left[ \left( \frac{\partial u}{\partial x} \right)^2 + 2 \frac{\partial u}{\partial y} \frac{\partial v}{\partial x} + \left( \frac{\partial v}{\partial y} \right)^2 \right]
$$

### Discretized Pressure Equation

$$
\frac{p_{i+1,j}^n - 2p_{i,j}^n + p_{i-1,j}^n}{\Delta x^2} + \frac{p_{i,j+1}^n - 2p_{i,j}^n + p_{i,j-1}^n}{\Delta y^2} = b_{i,j}^n
$$

Where the source term $b_{i,j}^n$ represents the RHS components evaluated at time step $n$.

Rearranging to solve for $p_{i,j}^n$ iteratively (Jacobi method):

$$
p_{i,j}^n = \frac{ (p_{i+1,j}^n + p_{i-1,j}^n) \Delta y^2 + (p_{i,j+1}^n + p_{i,j-1}^n) \Delta x^2 - b_{i,j}^n \Delta x^2 \Delta y^2 }{ 2(\Delta x^2 + \Delta y^2) }
$$

## 4. Algorithm Steps

For each time step $n = 0 \dots N_t$:

1.  **Calculate Source Term ($b$)**: Compute the RHS of the pressure equation using current velocities $u^n, v^n$.
2.  **Solve Pressure Poisson**: Iterate (using Jacobi or SOR) to find the pressure field $p^n$ that satisfies continuity.
    *   Apply Pressure Boundary Conditions at every iteration.
3.  **Update Velocity**: Calculate $u^{n+1}, v^{n+1}$ using the momentum equations with $u^n, v^n, p^n$.
4.  **Apply Velocity Boundary Conditions**: Enforce no-slip and lid-driven conditions.
5.  **Advance**: $t \to t + \Delta t$.

## 5. Boundary Conditions

**Cavity Walls (fixed)** ($x=0, x=2, y=0$):
*   $u = 0, v = 0$ (No-slip)
*   $\frac{\partial p}{\partial n} = 0$ (Neumann for pressure)

**Lid (moving)** ($y=2$):
*   $u = 1, v = 0$ (Dirichlet for velocity)
*   $p = 0$ (Dirichlet for pressure reference)
