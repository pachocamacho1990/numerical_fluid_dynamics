# Navier-Stokes Equations: Finite Difference Method Derivation

This document details the step-by-step derivation of the discrete Navier-Stokes equations used in the Lid-Driven Cavity flow simulation.

## 1. The Governing Equations

For an incompressible, viscous fluid, the equations are:

**Continuity Equation (Conservation of Mass):**

$$
\nabla \cdot \mathbf{u} = 0
$$

**Momentum Equation (Conservation of Momentum):**

$$
\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla) \mathbf{u} = -\frac{1}{\rho} \nabla p + \nu \nabla^2 \mathbf{u}
$$

In 2D Cartesian coordinates $(x, y)$ with velocity components $(u, v)$, these expand to:

**2D Continuity:**

$$
\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0
$$

**x-Momentum:**

$$
\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} = -\frac{1}{\rho} \frac{\partial p}{\partial x} + \nu \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right)
$$

**y-Momentum:**

$$
\frac{\partial v}{\partial t} + u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} = -\frac{1}{\rho} \frac{\partial p}{\partial y} + \nu \left( \frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2} \right)
$$

---

## 2. Finite Difference Discretization

We discretize the domain into a grid with spacing $\Delta x$ and $\Delta y$, and time with step $\Delta t$.
$u_{i,j}^n$ denotes velocity at $x = i\Delta x$, $y = j\Delta y$ and time $t = n\Delta t$.

### Derivatives Approximation

*   **Time Derivative (Forward Difference):**

$$ \frac{\partial u}{\partial t} \approx \frac{u_{i,j}^{n+1} - u_{i,j}^n}{\Delta t} $$

*   **First Spatial Derivative (Backward Difference for Convection - Upwind Scheme generally better, but here we use Central/Backward mixes for simplicity in this basic solver):**
    *In this specific implementation, we used backward difference for convection:*
    
$$ \frac{\partial u}{\partial x} \approx \frac{u_{i,j}^n - u_{i-1,j}^n}{\Delta x} $$

*   **First Spatial Derivative (Central Difference for Pressure):**

$$ \frac{\partial p}{\partial x} \approx \frac{p_{i+1,j}^n - p_{i-1,j}^n}{2\Delta x} $$

*   **Second Spatial Derivative (Central Difference for Diffusion):**
  
$$ \frac{\partial^2 u}{\partial x^2} \approx \frac{u_{i+1,j}^n - 2u_{i,j}^n + u_{i-1,j}^n}{\Delta x^2} $$

---

## 3. Discretized Momentum Equations

Substituting the approximations into the momentum equations and solving for $u_{i,j}^{n+1}$:

### x-Momentum Discrete Form:

$$
\begin{aligned}
u_{i,j}^{n+1} = u_{i,j}^n & - u_{i,j}^n \frac{\Delta t}{\Delta x} (u_{i,j}^n - u_{i-1,j}^n) \\
& - v_{i,j}^n \frac{\Delta t}{\Delta y} (u_{i,j}^n - u_{i,j-1}^n) \\
& - \frac{\Delta t}{2\rho\Delta x} (p_{i+1,j}^n - p_{i-1,j}^n) \\
& + \nu \Delta t \left( \frac{u_{i+1,j}^n - 2u_{i,j}^n + u_{i-1,j}^n}{\Delta x^2} + \frac{u_{i,j+1}^n - 2u_{i,j}^n + u_{i,j-1}^n}{\Delta y^2} \right)
\end{aligned}
$$

### y-Momentum Discrete Form:

Similarly for $v$:

$$
\begin{aligned}
v_{i,j}^{n+1} = v_{i,j}^n & - u_{i,j}^n \frac{\Delta t}{\Delta x} (v_{i,j}^n - v_{i-1,j}^n) \\
& - v_{i,j}^n \frac{\Delta t}{\Delta y} (v_{i,j}^n - v_{i,j-1}^n) \\
& - \frac{\Delta t}{2\rho\Delta y} (p_{i,j+1}^n - p_{i,j-1}^n) \\
& + \nu \Delta t \left( \frac{v_{i+1,j}^n - 2v_{i,j}^n + v_{i-1,j}^n}{\Delta x^2} + \frac{v_{i,j+1}^n - 2v_{i,j}^n + v_{i,j-1}^n}{\Delta y^2} \right)
\end{aligned}
$$

---

## 4. Pressure Poisson Equation (PPE)

Pressure does not have an explicit evolution equation. We derive one by taking the divergence of the momentum equation and enforcing the continuity constraint $\nabla \cdot \mathbf{u}^{n+1} = 0$.

The resulting Poisson Equation is:

$$
\frac{\partial^2 p}{\partial x^2} + \frac{\partial^2 p}{\partial y^2} = -\rho \left[ \left(\frac{\partial u}{\partial x}\right)^2 + 2 \frac{\partial u}{\partial y} \frac{\partial v}{\partial x} + \left(\frac{\partial v}{\partial y}\right)^2 \right] + \dots
$$

*(Note: Continuity terms simplify substantially)*

### Discrete PPE
Solving for $p_{i,j}$:

$$
p_{i,j}^n = \frac{(p_{i+1,j}^n + p_{i-1,j}^n)\Delta y^2 + (p_{i,j+1}^n + p_{i,j-1}^n)\Delta x^2 - b_{source}\Delta x^2 \Delta y^2}{2(\Delta x^2 + \Delta y^2)}
$$

Where $b_{source}$ is the discretized source term calculated from the velocity field at the current step. This equation is solved iteratively (relaxation) at every time step.
