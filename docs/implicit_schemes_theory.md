# Implicit Numerical Schemes for Incompressible Navier-Stokes

This document covers **implicit time-stepping methods** for solving the incompressible Navier-Stokes equations. These methods remove the CFL stability constraint, enabling stable simulations at high Reynolds numbers.

## 1. Motivation: Why Implicit?

### Explicit Scheme Limitations

From our explicit cavity flow implementation, we learned that stability requires:

**CFL Condition (Advection):**

$$
\Delta t \cdot \left(\frac{|u_{\max}|}{\Delta x} + \frac{|v_{\max}|}{\Delta y}\right) < 1
$$

**Diffusion Stability:**

$$
\Delta t < \frac{\Delta x^2}{4\nu}
$$

For high Reynolds numbers ($Re = \frac{UL}{\nu}$), the viscosity $\nu$ becomes very small, and velocities can become large, forcing **extremely small time steps**.

### Implicit Advantage

Implicit schemes treat some terms at the **new time level** $n+1$, requiring the solution of linear systems but gaining **unconditional stability** for those terms.

---

## 2. The Fractional Step (Projection) Method

Developed independently by **Chorin (1967)** and **Temam (1968)**, this method decouples the velocity and pressure calculations.

### The Core Idea

The incompressibility constraint $\nabla \cdot \mathbf{u} = 0$ couples velocity and pressure. The projection method:
1. First solves for velocity **ignoring pressure**
2. Then **projects** the result onto a divergence-free space using pressure

### Algorithm

#### Step 1: Predictor (Intermediate Velocity)

Solve for $\mathbf{u}^*$ without pressure gradient:

$$
\frac{\mathbf{u}^* - \mathbf{u}^n}{\Delta t} + (\mathbf{u}^n \cdot \nabla)\mathbf{u}^n = \nu \nabla^2 \mathbf{u}^*
$$

Note: The diffusion term uses $\mathbf{u}^*$ (implicit), while advection uses $\mathbf{u}^n$ (explicit).

#### Step 2: Pressure Poisson Equation

Find $\phi$ such that $\mathbf{u}^{n+1}$ will be divergence-free:

$$
\nabla^2 \phi = \frac{\rho}{\Delta t} \nabla \cdot \mathbf{u}^*
$$

#### Step 3: Corrector (Velocity Update)

Project onto divergence-free space:

$$
\mathbf{u}^{n+1} = \mathbf{u}^* - \frac{\Delta t}{\rho} \nabla \phi
$$

$$
p^{n+1} = \phi \quad \text{(or accumulated pressure)}
$$

### Why It Works

From the Helmholtz-Hodge decomposition, any vector field can be written as:

$$
\mathbf{u}^* = \mathbf{u}^{n+1} + \nabla \phi
$$

where $\mathbf{u}^{n+1}$ is divergence-free and $\phi$ is a scalar potential.

---

## 3. Crank-Nicolson Time Discretization

For the diffusion term, we use **Crank-Nicolson** (average of old and new time levels):

$$
\frac{\partial u}{\partial t} = \nu \nabla^2 u \quad \Rightarrow \quad \frac{u^{n+1} - u^n}{\Delta t} = \frac{\nu}{2}\left(\nabla^2 u^{n+1} + \nabla^2 u^n\right)
$$

### Properties

| Property | Value |
|----------|-------|
| **Order** | 2nd order in time |
| **Stability** | Unconditionally stable (for diffusion) |
| **Damping** | Minimal numerical damping |

### Resulting Helmholtz Equation

Rearranging the predictor step with Crank-Nicolson diffusion:

$$
\left(I - \frac{\nu \Delta t}{2} \nabla^2\right) u^* = \left(I + \frac{\nu \Delta t}{2} \nabla^2\right) u^n - \Delta t \cdot \text{(advection terms)}
$$

This is a **Helmholtz equation** of the form:

$$
(I - \alpha \nabla^2) u = f
$$

where $\alpha = \frac{\nu \Delta t}{2}$.

---

## 4. Alternating Direction Implicit (ADI) Method

The Helmholtz equation in 2D becomes a large linear system. **ADI** makes this tractable by splitting into two 1D problems.

### The Douglas-Gunn Scheme

For the Helmholtz equation:

$$
(I - \alpha \nabla^2) u = f
$$

Split into two half-steps:

**Half-step 1 (x-direction implicit):**

$$
\left(I - \alpha \frac{\partial^2}{\partial x^2}\right) u^{*} = f + \alpha \frac{\partial^2 u^n}{\partial y^2}
$$

**Half-step 2 (y-direction implicit):**

$$
\left(I - \alpha \frac{\partial^2}{\partial y^2}\right) u^{n+1} = u^{*} - \alpha \frac{\partial^2 u^n}{\partial y^2}
$$

### Why ADI is Efficient

Each 1D problem gives a **tridiagonal system**, solvable in $O(N)$ time using the Thomas algorithm (as opposed to $O(N^2)$ or $O(N^3)$ for full 2D).

### Discretized Form

For the x-sweep at row $j$:

$$
-r_x u_{i-1,j}^* + (1 + 2r_x) u_{i,j}^* - r_x u_{i+1,j}^* = \text{RHS}_{i,j}
$$

where $r_x = \frac{\alpha}{\Delta x^2} = \frac{\nu \Delta t}{2 \Delta x^2}$

This is a tridiagonal system that can be solved efficiently.

---

## 5. Complete Algorithm Summary

```
For each time step n → n+1:

1. COMPUTE ADVECTION (explicit)
   adv_u = u^n * du/dx + v^n * du/dy  (backward differences)
   adv_v = u^n * dv/dx + v^n * dv/dy

2. PREDICTOR STEP (implicit diffusion via ADI)
   RHS_u = u^n + dt/2 * nu * Laplacian(u^n) - dt * adv_u
   RHS_v = v^n + dt/2 * nu * Laplacian(v^n) - dt * adv_v
   
   Solve: (I - dt/2 * nu * Laplacian) u* = RHS_u  [using ADI]
   Solve: (I - dt/2 * nu * Laplacian) v* = RHS_v  [using ADI]

3. PRESSURE POISSON
   RHS_p = rho/dt * divergence(u*, v*)
   Solve: Laplacian(phi) = RHS_p  [iterative solver]

4. CORRECTOR STEP
   u^{n+1} = u* - dt/rho * dp/dx
   v^{n+1} = v* - dt/rho * dp/dy

5. APPLY BOUNDARY CONDITIONS
```

---

## 6. Stability Analysis

### Explicit vs Implicit Comparison

| Term | Explicit | Implicit |
|------|----------|----------|
| **Advection** | CFL: $\Delta t < \frac{\Delta x}{|u|}$ | Same (often kept explicit) |
| **Diffusion** | $\Delta t < \frac{\Delta x^2}{4\nu}$ | **None** (unconditionally stable) |
| **Pressure** | — | Solved implicitly |

### What We Gain

For our cavity flow at $Re = 10000$:

**Explicit scheme:**
- $\nu = 0.0002$, $\Delta x = 0.01$ (100×100 grid)
- $\Delta t_{\max} = \frac{0.01^2}{4 \times 0.0002} = 0.125$
- But CFL with $u \approx 1.5$: $\Delta t < \frac{0.01}{1.5} = 0.0067$

**Implicit scheme:**
- Diffusion stability: **None**
- Only CFL for advection matters (or treat implicitly too)
- Can use $\Delta t = 0.1$ or larger!

---

## 7. Boundary Conditions

### Velocity Boundaries

Same as explicit scheme:
- **Walls:** $u = v = 0$ (no-slip)
- **Lid:** $u = 1$, $v = 0$ (moving lid)

### Pressure Boundaries

For the Poisson equation:
- **Walls:** $\frac{\partial p}{\partial n} = 0$ (Neumann)
- **Lid:** $p = 0$ (Dirichlet reference)

### ADI Boundary Treatment

At boundaries, the tridiagonal system is modified:
- Dirichlet: Set diagonal entry to 1, RHS to boundary value
- Neumann: Modify coefficients to enforce zero gradient

---

## 8. References

1. Chorin, A.J. (1967). "A numerical method for solving incompressible viscous flow problems." *Journal of Computational Physics*, 2(1), 12-26.

2. Temam, R. (1969). "Sur l'approximation de la solution des équations de Navier-Stokes par la méthode des pas fractionnaires." *Archive for Rational Mechanics and Analysis*, 33(5), 377-385.

3. Peyret, R., & Taylor, T.D. (1983). *Computational Methods for Fluid Flow*. Springer.

4. Douglas, J., & Gunn, J.E. (1964). "A general formulation of alternating direction methods." *Numerische Mathematik*, 6(1), 428-453.

5. Kim, J., & Moin, P. (1985). "Application of a fractional-step method to incompressible Navier-Stokes equations." *Journal of Computational Physics*, 59(2), 308-323.
