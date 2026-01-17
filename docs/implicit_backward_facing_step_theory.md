# Implicit Backward-Facing Step Flow - Masked ADI Method

This document details the **Implicit Masked ADI** numerical scheme used for the backward-facing step simulation. This method enables stable simulations at higher Reynolds numbers on a simple Cartesian grid by handling the step geometry through mathematical masking.

## 1. Overview

The solver uses a **Fractional Step (Projection) Method** combined with an **Alternating Direction Implicit (ADI)** scheme for the diffusion terms. The key innovation is the use of a **Masked ADI** solver to handle the non-rectangular domain (step geometry) without requiring unstructured meshes or coordinate transformations.

### Key Features
*   **Time Integration**: Crank-Nicolson for diffusion (2nd order, Unconditionally Stable), Explicit for advection (1st order upwind).
*   **Spatial Discretization**: Finite Difference on a Cartesian grid.
*   **Geometry Handling**: "Masked" approach where solid regions are mathematically solved as identity equations ($u=0$).

---

## 2. Masked ADI Solver

Standard ADI solvers typically require rectangular domains $(I - \alpha \nabla^2)u = f$. For the backward-facing step, the domain is irregular. We modify the ADI tridiagonal systems to enforce no-slip conditions in the solid regions.

### The Concept
We introduce a binary mask $M(x,y)$:
*   $M=1$: Fluid Region
*   $M=0$: Solid Region (Step)

### Modified Tridiagonal Systems
In the ADI sweeps (both X and Y directions), we build tridiagonal systems $A \phi = b$. The coefficients are modified based on the mask:

#### Fluid Points ($M=1$)
Standard diffusion discretization:
$$
-r \phi_{i-1} + (1 + 2r) \phi_{i} - r \phi_{i+1} = \text{RHS}_{fluid}
$$

#### Solid Points ($M=0$)
We enforce the identity equation $\phi = 0$:
$$
\phi_i = 0 \implies 0 \cdot \phi_{i-1} + 1 \cdot \phi_{i} + 0 \cdot \phi_{i+1} = 0
$$

By adjusting the matrix coefficients ($a=0, b=1, c=0$) and RHS ($d=0$) for masked rows/columns, the ADI solver naturally computes zero velocity in the step region while solving the diffusion equation in the fluid region. This avoids the complexity of block-structured grids.

---

## 3. Pressure Poisson with Step BCs

The most critical part of the fractional step method is the Pressure Poisson Equation (PPE):
$$
\nabla^2 p = \frac{\rho}{\Delta t} \nabla \cdot \mathbf{u}^*
$$

### Boundary Conditions
Proper boundary conditions are essential for mass conservation, especially at the re-entrant corner of the step.

#### 1. Global Boundaries
*   **Inlet ($x=-L_{in}$)**: $\partial p / \partial x = 0$
*   **Outlet ($x=L_{out}$)**: $p = 0$ (Dirichlet)
*   **Top/Bottom Walls**: $\partial p / \partial y = 0$

#### 2. Step Boundaries (Re-entrant Corner)
The step introduces internal boundaries where we must apply Neumann conditions ($\partial p / \partial n = 0$).

*   **Step Top Surface** ($y=h, x<0$):
    *   Condition: $\partial p / \partial y = 0$
    *   Implementation: $p_{solid}(x, h-\Delta y) = p_{fluid}(x, h)$ (Ghost point copy)
    
*   **Step Vertical Face** ($x=0, y<h$):
    *   Condition: $\partial p / \partial x = 0$
    *   Implementation: $p_{solid}(-\Delta x, y) = p_{fluid}(0, y)$ (Ghost point copy)

These conditions ensure that the pressure gradient does not drive flux across the solid walls.

---

## 4. Algorithm Steps

1.  **Advection (Explicit)**:
    Compute advective terms using explicit upwinding on the current velocity field $\mathbf{u}^n$.
    $$
    \mathbf{A}^n = -(\mathbf{u}^n \cdot \nabla) \mathbf{u}^n
    $$

2.  **Predictor (Masked ADI)**:
    Solve the Helmholtz equation for intermediate velocity $\mathbf{u}^*$:
    $$
    \left(I - \frac{\nu \Delta t}{2} \nabla^2\right) \mathbf{u}^* = \mathbf{u}^n + \Delta t \mathbf{A}^n + \frac{\nu \Delta t}{2} \nabla^2 \mathbf{u}^n
    $$
    *Solved using Masked ADI to enforce $\mathbf{u}^*=0$ in the step.*

3.  **Pressure Solve**:
    Solve $\nabla^2 p = \frac{\rho}{\Delta t} \nabla \cdot \mathbf{u}^*$ with the boundary conditions described above.
    *Implemented using an iterative solver (Jacobi/SOR) that respects the step geometry.*

4.  **Corrector**:
    Update velocity to be divergence-free:
    $$
    \mathbf{u}^{n+1} = \mathbf{u}^* - \frac{\Delta t}{\rho} \nabla p
    $$
    *Re-apply mask to ensure $\mathbf{u}^{n+1}=0$ in solid regions.*

---

## 5. References
*   **Armaly, B. F., et al. (1983).** "Experimental and theoretical investigation of backward-facing step flow." *Journal of Fluid Mechanics*, 127, 473-496.
*   **Patankar, S. V. (1980).** *Numerical Heat Transfer and Fluid Flow*. (Concept of "Blocked-off regions").
