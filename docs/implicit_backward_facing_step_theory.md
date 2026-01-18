# Theory: Implicit Backward-Facing Step Flow

This document provides a comprehensive theoretical background for the implicit (Masked ADI) solution of the backward-facing step flow. It covers the flow physics at various Reynolds numbers, the mathematical formulation of the Masked ADI method, and the rigorous implementation of boundary conditions.

## 1. Flow Physics & Regimes

The backward-facing step (BFS) is a canonical problem for studying flow separation and reattachment. The geometry induces a shear layer separation at the step edge ($x=0, y=h$), which forms a primary recirculation zone (vortex) downstream.

### 1.1 Reynolds Number Definition
We use the standard Armaly et al. definition based on the step height $h$ and the average inlet velocity $\bar{U}_{inlet}$:
$$ Re_h = \frac{\bar{U}_{inlet} h}{\nu} $$
*Note: Our code input uses $U_{max}$. For a parabolic profile, $\bar{U} = \frac{2}{3} U_{max}$.*

### 1.2 Flow Regimes
As $Re$ increases, the flow undergoes several distinct transitions:

1.  **Viscous Laminar Flow ($Re < 100$)**:
    *   Single primary recirculation zone attached to the step.
    *   Flow is steady and 2D.
    *   Reattachment length $X_1$ grows linearly with $Re$.

2.  **Transitional Laminar Flow ($100 < Re < 400$)**:
    *   Separation on the upper wall (roof eddy) may appear around $Re \approx 400$.
    *   Flow remains steady and 2D.

3.  **Transitional Multi-Vortex Flow ($400 < Re < 1200$)**:
    *   **Primary Zone ($X_1$)**: Attached to the step bottom wall.
    *   **Secondary Zone ($X_2$)**: Forms on the top wall (roof eddy).
    *   **Tertiary Zone ($X_3$)**: May form on the bottom wall further downstream.
    *   *Note: In 2D simulations, these eddies are stable. In 3D experiments (Armaly), 3D effects reduce the size of $X_1$ significantly for $Re > 400$. Our 2D simulation follows the 2D solution (Ghia et al.), which actually predicts longer reattachment lengths than experiments in this regime.*

4.  **Unsteady/Turbulent Flow ($Re > 1200$)**:
    *   Kelvin-Helmholtz instabilities in the shear layer lead to vortex shedding.
    *   The flow becomes unsteady and eventually fully turbulent.
    *   At **$Re=5000$**, the flow is dominated by strong vortex shedding and complex interactions between the shear layer and the walls. The "steady" solution is no longer physically realizable; we observe a time-averaged mean flow with fluctuating components.

---

## 2. Mathematical Formulation: Masked ADI

To solve the Navier-Stokes equations on the irregular step domain $\Omega$ using a simple Cartesian grid, we employ a **Masked Alternating Direction Implicit (ADI)** method.

### 2.1 The Domain & Mask
We define a rectangular computational domain $\Omega_{comp} = [0, L] \times [0, H]$. The physical domain $\Omega_{fluid}$ is a subset. We define a mask function $M(i,j)$:

$$
M(i,j) = \begin{cases} 
1 & \text{if } (x_i, y_j) \in \Omega_{fluid} \\
0 & \text{if } (x_i, y_j) \in \Omega_{solid} \text{ (The Step)}
\end{cases}
$$

### 2.2 The Helmholtz Problem
The implicit diffusion step involves solving a Helmholtz equation for velocity component $\phi$ (where $\phi$ is $u$ or $v$):
$$
(I - \alpha \nabla^2) \phi = f
$$
where $\alpha = \frac{\nu \Delta t}{2}$.

### 2.3 Factorization and Masking
The ADI method factors this into X and Y sweeps.

#### X-Sweep (Row $j$)
We solve for intermediate $\phi^*$:
$$ (I - \alpha \partial_{xx}) \phi^* = \text{RHS} $$

For a grid row $j$, this leads to a tridiagonal system $A_x \mathbf{\phi}^*_j = \mathbf{b}$. We modify the matrix rows based on the mask $M(i,j)$:

*   **If $M(i,j) = 1$ (Fluid)**:
    Standard central difference stencil:
    $$ -\frac{\alpha}{\Delta x^2} \phi^*_{i-1} + \left(1 + \frac{2\alpha}{\Delta x^2}\right) \phi^*_i - \frac{\alpha}{\Delta x^2} \phi^*_{i+1} = b_i $$

*   **If $M(i,j) = 0$ (Solid)**:
    Identity equation (Dirichlet $u=0$):
    $$ 0 \cdot \phi^*_{i-1} + 1 \cdot \phi^*_i + 0 \cdot \phi^*_{i+1} = 0 $$
    *(We set $b_i = 0$)*.

This ensures that in the solid region, the velocity is forced to zero, effectively applying the no-slip condition at the staircase boundary of the step.

#### Y-Sweep (Column $i$)
Similarly, we solve $(I - \alpha \partial_{yy}) \phi^{n+1} = \phi^*$.
For solid points $M(i,j)=0$, we again enforce $1 \cdot \phi^{n+1}_j = 0$.

---

## 3. Pressure Boundary Conditions at the Corner

The most challenging aspect of the step geometry on a Cartesian grid is the "re-entract corner" ($x=0, y=h$).

### 3.1 The Physical Condition
For high Re incompressible flow, the pressure boundary condition at a solid wall comes from projecting the momentum equation onto the wall normal $\mathbf{n}$:
$$ \frac{\partial p}{\partial n} = \mathbf{n} \cdot (\nu \nabla^2 \mathbf{u} - \frac{\partial \mathbf{u}}{\partial t} - (\mathbf{u} \cdot \nabla)\mathbf{u}) $$
For a stationary no-slip wall ($Re \to \infty$), this simplifies to $\frac{\partial p}{\partial n} \approx 0$.

### 3.2 Discrete Implementation (Ghost Points)
Our Poisson solver uses a 5-point stencil. Near boundaries, this stencil reaches outside the fluid domain.

#### Step Top Surface ($y=h, x<0$)
The fluid nodes are at $j_{step}$. The solid nodes are at $j_{step}-1$.
To correct the stencil at $j_{step}$, we need a value for $p_{j_{step}-1}$.
Condition $\frac{\partial p}{\partial y} = 0 \implies p_{j_{step}-1} = p_{j_{step}}$.

In the code, we explicitly copy the pressure values from the first fluid layer into the last solid layer (the "ghost" layer) before the Poisson iteration (or during it):
```python
# Copy row j_step to j_step-1
p[j_step-1, :i_step] = p[j_step, :i_step] 
```

#### Step Vertical Face ($x=0, y<h$)
The fluid nodes are at $i_{step}$. The solid nodes are at $i_{step}-1$.
Condition $\frac{\partial p}{\partial x} = 0 \implies p_{i_{step}-1} = p_{i_{step}}$.

```python
# Copy col i_step to i_step-1
p[:j_step, i_step-1] = p[:j_step, i_step]
```

### 3.3 Why This Matters
Failure to implement these corner BCs correctly results in "pressure leakage" into the step, causing non-physical gradients that can destabilize the simulation at high Reynolds numbers (like $Re=5000$). Our rigorous implementation ensures mass is strictly conserved within the fluid domain.

---

## 4. Simulation Strategy for High Re

For $Re=5000$:
1.  **Resolution**: The boundary layers become very thin ($\delta \propto Re^{-1/2}$). A fine grid ($801 \times 301$) allows capturing these gradients.
2.  **Time Step**: Although the implicit diffusion allows large $\Delta t$, the explicit advection (and the physics of vortex shedding) limits $\Delta t$. We need $CFL < 1$ to accurately resolve the advection of vortices.
3.  **Long Integration**: At $Re=5000$, the flow never reaches a steady state. We must integrate for a long time ($T=400$ or more) to observe the quasi-periodic shedding cycle. 40,000 steps at $dt=0.005$ gives $T=200$, sufficient to see the dynamics.
