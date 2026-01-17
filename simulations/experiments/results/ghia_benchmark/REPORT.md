# Validation Report: Ghia et al. (1982) Benchmark

**Date:** 2026-01-16
**Solver:** Implicit JAX (Fractional Step)
**Grid:** 129x129
**Scheme:** First-Order Upwind Advection / Crank-Nicolson Diffusion

## 1. Introduction: The Lid-Driven Cavity

The **Lid-Driven Cavity** is the "Hello World" of Computational Fluid Dynamics (CFD). Ideally simple in geometry (a square box), it exhibits complex fluid physics governed by the Navier-Stokes equations:

$$
\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla) \mathbf{u} = -\frac{1}{\rho}\nabla p + \nu \nabla^2 \mathbf{u}
$$

$$
\nabla \cdot \mathbf{u} = 0
$$

The flow is driven by the top lid moving at constant velocity $U=1$. This creates a primary vortex in the center and secondary corner vortices. The balance between advection (inertia) and diffusion (viscosity) is determined by the Reynolds Number:

$$
Re = \frac{U L}{\nu}
$$

As $Re$ increases, the flow becomes dominated by inertia, the boundary layers become thinner, and the center vortex core moves towards the geometric center.

## 2. The Benchmark: Ghia, Ghia, and Shin (1982)

In 1982, U. Ghia, K.N. Ghia, and C.T. Shin published highly accurate numerical solutions for this problem using a coupled vorticity-streamfunction formulation on very fine grids (up to 257x257).

**Why is this the "Gold Standard"?**
- **Non-Linearity:** It stresses the non-linear advection term $(\mathbf{u} \cdot \nabla) \mathbf{u}$. If a solver's advection scheme is unstable (e.g., standard central differences at high Re), it will blow up.
- **Recirculation:** Unlike pipe flow, the velocity reverses direction ($u < 0$). This tests the solver's ability to handle "upwind" information flow correctly in all directions.
- **Conservation:** The closed box requires strict mass conservation. Any divergence error accumulates and destroys the solution.

Real-world experimental data (e.g., Koseff & Street) aligns closely with Ghia's 2D results for the centerline profiles, making this a reliable proxy for physical validity.

## 3. Experimental Setup

We validated our **Implicit JAX Solver** against the data for three flow regimes:

| Regime | Re | Characteristics | Challenge |
|:---|:---|:---|:---|
| **Laminar** | 100 | Viscosity dominated, thick boundary layers. | Easy. Tests basic diffusion logic. |
| **Transition** | 400 | Inertia growing, corner vortices appear. | Moderate. Standard explicit schemes often fail here. |
| **High Inertia** | 1000 | Thin boundary layers, nearly inviscid core. | Hard. Requires robust upwind advection and mass conservation. |

**Simulation Parameters:**
- **Grid:** $129 \times 129$ (Uniform)
- **Time Step:** $dt = 0.001$
- **Duration:** $t_{end} = 25.0$ (to ensure steady state)
- **Advection Scheme:** First-Order Upwind (Conditional)

## 4. Results and Validation

The plot below compares our simulation's **Vertical Centerline Velocity** $u(0.5, y)$ against the tabulated data points from Ghia et al. (1982).

![Ghia Validation Result](file:///Users/pacho-home-server/numerical_fluid_dynamics/simulations/experiments/results/ghia_benchmark/ghia_validation_full.png)

### Analysis
1.  **Re=100 (Blue):** Perfect agreement. This confirms the diffusion terms and pressure projection are correct.
2.  **Re=400 (Green):** Excellent agreement. The flow correctly captures the boundary layer steepness.
3.  **Re=1000 (Red):** The solution remains **stable** and tracks the benchmark closely.
    - Note the sharp gradients near the walls ($y=0$ and $y=1$).
    - The "S" shape indicates the strong inviscid core rotation.
    - Small deviations in the core are expected due to the artificial diffusion of the First-Order Upwind scheme, but the stability is absolute.

## 5. Conclusion

The standard validation confirms that the solver is **physically correct and numerically stable** up to $Re=1000$. The switch to an Upwind Advection scheme was the critical factor in enabling these high-Reynolds number result.
