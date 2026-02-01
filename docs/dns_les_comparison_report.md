# DNS vs LES Comparison Study: Report

**Date**: February 1, 2026  
**Authors**: Numerical Fluid Dynamics Team  
**Status**: Completed

---

## 1. Executive Summary

This study verifies the accuracy and efficiency of a new Large Eddy Simulation (LES) implementation for the 3D backward-facing step flow. The performance of the LES solver (using a Smagorinsky SGS model) was compared against a high-fidelity Direct Numerical Simulation (DNS) baseline at a Reynolds number of **Re = 2000**.

**Key Results**:
- **Accuracy**: The LES model correctly captures the mean flow physics, including recirculation length and velocity recovery profiles. The maximum velocity matches within **0.06%** (1.4984 vs 1.4993).
- **Efficiency**: The LES simulation is approximately **68× faster** than the DNS reference (17.6 minutes vs 20.3 hours) while using a **13× coarser grid**.
- **Validation**: Explicit wall boundary conditions were critical for correct results; prior to this fix, the LES incorrectly predicted flow slip at the walls.

---

## 2. Methodology

### 2.1 Simulation Configuration

All simulations were performed using the JAX-based implicit solver on Apple Silicon (M1) hardware.

| Parameter | DNS (Reference) | LES (Test) |
|-----------|-----------------|------------|
| **Reynolds Number** | 2000 | 2000 |
| **Grid Resolution** | 301 × 76 × 76 (1.74M cells) | 128 × 32 × 32 (131k cells) |
| **Grid Spacing** | $\Delta x \approx 0.057$, $\Delta y_{min} \approx 0.013$ | $\Delta x \approx 0.134$, $\Delta y \approx 0.032$ |
| **Time Step ($\Delta t$)** | 0.005 | 0.005 |
| **Total Steps** | 120,000 | 120,000 |
| **Physical Time** | $T = 600$ | $T = 600$ |
| **Turbulence Model** | None (Direct Resolution) | Smagorinsky SGS ($C_s=0.12$) |
| **Wall Treatment** | Resolved (No-slip) | Explicit No-slip + Van Driest Damping |

### 2.2 Numerical Methods

- **Time Integration**: Implicit ADI (Alternating Direction Implicit) scheme.
- **Pressure Solver**: Iterative Pressure Poisson Equation solver.
- **LES Implementation**:
    - Strain rate tensor $S_{ij}$ computed at cell centers.
    - Eddy viscosity $\nu_t = (C_s \Delta)^2 |S|$ with wall damping.
    - Explicit variable-viscosity integration in the ADI sweeps.

---

## 3. Results Analysis

### 3.1 Velocity Field Structure

The following comparison shows the instantaneous velocity magnitude ($u$) in the z-midplane at $T=600$.

![Velocity Field Comparison](images/velocity_field_comparison.png)

**Observations**:
1.  **Macro Structure**: Both simulations clearly show the characteristic recirculation zone behind the step ($x < 6$) and flow reattachment.
2.  **Turbulence Scales**: The DNS (top) reveals finer turbulent structures and sharper gradients in the shear layer. The LES (bottom) shows a smoother field, consistent with the filtering of subgrid scales, but preserves the correct mean flow topology.
3.  **Boundary Layers**: Initial issues with wall slip in the LES were resolved. Both simulations now show zero velocity at the top and bottom walls.

### 3.2 Velocity Profiles

Velocity profiles were extracted at three locations downstream of the step: $x/h = 4$ (recirculation), $x/h = 7$ (near reattachment), and $x/h = 10$ (recovery).

![Velocity Profiles](images/velocity_profiles_comparison.png)

**Quantitative Comparison**:
- **$x/h = 4.0$**: The LES captures the reverse flow near the wall and the acceleration of the freestream jet. The peak velocity agrees closely ($\sim 1.44$ LES vs $\sim 1.49$ DNS).
- **$x/h = 7.0$**: The shear layer spread rate is captured well, though the LES profile is slightly more diffusive, which is expected due to the added eddy viscosity.
- **$x/h = 10.0$**: In the recovery region, the profiles align exceptionally well, indicating the LES predicts the correct reattachment physics.

### 3.3 Performance Metrics

| Metric | DNS | LES | Speedup / Reduction |
|--------|-----|-----|---------------------|
| **Runtime** | 73,132 s (~20.3 hrs) | 1,054 s (17.6 min) | **69.4× Faster** |
| **Throughput** | ~1.6 steps/s | ~114 steps/s | **71× Higher** |
| **Data Size** | ~55 MB / checkpoint | ~2 MB / checkpoint | **27× Smaller** |

---

## 4. Conclusion

The LES implementation has been successfully validated against DNS results. The massive reduction in computational cost (from ~20 hours to ~18 minutes) makes the LES solver a viable tool for parametric exploration and higher Reynolds number studies (e.g., Re = 10,000+) where DNS would be computationally prohibitive.
