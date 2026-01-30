# LES Implementation Plan: Large Eddy Simulation with Smagorinsky Subgrid Model

## Executive Summary

This document outlines a comprehensive plan to extend the current 3D implicit Navier-Stokes solver with **Large Eddy Simulation (LES)** capabilities using the **Smagorinsky subgrid-scale model**. This upgrade enables industrially-relevant Reynolds numbers (Re = 10^4 - 10^6) while maintaining computational tractability and setting the foundation for future ML-based turbulence closure models.

### Goals
1. Implement LES with Smagorinsky SGS model in the existing JAX framework
2. Enable simulations at Re = 10,000 - 100,000+ (industrial regime)
3. Validate against experimental and DNS data
4. Create infrastructure for future ML turbulence closure integration

### Current State
- **Solver**: 3D Implicit Fractional Step with Crank-Nicolson diffusion (ADI)
- **Max Tested Re**: 2,000 (transitional/turbulent)
- **Approach**: Implicit LES (ILES) - numerical dissipation acts as implicit SGS
- **Limitation**: No explicit turbulence model; resolution requirements scale as Re^(9/4) for DNS

---

## 1. Background: Why LES?

### 1.1 The Turbulence Resolution Problem

| Approach | Grid Scaling | Re=10,000 Grid | Re=100,000 Grid | Feasibility |
|----------|--------------|----------------|-----------------|-------------|
| **DNS** | N ~ Re^(9/4) | ~10^9 cells | ~10^12 cells | Supercomputer |
| **LES** | N ~ Re^(1.8) | ~10^6 cells | ~10^8 cells | Workstation |
| **RANS** | N ~ 10^5-6 | ~10^5 cells | ~10^5 cells | Laptop |

DNS resolves all scales down to the Kolmogorov length η ~ Re^(-3/4). This is computationally prohibitive for industrial applications.

### 1.2 LES Philosophy

LES resolves the **energy-containing eddies** (large scales) while **modeling** the effect of small scales (subgrid scales):

```
Total Flow = Resolved (Computed) + Subgrid (Modeled)
    ũ     =      ū           +        u'
```

The grid acts as an implicit spatial filter with cutoff at Δ (filter width):
- Scales > Δ: Directly computed
- Scales < Δ: Modeled via SGS stress tensor

### 1.3 Subgrid-Scale Stress Tensor

The filtered Navier-Stokes equations introduce the **SGS stress tensor**:

$$
\tau_{ij}^{sgs} = \overline{u_i u_j} - \bar{u}_i \bar{u}_j
$$

This term represents momentum transfer from unresolved to resolved scales and must be modeled.

---

## 2. Smagorinsky Model Theory

### 2.1 Eddy Viscosity Hypothesis

The Smagorinsky model (1963) uses the **Boussinesq eddy viscosity** assumption:

$$
\tau_{ij}^{sgs} - \frac{1}{3}\tau_{kk}\delta_{ij} = -2\nu_t \bar{S}_{ij}
$$

where $\bar{S}_{ij}$ is the resolved strain-rate tensor:

$$
\bar{S}_{ij} = \frac{1}{2}\left(\frac{\partial \bar{u}_i}{\partial x_j} + \frac{\partial \bar{u}_j}{\partial x_i}\right)
$$

### 2.2 Smagorinsky Eddy Viscosity

The turbulent (eddy) viscosity is:

$$
\nu_t = (C_s \Delta)^2 |\bar{S}|
$$

where:
- $C_s$ = Smagorinsky constant (typically 0.1 - 0.2)
- $\Delta$ = Filter width (grid scale)
- $|\bar{S}| = \sqrt{2\bar{S}_{ij}\bar{S}_{ij}}$ = Strain rate magnitude

### 2.3 Filter Width Definition

For anisotropic grids:

$$
\Delta = (\Delta x \cdot \Delta y \cdot \Delta z)^{1/3}
$$

Or the maximum:

$$
\Delta = \max(\Delta x, \Delta y, \Delta z)
$$

The cubic root formulation is preferred for accuracy.

### 2.4 Effective Viscosity

The total viscosity becomes:

$$
\nu_{eff} = \nu + \nu_t = \nu + (C_s \Delta)^2 |\bar{S}|
$$

This is **spatially and temporally varying** - computed at each grid point and time step.

### 2.5 Van Driest Damping (Wall Treatment)

Near walls, the Smagorinsky model over-predicts $\nu_t$. The Van Driest damping function corrects this:

$$
\nu_t = (C_s \Delta)^2 \cdot \left[1 - \exp\left(-\frac{y^+}{A^+}\right)\right]^2 \cdot |\bar{S}|
$$

where:
- $y^+ = y u_\tau / \nu$ = Wall distance in wall units
- $u_\tau = \sqrt{\tau_w / \rho}$ = Friction velocity
- $A^+ \approx 25$ = Van Driest constant

### 2.6 Strain Rate Tensor Components (3D)

$$
\bar{S}_{11} = \frac{\partial \bar{u}}{\partial x}, \quad
\bar{S}_{22} = \frac{\partial \bar{v}}{\partial y}, \quad
\bar{S}_{33} = \frac{\partial \bar{w}}{\partial z}
$$

$$
\bar{S}_{12} = \bar{S}_{21} = \frac{1}{2}\left(\frac{\partial \bar{u}}{\partial y} + \frac{\partial \bar{v}}{\partial x}\right)
$$

$$
\bar{S}_{13} = \bar{S}_{31} = \frac{1}{2}\left(\frac{\partial \bar{u}}{\partial z} + \frac{\partial \bar{w}}{\partial x}\right)
$$

$$
\bar{S}_{23} = \bar{S}_{32} = \frac{1}{2}\left(\frac{\partial \bar{v}}{\partial z} + \frac{\partial \bar{w}}{\partial y}\right)
$$

Strain rate magnitude:
$$
|\bar{S}| = \sqrt{2(S_{11}^2 + S_{22}^2 + S_{33}^2 + 2S_{12}^2 + 2S_{13}^2 + 2S_{23}^2)}
$$

---

## 3. Implementation Phases

### Phase 1: Core LES Infrastructure (Foundation)

**Objective**: Add eddy viscosity computation without changing solver structure

#### 1.1 Create SGS Module
- New file: `simulations/les/sgs_models.py`
- Implement strain rate tensor computation
- Implement Smagorinsky eddy viscosity
- Implement filter width calculation

#### 1.2 Modify Diffusion Term
- Replace constant `nu` with `nu_eff(x,y,z,t)`
- Update ADI solver to accept spatially-varying viscosity
- Ensure stability with variable coefficients

#### 1.3 Deliverables
- [ ] `compute_strain_rate_tensor_3d()` - 6 components
- [ ] `compute_strain_magnitude()` - |S|
- [ ] `compute_smagorinsky_viscosity()` - ν_t
- [ ] `compute_effective_viscosity()` - ν + ν_t
- [ ] Unit tests for all functions

---

### Phase 2: Variable-Viscosity ADI Solver

**Objective**: Extend ADI to handle non-constant diffusion coefficients

#### 2.1 Modified Diffusion Equation

Current (constant ν):
$$
\frac{\partial u}{\partial t} = \nu \nabla^2 u
$$

LES (variable ν_eff):
$$
\frac{\partial u}{\partial t} = \nabla \cdot (\nu_{eff} \nabla u)
$$

In expanded form:
$$
\frac{\partial u}{\partial t} = \frac{\partial}{\partial x}\left(\nu_{eff}\frac{\partial u}{\partial x}\right) + \frac{\partial}{\partial y}\left(\nu_{eff}\frac{\partial u}{\partial y}\right) + \frac{\partial}{\partial z}\left(\nu_{eff}\frac{\partial u}{\partial z}\right)
$$

#### 2.2 Discrete Form

Using central differences with interface viscosities:

$$
\frac{\partial}{\partial x}\left(\nu_{eff}\frac{\partial u}{\partial x}\right) \approx \frac{1}{\Delta x}\left[\nu_{i+1/2}\frac{u_{i+1} - u_i}{\Delta x} - \nu_{i-1/2}\frac{u_i - u_{i-1}}{\Delta x}\right]
$$

where $\nu_{i+1/2} = \frac{1}{2}(\nu_i + \nu_{i+1})$ (harmonic or arithmetic mean).

#### 2.3 Modified Tridiagonal Coefficients

For the X-sweep:
$$
a_i = -\frac{\alpha \cdot \nu_{i-1/2}}{\Delta x^2}, \quad
b_i = 1 + \alpha\left(\frac{\nu_{i-1/2} + \nu_{i+1/2}}{\Delta x^2}\right), \quad
c_i = -\frac{\alpha \cdot \nu_{i+1/2}}{\Delta x^2}
$$

#### 2.4 Deliverables
- [ ] `build_tridiagonal_variable_nu()` - Modified Thomas coefficients
- [ ] `adi_step_3d_les()` - Variable-viscosity 3D ADI
- [ ] Verification against constant-ν case (should match when ν_t = 0)

---

### Phase 3: Wall Treatment

**Objective**: Implement Van Driest damping for accurate near-wall behavior

#### 3.1 Wall Distance Computation
- Pre-compute wall distance field d(x,y,z) for all grid points
- Handle step geometry (distance to nearest wall including step surfaces)

#### 3.2 Friction Velocity Estimation
- Compute wall shear stress τ_w from velocity gradients
- Update u_τ iteratively or use log-law approximation

#### 3.3 Van Driest Function
```python
def van_driest_damping(y_plus, A_plus=25.0):
    return (1.0 - jnp.exp(-y_plus / A_plus))**2
```

#### 3.4 Deliverables
- [ ] `compute_wall_distance_3d()` - Distance field
- [ ] `compute_friction_velocity()` - u_τ estimation
- [ ] `apply_van_driest_damping()` - Near-wall correction
- [ ] Validation: u+ vs y+ profile should follow law of the wall

---

### Phase 4: Integration & Testing

**Objective**: Full LES solver integration and validation

#### 4.1 New Solver: `solver_3d_les.py`
- Integrate all components into unified time-stepping
- Add diagnostic outputs (ν_t field, SGS dissipation)

#### 4.2 Configuration System
```python
@dataclass
class LESConfig:
    sgs_model: str = "smagorinsky"  # or "dynamic", "wale", "ml"
    cs: float = 0.1                  # Smagorinsky constant
    wall_damping: bool = True        # Van Driest damping
    a_plus: float = 25.0            # Van Driest constant
```

#### 4.3 Deliverables
- [ ] `solver_3d_les.py` - Main LES solver
- [ ] `run_les_3d.py` - Runner script with LES options
- [ ] Integration tests
- [ ] Performance benchmarks

---

### Phase 5: Validation & Benchmarking

**Objective**: Validate LES implementation against reference data

#### 5.1 Channel Flow (Primary Validation)
- **Case**: Turbulent channel flow Re_τ = 180, 395, 590
- **Reference**: DNS data from Moser, Kim & Mansour (1999)
- **Metrics**:
  - Mean velocity profile (u+ vs y+)
  - Reynolds stresses (u'u', v'v', u'v')
  - Law of the wall compliance

#### 5.2 Backward-Facing Step (Extended Validation)
- **Case**: BFS at Re = 5,000, 10,000, 37,000
- **Reference**: Experiments (Driver & Seegmiller, 1985)
- **Metrics**:
  - Reattachment length X_r/h
  - Mean velocity profiles
  - Turbulence intensities

#### 5.3 Deliverables
- [ ] Channel flow validation case
- [ ] Extended BFS validation (Re > 5000)
- [ ] Validation report with plots

---

## 4. Detailed Code Architecture

### 4.1 New File Structure

```
simulations/
├── les/
│   ├── __init__.py
│   ├── sgs_models.py          # SGS model implementations
│   │   ├── compute_strain_rate_tensor_3d()
│   │   ├── compute_strain_magnitude()
│   │   ├── SmagorinskyModel class
│   │   ├── DynamicSmagorinskyModel class (future)
│   │   └── MLClosure class (future placeholder)
│   │
│   ├── wall_treatment.py       # Near-wall modeling
│   │   ├── compute_wall_distance()
│   │   ├── compute_friction_velocity()
│   │   └── van_driest_damping()
│   │
│   ├── variable_viscosity_adi.py  # Modified ADI solver
│   │   ├── build_tridiagonal_variable_nu()
│   │   └── adi_step_3d_les()
│   │
│   ├── solver_3d_les.py        # Main LES solver
│   │   └── time_step_3d_les()
│   │
│   ├── diagnostics.py          # LES-specific outputs
│   │   ├── compute_sgs_dissipation()
│   │   ├── compute_resolved_tke()
│   │   └── compute_energy_spectrum()
│   │
│   └── config.py               # LES configuration
│       └── LESConfig dataclass
│
├── backward_facing_step_3d/    # Existing (unchanged)
│   └── solver_3d.py            # Original DNS/ILES solver
│
└── channel_flow_3d/            # New validation case
    ├── geometry_channel.py
    ├── run_channel_les.py
    └── validate_channel.py
```

### 4.2 Core Functions Specification

#### `compute_strain_rate_tensor_3d(u, v, w, dx, dy, dz)`
```python
@jit
def compute_strain_rate_tensor_3d(u, v, w, dx, dy, dz):
    """
    Compute the 6 independent components of the strain rate tensor.

    Returns:
        S11, S22, S33, S12, S13, S23: JAX arrays of shape (nz, ny, nx)
    """
    # Central differences for diagonal components
    S11 = (u[..., 2:] - u[..., :-2]) / (2 * dx)  # du/dx
    S22 = (v[:, 2:, :] - v[:, :-2, :]) / (2 * dy)  # dv/dy
    S33 = (w[2:, ...] - w[:-2, ...]) / (2 * dz)  # dw/dz

    # Off-diagonal (symmetric)
    du_dy = (u[:, 2:, :] - u[:, :-2, :]) / (2 * dy)
    dv_dx = (v[..., 2:] - v[..., :-2]) / (2 * dx)
    S12 = 0.5 * (du_dy + dv_dx)

    du_dz = (u[2:, ...] - u[:-2, ...]) / (2 * dz)
    dw_dx = (w[..., 2:] - w[..., :-2]) / (2 * dx)
    S13 = 0.5 * (du_dz + dw_dx)

    dv_dz = (v[2:, ...] - v[:-2, ...]) / (2 * dz)
    dw_dy = (w[:, 2:, :] - w[:, :-2, :]) / (2 * dy)
    S23 = 0.5 * (dv_dz + dw_dy)

    return S11, S22, S33, S12, S13, S23
```

#### `compute_smagorinsky_viscosity(S11, S22, S33, S12, S13, S23, cs, delta)`
```python
@jit
def compute_smagorinsky_viscosity(S11, S22, S33, S12, S13, S23, cs, delta):
    """
    Compute Smagorinsky eddy viscosity.

    Args:
        S11-S23: Strain rate components
        cs: Smagorinsky constant (typically 0.1-0.2)
        delta: Filter width (scalar or array)

    Returns:
        nu_t: Eddy viscosity field (nz, ny, nx)
    """
    # Strain magnitude |S| = sqrt(2 * Sij * Sij)
    S_mag = jnp.sqrt(2 * (S11**2 + S22**2 + S33**2 +
                          2*S12**2 + 2*S13**2 + 2*S23**2))

    # Smagorinsky viscosity
    nu_t = (cs * delta)**2 * S_mag

    return nu_t
```

#### `time_step_3d_les(u, v, w, p, dt, dx, dy, dz, rho, nu, les_config, ...)`
```python
def time_step_3d_les(u, v, w, p, dt, dx, dy, dz, rho, nu,
                     les_config, mask, u_inlet, geom_params):
    """
    LES time step with Smagorinsky SGS model.

    Main differences from DNS:
    1. Compute strain rate tensor
    2. Compute eddy viscosity nu_t
    3. Use nu_eff = nu + nu_t in diffusion
    4. Apply wall damping near boundaries
    """
    # 1. Compute SGS eddy viscosity
    S11, S22, S33, S12, S13, S23 = compute_strain_rate_tensor_3d(u, v, w, dx, dy, dz)
    delta = (dx * dy * dz)**(1/3)
    nu_t = compute_smagorinsky_viscosity(S11, S22, S33, S12, S13, S23,
                                          les_config.cs, delta)

    # 2. Apply wall damping
    if les_config.wall_damping:
        nu_t = apply_van_driest_damping(nu_t, wall_distance, u, v, w, nu)

    # 3. Effective viscosity
    nu_eff = nu + nu_t

    # 4. Advection (same as DNS)
    adv_u, adv_v, adv_w = compute_advection_3d(u, v, w, dx, dy, dz)

    # 5. Diffusion with VARIABLE viscosity (modified ADI)
    u_star = adi_step_3d_les(u, rhs_u, nu_eff, dt, dx, dy, dz, mask)
    v_star = adi_step_3d_les(v, rhs_v, nu_eff, dt, dx, dy, dz, mask)
    w_star = adi_step_3d_les(w, rhs_w, nu_eff, dt, dx, dy, dz, mask)

    # 6. Pressure & Correction (same as DNS)
    # ...

    return u_new, v_new, w_new, p_new, nu_t, cfl
```

---

## 5. Smagorinsky Constant Calibration

### 5.1 Standard Values

| Flow Type | Recommended C_s |
|-----------|-----------------|
| Isotropic turbulence | 0.17-0.21 |
| Channel flow | 0.1 |
| Free shear flows | 0.12-0.14 |
| Backward-facing step | 0.1-0.12 |

### 5.2 Grid Sensitivity

The Smagorinsky model is sensitive to grid resolution:
- Too coarse: Excessive dissipation, smeared structures
- Recommended: At least 10-20 cells across shear layers
- Guideline: Δ/η < 20-30 (η = Kolmogorov scale)

### 5.3 Dynamic Procedure (Future Enhancement)

The **Dynamic Smagorinsky** model (Germano et al., 1991) computes C_s locally:

$$
C_s^2 = \frac{\langle L_{ij} M_{ij} \rangle}{\langle M_{ij} M_{ij} \rangle}
$$

This removes the need for constant tuning and provides better predictions for transitional flows.

---

## 6. Validation Strategy

### 6.1 Unit Tests (Per-Function)

| Function | Test Case | Expected Result |
|----------|-----------|-----------------|
| `strain_rate_tensor` | Uniform shear | Exact S12 value |
| `strain_rate_tensor` | Solid body rotation | S_ij = 0 |
| `smagorinsky_viscosity` | Zero velocity | nu_t = 0 |
| `variable_nu_adi` | Constant nu | Match original ADI |
| `van_driest` | y+ >> A+ | Damping → 1 |

### 6.2 Integration Tests

1. **Laminar Recovery**: At low Re (Re=100), nu_t << nu, results should match DNS
2. **Conservation**: Mass conservation check with variable viscosity
3. **Stability**: No blow-up at CFL < 1

### 6.3 Validation Cases

#### Case 1: Turbulent Channel Flow (Primary)
- **Configuration**: Re_τ = 180 (Re_bulk ≈ 2,800)
- **Grid**: 64 × 64 × 64 (minimum), 128 × 128 × 128 (recommended)
- **Reference**: Moser, Kim & Mansour (1999) DNS database
- **Metrics**:
  - [ ] Mean velocity profile matches log law
  - [ ] u_rms profile within 10% of DNS
  - [ ] Correct wall shear stress

#### Case 2: Backward-Facing Step (Extended)
- **Configuration**: Re_h = 5,000, 10,000
- **Grid**: 256 × 64 × 64 (minimum)
- **Reference**: Driver & Seegmiller (1985) experiments
- **Metrics**:
  - [ ] Reattachment length within 10% of experiment
  - [ ] Mean velocity profiles at x/h = 4, 6, 8

#### Case 3: Decaying Isotropic Turbulence (Optional)
- **Purpose**: Verify correct energy cascade
- **Metric**: Energy spectrum follows k^(-5/3) in inertial range

---

## 7. Performance Considerations

### 7.1 Computational Overhead

| Operation | Additional Cost | Notes |
|-----------|----------------|-------|
| Strain tensor | ~10% | 6 gradient computations |
| nu_t computation | ~2% | Element-wise operations |
| Variable ADI | ~15-20% | Non-constant coefficients |
| Wall distance | One-time | Pre-computed |
| **Total Overhead** | **~25-30%** | vs. constant-ν solver |

### 7.2 JAX Optimization

- All LES functions should be `@jit` compiled
- Strain tensor: Use single fused kernel
- nu_t: In-place computation where possible
- Consider `jax.checkpoint` for memory-limited cases

### 7.3 GPU Considerations

- Variable viscosity ADI is less amenable to GPU parallelization
- Alternative: Semi-implicit treatment (explicit nu_t, implicit molecular nu)
- For very large grids: Consider multigrid pressure solver

---

## 8. Path to ML Turbulence Closure

### 8.1 Why ML?

The Smagorinsky model has known limitations:
- Requires tuning of C_s for each flow
- Cannot predict backscatter (reverse energy cascade)
- Poor performance in transitional flows

**ML closures** can learn these effects from DNS data.

### 8.2 ML Integration Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     LES Solver                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   ┌──────────────┐         ┌──────────────────────┐    │
│   │ Resolved     │         │  SGS Model           │    │
│   │ Quantities   │────────▶│  ┌──────────────────┐│    │
│   │ (u, v, w, S) │         │  │ Smagorinsky      ││    │
│   └──────────────┘         │  │ Dynamic          ││    │
│                            │  │ **ML Closure** ◀──┼────┼── NEW
│                            │  └──────────────────┘│    │
│                            └──────────┬───────────┘    │
│                                       │                 │
│                                       ▼                 │
│                              ┌────────────────┐        │
│                              │ nu_t or τ_ij   │        │
│                              └────────────────┘        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 8.3 ML Closure Design

**Input Features** (invariant to rotation/reflection):
- Strain rate tensor invariants: I₁ = tr(S), I₂ = tr(S²), I₃ = tr(S³)
- Rotation rate tensor invariants
- Wall distance (normalized)
- Local Reynolds number

**Output Options**:
1. Scalar: Predict C_s or nu_t directly
2. Tensor: Predict full τ_ij (more general, harder to train)

**Training Data**:
- Filtered DNS fields (channel, BFS, jets)
- A-priori: Predict τ_ij from filtered DNS
- A-posteriori: Train in coupled LES loop (differentiable solver!)

### 8.4 JAX Differentiability Advantage

Your JAX solver is already differentiable. This enables:
- End-to-end training of ML closures
- Gradient-based optimization of model parameters
- Adjoint-based sensitivity analysis

**Implementation**:
```python
def ml_closure(features, params):
    """Neural network SGS model."""
    x = features  # (batch, n_invariants)
    for W, b in params[:-1]:
        x = jax.nn.relu(x @ W + b)
    return x @ params[-1][0] + params[-1][1]  # nu_t prediction

# Training loop with differentiable solver
def loss_fn(params, initial_state, target_stats):
    final_state = run_les_with_ml_closure(initial_state, params)
    return mse(compute_statistics(final_state), target_stats)

grads = jax.grad(loss_fn)(params, ...)  # Backprop through entire simulation!
```

---

## 9. Milestones & Dependencies

```
Phase 1: Core LES Infrastructure
├── 1.1 Create sgs_models.py
├── 1.2 Implement strain rate tensor
├── 1.3 Implement Smagorinsky viscosity
└── 1.4 Unit tests
    │
    ▼
Phase 2: Variable-Viscosity ADI
├── 2.1 Derive modified tridiagonal coefficients
├── 2.2 Implement variable_viscosity_adi.py
├── 2.3 Integration test (constant nu → match original)
└── 2.4 Performance benchmark
    │
    ▼
Phase 3: Wall Treatment
├── 3.1 Wall distance computation
├── 3.2 Friction velocity estimation
└── 3.3 Van Driest damping
    │
    ▼
Phase 4: Integration
├── 4.1 Create solver_3d_les.py
├── 4.2 Create run_les_3d.py
├── 4.3 LES config system
└── 4.4 Diagnostics (nu_t output, SGS dissipation)
    │
    ▼
Phase 5: Validation
├── 5.1 Channel flow Re_τ = 180
├── 5.2 Channel flow Re_τ = 395
├── 5.3 BFS Re = 5,000
└── 5.4 Validation report
    │
    ▼
Future: ML Closure
├── F.1 Feature extraction module
├── F.2 Simple MLP closure
├── F.3 A-priori training
└── F.4 A-posteriori training (differentiable solver)
```

---

## 10. Risk Assessment & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Variable-nu ADI instability | High | Medium | Add stability regularization; fall back to explicit diffusion |
| Excessive SGS dissipation | Medium | Medium | Careful Cs calibration; implement dynamic model |
| Wall treatment inaccuracy | Medium | Low | Compare with wall-resolved LES reference |
| Performance degradation | Low | High | Profile & optimize; consider semi-implicit |
| ML closure training divergence | High | Medium | Start with simple MLP; use physics constraints |

---

## 11. References

### LES Fundamentals
1. Smagorinsky, J. (1963). "General circulation experiments with the primitive equations." *Monthly Weather Review*, 91(3), 99-164.
2. Pope, S.B. (2000). *Turbulent Flows*. Cambridge University Press. (Chapters 13-14)
3. Sagaut, P. (2006). *Large Eddy Simulation for Incompressible Flows*. Springer.

### Dynamic Models
4. Germano, M., Piomelli, U., Moin, P., & Cabot, W.H. (1991). "A dynamic subgrid-scale eddy viscosity model." *Physics of Fluids A*, 3(7), 1760-1765.
5. Lilly, D.K. (1992). "A proposed modification of the Germano subgrid-scale closure method." *Physics of Fluids A*, 4(3), 633-635.

### Validation Data
6. Moser, R.D., Kim, J., & Mansour, N.N. (1999). "Direct numerical simulation of turbulent channel flow up to Re_τ = 590." *Physics of Fluids*, 11(4), 943-945.
7. Driver, D.M., & Seegmiller, H.L. (1985). "Features of a reattaching turbulent shear layer in divergent channel flow." *AIAA Journal*, 23(2), 163-171.

### ML Turbulence Modeling
8. Ling, J., Kurzawski, A., & Templeton, J. (2016). "Reynolds averaged turbulence modelling using deep neural networks with embedded invariance." *Journal of Fluid Mechanics*, 807, 155-166.
9. Duraisamy, K., Iaccarino, G., & Xiao, H. (2019). "Turbulence modeling in the age of data." *Annual Review of Fluid Mechanics*, 51, 357-377.
10. Kochkov, D., et al. (2021). "Machine learning–accelerated computational fluid dynamics." *PNAS*, 118(21).

---

## 12. Conclusion

This plan provides a structured path to extend your CFD solver with industry-relevant LES capabilities. The Smagorinsky model serves as both a practical turbulence model and a stepping stone to more advanced ML-based closures.

**Key advantages of this approach**:
1. Leverages existing JAX infrastructure
2. Incremental complexity (can stop at any phase with working code)
3. Clear validation path with established benchmarks
4. Direct path to differentiable ML turbulence modeling

The combination of LES + JAX differentiability positions this solver for cutting-edge AI-CFD research.
