# 2D Flow Over a Backward-Facing Step

A numerical simulation of incompressible viscous flow over a backward-facing step, implemented using the Finite Difference Method. This extends the lid-driven cavity flow solver to handle more complex geometries with inlet/outlet conditions.

## Acknowledgment

This implementation builds upon the methodology from Lorena A. Barba's [CFD Python](https://github.com/barbagroup/CFDPython) course and validates against the experimental benchmark of **Armaly et al. (1983)**.

---

## 1. Physical Problem

### 1.1 Description

Flow enters a narrow inlet channel and encounters an abrupt expansion at the step. The sudden expansion causes the flow to **separate** from the bottom wall, creating a **recirculation zone** behind the step. Downstream, the flow **reattaches** to the wall.

```
                              L_total = 17.85
    ├──────────────────────────────────────────────────────────────┤
    
                    x = 0 (step location)
                      │
    ┌─────────────────┼────────────────────────────────────────────┐ ─┬─
    │                 │                                            │  │
    │   INLET         │              OUTLET CHANNEL                │  │ H = 1.0
    │  CHANNEL        │                                            │  │
    │   (h_in)        │     ↻ Recirculation      → Reattachment    │  │
    ├─────────────────┘       Zone                                 │ ─┼─ h = 0.51
    │       STEP                    ↑                              │  │
    │      (solid)              X_r (reattachment length)          │  │
    └──────────────────────────────────────────────────────────────┘ ─┴─
    
    ├──── L_in ──────┤├────────────── L_out ───────────────────────┤
         (≈ 2.55)                  (≈ 15.3)
```

### 1.2 Key Physics

| Phenomenon | Description |
|------------|-------------|
| **Separation** | Flow detaches at step corner due to adverse pressure gradient |
| **Recirculation** | Reversed flow region behind step |
| **Reattachment** | Flow returns to wall downstream; location $X_r/h$ depends on $Re$ |
| **Corner vortex** | Secondary vortex at step corner (at higher $Re$) |

### 1.3 Why This Problem?

The backward-facing step is a classic CFD benchmark because:
- Simple geometry with rich physics
- Well-documented experimental data (Armaly et al., 1983)
- Tests solver's ability to handle separation/reattachment
- Natural progression from enclosed cavity to open flows

---

## 2. Geometry (Armaly Benchmark)

We use the geometry from **Armaly et al. (1983)** for validation:

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Channel height | $H$ | 1.0 | Normalized total height |
| Step height | $h$ | 0.51 | Step height ($h/H = 0.51$) |
| Inlet height | $H - h$ | 0.49 | Inlet channel height |
| Expansion ratio | $ER$ | 1.94 | $ER = H/(H-h)$ |
| Inlet length | $L_{in}$ | $5h ≈ 2.55$ | For flow development |
| Outlet length | $L_{out}$ | $30h ≈ 15.3$ | To capture reattachment |
| Total length | $L_{total}$ | ≈ 17.85 | $L_{in} + L_{out}$ |

### Reynolds Number Definition

For backward-facing step flows, $Re$ is based on **step height** and **average inlet velocity**:

$$Re = \frac{U_{avg} \cdot h}{\nu}$$

where $U_{avg} = \frac{2}{3}U_{max}$ for a parabolic inlet profile.

---

## 3. Governing Equations

The 2D incompressible Navier-Stokes equations:

### Continuity (Mass Conservation)
$$\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0$$

### Momentum (x-direction)
$$\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} + v\frac{\partial u}{\partial y} = -\frac{1}{\rho}\frac{\partial p}{\partial x} + \nu\left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right)$$

### Momentum (y-direction)
$$\frac{\partial v}{\partial t} + u\frac{\partial v}{\partial x} + v\frac{\partial v}{\partial y} = -\frac{1}{\rho}\frac{\partial p}{\partial y} + \nu\left(\frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2}\right)$$

**Variables:**
- $u, v$: velocity components
- $p$: pressure
- $\rho$: density (= 1)
- $\nu$: kinematic viscosity

---

## 4. Boundary Conditions

| Boundary | Location | Velocity | Pressure |
|----------|----------|----------|----------|
| **Inlet** | $x = -L_{in}$ | Parabolic: $u(y) = U_{max}\frac{4y(H-h-y)}{(H-h)^2}$, $v=0$ | $\frac{\partial p}{\partial x} = 0$ |
| **Outlet** | $x = L_{out}$ | $\frac{\partial u}{\partial x} = 0$, $\frac{\partial v}{\partial x} = 0$ | $p = 0$ (reference) |
| **Top wall** | $y = H$ | $u = 0$, $v = 0$ (no-slip) | $\frac{\partial p}{\partial y} = 0$ |
| **Bottom wall** | $y = 0$, $x > 0$ | $u = 0$, $v = 0$ (no-slip) | $\frac{\partial p}{\partial y} = 0$ |
| **Step surface** | Step region | $u = 0$, $v = 0$ (no-slip) | $\frac{\partial p}{\partial n} = 0$ |

### Inlet Profile

The inlet uses a fully-developed parabolic (Poiseuille) profile:

$$u(y) = U_{max} \cdot 4 \cdot \frac{y}{h_{in}} \cdot \left(1 - \frac{y}{h_{in}}\right)$$

where $h_{in} = H - h = 0.49$ is the inlet channel height.

---

## 5. Numerical Method

### 5.1 Discretization

Same approach as the cavity flow solver:

| Term | Scheme |
|------|--------|
| Time | Explicit Euler (forward difference) |
| Advection | Conditional Upwinding (handles periodicity/recurculation) |
| Diffusion | Central difference |
| Pressure gradient | Central difference |
| Pressure Poisson | Iterative Jacobi (50 iterations) |

### 5.2 Grid

We use a **collocated grid** where $u$, $v$, $p$ are stored at cell centers.

| Grid | $N_x$ | $N_y$ | $\Delta x$ | $\Delta y$ |
|------|-------|-------|------------|------------|
| Coarse | 201 | 51 | 0.089 | 0.020 |
| Medium | 401 | 101 | 0.045 | 0.010 |
| Fine | 801 | 201 | 0.022 | 0.005 |

### 5.3 Step Masking

The step is enforced using a **mask array**:

```
  j (y-index)
  ↑
  │  ┌─────────────────────────────────────────────────────┐
Ny-1 │                    FLUID (mask = 1)                   │
  │  ├────────────────┐                                    │
j_step   SOLID (mask = 0)│          FLUID (mask = 1)            │
  0  └────────────────┴────────────────────────────────────┘
     0              i_step                              Nx-1
```

After each velocity update: `u = u * mask`, `v = v * mask`

### 5.4 Stability Criteria

Same as cavity flow:
- **CFL condition**: $\Delta t \left(\frac{|u|}{\Delta x} + \frac{|v|}{\Delta y}\right) < 1$
- **Diffusion condition**: $\frac{\nu \Delta t}{\Delta x^2} < 0.25$

---

## 6. Validation: Armaly et al. (1983)

The primary validation metric is the **reattachment length** $X_r/h$:

| $Re$ | $X_r/h$ (Experimental) | $X_r/h$ (Simulated) |
|------|------------------------|---------------------|
| 100 | ~3.0 - 3.5 | 3.85 |
| 200 | ~5.0 | 5.25 |
| 400 | ~8.0 | *Requires longer integration* |
| 800 | ~9.0 | - |

The reattachment point is where the **wall shear stress** $\tau_w = \mu \frac{\partial u}{\partial y}\big|_{y=0}$ changes sign from negative to positive.

---

## 7. Usage

### NumPy Implementation
```bash
# Default: Re=100, coarse grid
python simulations/backward_facing_step/backward_facing_step.py

# Custom Reynolds number
python simulations/backward_facing_step/backward_facing_step.py --re 200 --nt 10000

# Verbose stability monitoring
python simulations/backward_facing_step/backward_facing_step.py --re 100 --verbose
```

### JAX Implementation (Accelerated)
```bash
# Auto-detect backend
python simulations/backward_facing_step/backward_facing_step_jax.py --re 200

# Force CPU
JAX_PLATFORMS=cpu python simulations/backward_facing_step/backward_facing_step_jax.py
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--nx` | 201 | Grid points in x |
| `--ny` | 51 | Grid points in y |
| `--nt` | 5000 | Time steps |
| `--re` | 100 | Reynolds number |
| `--dt` | 0.001 | Time step size |
| `--verbose` | False | Real-time monitoring |
| `--output-dir` | outputs/baseline | Output directory |

---

## 8. Output

```
simulations/backward_facing_step/
├── outputs/
│   ├── solution_numpy.npz        # u, v, p arrays
│   ├── reattachment_data.csv     # X_r/h vs Re
│   └── wall_shear_stress.csv     # τ_w along bottom wall
└── plots/
    ├── backward_step_result.png  # Pressure + streamlines
    ├── wall_shear_stress.png     # τ_w showing reattachment
    └── armaly_validation.png     # Comparison with benchmark
```

---

## 9. References

1. **Armaly, B.F., Durst, F., Pereira, J.C.F., Schönung, B.** (1983). "Experimental and theoretical investigation of backward-facing step flow." *Journal of Fluid Mechanics*, 127, 473-496.

2. **Barba, L.A., Forsyth, G.F.** (2018). CFD Python: the 12 steps to Navier-Stokes equations. *Journal of Open Source Education*, 1(9), 21. [GitHub](https://github.com/barbagroup/CFDPython)
