# Common Utilities for Numerical Experiments

This directory contains shared infrastructure for running standardized experiments across different solver implementations (Explicit, Implicit, JAX, NumPy, etc.).

## 1. Solver Abstraction (`solvers.py`)

To ensure all solvers can be tested using the same experiments (e.g., Ghia Benchmark), we use a **lightweight functional registry**.

### The Concept

Instead of complex classes, we define a strict data contract:

1.  **Input:** A `SimulationConfig` object (Dataclass).
2.  **Output:** A `SimulationResult` object (Dataclass).
3.  **Process:** A simple function `run(config) -> result`.

### Data Structures

```python
@dataclass
class SimulationConfig:
    """Standardized input parameters for any 2D cavity flow simulation."""
    nx: int              # Grid points in X (and Y)
    re: float            # Reynolds number
    dt: float            # Time step size
    t_end: float         # Physical duration of simulation
    
    # Optional
    max_steps: int = None # Safety limit
    verbose: bool = False
    
@dataclass
class SimulationResult:
    """Standardized output for analysis and plotting."""
    u: np.ndarray        # Final Velocity X (ny, nx)
    v: np.ndarray        # Final Velocity Y (ny, nx)
    p: np.ndarray        # Final Pressure (ny, nx)
    t_end: float         # Actual physical time reached
    steps: int           # Number of steps taken
    duration_sec: float  # Wall-clock execution time
    meta: dict           # Solver-specific metadata (e.g., info about accelerator)
```

### The Registry

We maintain a dictionary mapping friendly names to wrapper functions:

```python
SOLVER_REGISTRY = {
    "jax_explicit": run_jax_explicit,
    "numpy_implicit": run_numpy_implicit,
    "jax_implicit": run_jax_implicit,
}

def get_solver(name: str):
    """Returns the wrapper function for the named solver."""
    ...
```

### How to Add a New Solver?

You don't need to rewrite your solver class! Just add a small wrapper function in `solvers.py`:

```python
def run_my_cool_solver(config: SimulationConfig) -> SimulationResult:
    # 1. Unpack config
    dx = 2.0 / (config.nx - 1)
    nu = 2.0 / config.re
    
    # 2. Call your existing code
    # (Assuming your solver has a function-based API)
    u, v, p = my_solver.solve(nx=config.nx, dt=config.dt, ...)
    
    # 3. Pack results
    return SimulationResult(u=u, v=v, p=p, ...)
```

## 2. Standardized Plotting (`plotting.py`)

*(To be implemented)*

Will provide a function `plot_comparison(results: list[SimulationResult], labels: list[str])` to generate side-by-side plots with:
- Correct physical time labels ($t=...$).
- Reynolds number annotation.
- Unified colormap limits (to make visual comparison valid).
