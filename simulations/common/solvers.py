
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, Optional
import numpy as np
import sys
import os
import math

# Add parend directories to path to allow importing sibling modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sim_dir = os.path.dirname(current_dir)
if sim_dir not in sys.path:
    sys.path.append(sim_dir)

# Delayed imports to avoid circular dependencies or import errors if JAX is missing
# We will import inside the wrapper functions

@dataclass
class SimulationConfig:
    """Standardized input parameters for any 2D cavity flow simulation."""
    nx: int              # Grid points in X (and Y)
    re: float            # Reynolds number
    dt: float            # Time step size
    t_end: float         # Physical duration of simulation
    
    # Optional
    max_steps: int = 1000000 # Safety limit
    verbose: bool = False
    
    @property
    def nu(self) -> float:
        """Kinematic viscosity derived from Re (L=2.0 standard)."""
        return 2.0 / self.re

@dataclass
class SimulationResult:
    """Standardized output for analysis and plotting."""
    u: np.ndarray        # Final Velocity X (ny, nx)
    v: np.ndarray        # Final Velocity Y (ny, nx)
    p: np.ndarray        # Final Pressure (ny, nx)
    t_end: float         # Actual physical time reached
    steps: int           # Number of steps taken
    duration_sec: float  # Wall-clock execution time
    meta: Dict[str, Any] = field(default_factory=dict) # Solver-specific metadata


# --- Wrapper Functions ---

def run_jax_explicit(config: SimulationConfig) -> SimulationResult:
    """Wrapper for the explicit JAX solver."""
    from cavity_flow.cavity_flow_jax import time_step, build_up_b, solve_pressure
    import jax
    import jax.numpy as jnp
    
    # Grid Setup (Standard L=2.0 domain)
    L = 2.0
    dx = L / (config.nx - 1)
    dy = L / (config.nx - 1)
    rho = 1.0
    
    # Initialization
    u = jnp.zeros((config.nx, config.nx))
    v = jnp.zeros((config.nx, config.nx))
    p = jnp.zeros((config.nx, config.nx))
    
    # JAX Compilation Warmup
    if config.verbose:
        print("Wrapper: Compiling JAX Explicit kernels...")
    _ = time_step(u, v, p, config.dt, dx, dy, rho, config.nu, 2)
    
    # Main Loop
    t = 0.0
    step = 0
    start_time = time.time()
    
    # Determine number of steps
    nt = int(config.t_end / config.dt)
    if config.max_steps and nt > config.max_steps:
        nt = config.max_steps
        if config.verbose:
            print(f"Wrapper: Limiting steps to {nt} (requested t_end needs more)")
            
    # Run loop
    # Note: For efficiency, we should ideally use lax.scan or the main loop from the module
    # But for this wrapper, reusing the python loop allows adhering to our exact t_end logic easily.
    # To be extremely efficient, we would refactor cavity_flow_jax to accept t_end.
    # For now, we will run the python loop calling the JIT function.
    
    for i in range(nt):
        u, v, p, cfl = time_step(u, v, p, config.dt, dx, dy, rho, config.nu, 50)
        t += config.dt
        step += 1
        
    # Sync and download
    u_np = np.array(u)
    v_np = np.array(v)
    p_np = np.array(p)
    
    duration = time.time() - start_time
    
    return SimulationResult(
        u=u_np, v=v_np, p=p_np,
        t_end=t, steps=step, duration_sec=duration,
        meta={"backend": "jax_explicit", "device": str(jax.devices()[0])}
    )

def run_numpy_implicit(config: SimulationConfig) -> SimulationResult:
    """Wrapper for the implicit NumPy solver."""
    # Ensure local modules (fractional_step, etc.) can be found
    solver_dir = os.path.join(sim_dir, "cavity_flow_implicit")
    if solver_dir not in sys.path:
        sys.path.append(solver_dir)
        
    # Since we added the dir to sys.path, we import the file directly
    from cavity_flow_implicit import cavity_flow_implicit, initialize_fields
    
    # Grid Setup
    L = 2.0
    dx = L / (config.nx - 1)
    dy = L / (config.nx - 1)
    rho = 1.0
    
    # Initialization
    u, v, p = initialize_fields(config.nx, config.nx)
    
    nt = int(config.t_end / config.dt)
    
    start_time = time.time()
    
    # Run
    # Explicitly disable verbal output unless verbose is on, to keep runner clean
    u, v, p, history, _ = cavity_flow_implicit(
        nt=nt, u=u, v=v, p=p, 
        dt=config.dt, dx=dx, dy=dy, rho=rho, nu=config.nu, 
        verbose=config.verbose
    )
    
    duration = time.time() - start_time
    
    return SimulationResult(
        u=u, v=v, p=p,
        t_end=nt * config.dt, steps=nt, duration_sec=duration,
        meta={"backend": "numpy_implicit"}
    )

def run_jax_implicit(config: SimulationConfig) -> SimulationResult:
    """Wrapper for the implicit JAX solver."""
    # Ensure local modules can be found
    solver_dir = os.path.join(sim_dir, "cavity_flow_implicit")
    if solver_dir not in sys.path:
        sys.path.append(solver_dir)
        
    from cavity_flow_implicit_jax import cavity_flow_implicit_jax, initialize_fields
    import jax
    
    # Grid Setup
    L = 2.0
    dx = L / (config.nx - 1)
    dy = L / (config.nx - 1)
    rho = 1.0
    
    # Initialization
    u, v, p = initialize_fields(config.nx, config.nx)
    
    nt = int(config.t_end / config.dt)
    
    start_time = time.time()
    
    # Run
    # Note: cavity_flow_implicit_jax returns numpy arrays at the end already
    u_np, v_np, p_np, history, _ = cavity_flow_implicit_jax(
        nt=nt, u=u, v=v, p=p, 
        dt=config.dt, dx=dx, dy=dy, rho=rho, nu=config.nu, 
        verbose=config.verbose
    )
    
    duration = time.time() - start_time
    
    return SimulationResult(
        u=u_np, v=v_np, p=p_np,
        t_end=nt * config.dt, steps=nt, duration_sec=duration,
        meta={"backend": "jax_implicit", "device": str(jax.devices()[0])}
    )

# --- Registry ---

SOLVER_REGISTRY = {
    "jax_explicit": run_jax_explicit,
    "numpy_implicit": run_numpy_implicit,
    "jax_implicit": run_jax_implicit,
}

def get_solver(name: str) -> Callable[[SimulationConfig], SimulationResult]:
    """Returns the solver function by name."""
    if name not in SOLVER_REGISTRY:
        raise ValueError(f"Unknown solver '{name}'. Available: {list(SOLVER_REGISTRY.keys())}")
    return SOLVER_REGISTRY[name]
