# Repository Setup Walkthrough

I have successfully established the permanent repository at `/Users/pacho-home-server/numerical_fluid_dynamics`.

## Structure Created
*   **`simulations/cavity_flow/`**: Contains:
    *   `cavity_flow.py` (NumPy implementation).
    *   `cavity_flow_jax.py` (JAX-optimized implementation).
*   **`docs/`**: `navier_stokes_fdm.md` (Theory) & `benchmarking_guide.md` (Testing).
*   **`AGENT_INSTRUCTIONS.md`**: Guide for AI agents.
*   **Environments**: `venv` (CPU) & `venv_metal` (GPU).
*   **Tools**:
    *   `run_benchmark.sh`: Master benchmark script.
    *   **Resource Estimator**: New `--check` flag to calculate stability limits.

## Flow Evolution Animation
![Flow Evolution](/Users/pacho-home-server/numerical_fluid_dynamics/simulations/cavity_flow/cavity_flow_evolution.gif)

## Performance Benchmarks (Apple Silicon)

We pushed the hardware to its limits with increasing grid sizes.

| Grid Size | Method | Backend | Speed (it/s) | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **41x41** | NumPy | CPU | ~1,290 | |
| | **JAX JIT** | **CPU** | **~3,880** | **Best for prototypes** |
| | JAX Metal | GPU | ~632 | |
| | | | | |
| **201x201** | NumPy | CPU | ~40 | |
| | **JAX JIT** | **CPU** | **~430** | **10x Faster than NumPy** |
| | JAX Metal | GPU | ~252 | |
| | | | | |
| **512x512** | NumPy | CPU | ~22 | *Unstable (needs more solver iter)* |
| | **JAX JIT** | **CPU** | **~84** | **Still likely the winner** |
| | JAX Metal | GPU | ~52 | Closing gap, but API overhead remains high |

**Key Findings**:
1.  **CPU is King (for 2D FD)**: The Apple Silicon CPU with JAX JIT compilation is incredibly efficient for this type of memory-bandwidth-bound 2D stencil code.
2.  **Metal Overhead**: The `jax-metal` backend has a fixed dispatch cost that outweighs its compute power until grid sizes become massive (likely $> 1024^2$).
3.  **Stability Matters**: The new `--check` tool effectively prevents wasted runs by calculating the strict CFL/Diffusion limits ($\Delta t \propto \Delta x^2$).

## Version Control
*   **Remote**: `origin` -> `github.com:pachocamacho1990/numerical_fluid_dynamics.git`
*   **Branch**: `main`
