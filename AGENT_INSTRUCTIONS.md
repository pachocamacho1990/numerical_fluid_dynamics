# Agent Instructions

This document provides optimized instructions for AI agents working on the `numerical_fluid_dynamics` repository. Follow these guidelines to ensure consistency, code quality, and proper documentation rendering.

## 1. Repository Structure

*   **`simulations/`**: Contains all simulation code.
    *   Each distinct problem (e.g., `cavity_flow`) gets its own subdirectory.
    *   **MUST** include a `README.md` with description, usage, and result images.
    *   **MUST** include the main Python script (e.g., `cavity_flow.py`).
    *   **Output Organization**:
        *   `outputs/` - Data files (`.npz`, `.csv`) - **gitignored**
        *   `plots/` - Visualization outputs (`.png`) - **tracked in git**
*   **`docs/`**: Contains mathematical derivations and theoretical background.
    *   Filename format: `topic_method.md` (e.g., `navier_stokes_fdm.md`).

## 2. Coding Standards (Python)

*   **Dependencies**: Use `numpy` for calculations, `matplotlib` for visualization, `jax` for acceleration.
*   **Math Documentation**: Include docstrings in functions that explain the mathematical equation being solved.
*   **Output Paths**:
    *   **Data files** (`.npz`, `.csv`): Save to `outputs/` subdirectories
    *   **Plots** (`.png`, `.gif`): Save to `plots/` subdirectory
    *   **Pattern**:
        ```python
        import os
        base_dir = "simulations/cavity_flow"
        output_dir = os.path.join(base_dir, "outputs/baseline")
        plots_dir = os.path.join(base_dir, "plots")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        ```
*   **Command-Line Arguments**: Use `argparse` for configurable parameters:
    *   `--nx`, `--nt`, `--dt`, `--re` for simulation parameters
    *   `--output-dir` for data output location
    *   `--input-dir` for data input location (plotting scripts)
    *   `--benchmark` for minimal output mode
*   **Animation**: If creating animations, save to `plots/` directory.

## 3. Documentation & LaTeX Standards

We use GitHub Flavored Markdown. GitHub's MathJax renderer is strict. To ensure equations render correctly:

*   **Block Equations (`$$`)**:
    *   **ALWAYS** add a blank line **before** and **after** the `$$` block.
    *   **DO NOT** indent the `$$` block inside lists. Break the list, insert the equation at the top level, and resume the list if necessary.
    *   **Format**:
        ```markdown
        The resulting equation is:

        $$
        \frac{\partial u}{\partial t} = \nu \nabla^2 u
        $$

        Where $\nu$ is viscosity.
        ```
*   **Inline Equations**: Use single `$`. Example: `$u_{i,j}$`.
*   **Images**: Reference plots using relative paths: `![Description](plots/image.png)`

## 4. Workflow for New Simulations

1.  **Plan**: Define the physics and the numerical method in a Plan artifact.
2.  **Derive**: Create a file in `docs/` detailing the discretization steps.
3.  **Implement**: 
    *   Create the folder in `simulations/`
    *   Write NumPy and JAX implementations
    *   Add `--output-dir` and `--input-dir` arguments
    *   Implement visualization saving to `plots/`
4.  **Verify**: Run the script, check for stability, and verify physical realism.
5.  **Document**: Update README with usage, arguments, and results.
6.  **Git**: Commit with clear messages. Plots are tracked, data files are ignored.

## 5. Benchmarking

*   Create shell scripts (`run_benchmark.sh`) for comparing implementations
*   Use `--benchmark` flag for minimal output
*   Use `--output-dir` to organize different experiment types
*   Generate comparison plots in `plots/` directory
