# Agent Instructions

This document provides optimized instructions for AI agents working on the `numerical_fluid_dynamics` repository. Follow these guidelines to ensure consistency, code quality, and proper documentation rendering.

## 1. Repository Structure

*   **`simulations/`**: Contains all simulation code.
    *   Each distinct problem (e.g., `cavity_flow`) gets its own subdirectory.
    *   **MUST** include a `README.md` with a brief description and result image.
    *   **MUST** include the main Python script (e.g., `cavity_flow.py`).
*   **`docs/`**: Contains mathematical derivations and theoretical background.
    *   Filename format: `topic_method.md` (e.g., `navier_stokes_fdm.md`).

## 2. Coding Standards (Python)

*   **Dependencies**: Use `numpy` for calculations and `matplotlib` for visualization.
*   **Math Documentation**: Include docstrings in functions that explain the mathematical equation being solved using clear ASCII or flattened text representation.
*   **Output Paths**:
    *   **CRITICAL**: Always save generated files (images, GIFs) relative to the script's location, NOT the current working directory.
    *   Use this pattern:
        ```python
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, "filename.png")
        plt.savefig(output_path)
        ```
*   **Animation**: If creating animations, provide both a static frame save and a GIF generation option (or dedicated script).

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

## 4. Workflow for New Simulations

1.  **Plan**: Define the physics and the numerical method in a Plan artifact.
2.  **Derive**: Create a file in `docs/` detailing the discretization steps.
3.  **Implement**: Create the folder in `simulations/`, write the script, and implement visualization.
4.  **Verify**: Run the script, check for stability, and verify the physical realism of the results (vortices, boundary layers, etc.).
5.  **Git**: Commit with clear messages.
