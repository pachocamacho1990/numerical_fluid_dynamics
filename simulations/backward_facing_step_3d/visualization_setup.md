# Visualization Setup Guide

## Overview

The 3D isosurface visualization requires **PyVista**, which has specific compatibility requirements with VTK and Python.

We use a dedicated virtual environment (`venv_vis`) running **Python 3.10 - 3.13** to ensure stability, as the default Python 3.14 environment requires an unstable VTK Release Candidate incompatible with PyVista.

## Setup Instructions

### 1. Automatic Setup (Recommended)
We provide a script to automatically verify your Python version and set up the environment:

```bash
./setup_vis_env.sh
```

This will:
1.  Find a compatible Python version (3.11 recommended).
2.  Create the `venv_vis` virtual environment.
3.  Install all required dependencies (`pyvista`, `numpy`, `imageio`, etc.).

---

### 2. Manual Setup (Alternative)
If you prefer to set it up manually:

```bash
# 1. Create venv with Python 3.11 (or 3.10/3.12)
# Note: Do NOT use Python 3.14
/opt/homebrew/bin/python3.11 -m venv venv_vis

# 2. Install dependencies
venv_vis/bin/pip install pyvista matplotlib numpy imageio
```

---

### 3. Running Visualization
Use the dedicated script `visualize_3d_isosurfaces.py` with the new environment:

```bash
# Interactive Mode (Recommended)
venv_vis/bin/python simulations/backward_facing_step_3d/visualize_3d_isosurfaces.py --interactive

# Generate Animation (Off-screen)
venv_vis/bin/python simulations/backward_facing_step_3d/visualize_3d_isosurfaces.py --q-criterion --rotate
```

### 4. Output
- Images/Animations are saved to: `simulations/backward_facing_step_3d/outputs/`

---

## Technical Details (Troubleshooting)

| Package | Version | Reason |
|---------|---------|--------|
| Python | 3.10-3.13 | Python 3.14 forces VTK 9.6.0rc2 (incompatible with PyVista) |
| PyVista | 0.46.5 | Requires stable VTK |
| VTK | 9.5.x | Validated stable version |

If you see `ImportError: cannot import name 'vtkCapsuleSource'`, it means you are likely running with Python 3.14. Switch to the `venv_vis` environment.
