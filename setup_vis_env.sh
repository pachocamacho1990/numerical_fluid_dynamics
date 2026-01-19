#!/bin/bash

# Setup script for 3D Flow Visualization Environment
# Ensures compatible Python version (3.10-3.13) is used for PyVista/VTK

set -e

# Defined allowed python versions (prefer 3.11, accept 3.10-3.13)
# Python 3.14 is currently incompatible with stable VTK
CANDIDATES=("python3.11" "python3.12" "python3.10" "python3.13")

PYTHON_CMD=""

echo "Searching for compatible Python version (3.10 - 3.13)..."

# Check standard paths and easy commands
for cmd in "${CANDIDATES[@]}"; do
    if command -v $cmd &> /dev/null; then
        PYTHON_CMD=$cmd
        echo "Found: $cmd"
        break
    fi
    
    # Check Homebrew paths explicitly
    if [ -f "/opt/homebrew/bin/$cmd" ]; then
        PYTHON_CMD="/opt/homebrew/bin/$cmd"
        echo "Found: $PYTHON_CMD"
        break
    fi
    
    if [ -f "/usr/local/bin/$cmd" ]; then
        PYTHON_CMD="/usr/local/bin/$cmd"
        echo "Found: $PYTHON_CMD"
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "Error: No compatible Python version found."
    echo "Please install Python 3.11 (recommended) or 3.10/3.12."
    echo "MacOS (Homebrew): brew install python@3.11"
    exit 1
fi

VENV_DIR="venv_vis"

echo "Creating virtual environment in '$VENV_DIR' using $PYTHON_CMD..."
$PYTHON_CMD -m venv $VENV_DIR

echo "Installing dependencies..."
$VENV_DIR/bin/pip install --upgrade pip
$VENV_DIR/bin/pip install pyvista matplotlib numpy imageio

echo ""
echo "=========================================================="
echo "Setup Complete!"
echo "=========================================================="
echo "To run the visualization:"
echo "  $VENV_DIR/bin/python simulations/backward_facing_step_3d/visualize_3d_isosurfaces.py --interactive"
echo ""
