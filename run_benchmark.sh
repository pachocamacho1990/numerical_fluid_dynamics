#!/bin/bash

echo "========================================================="
echo "       NUMERICAL FLUID DYNAMICS BENCHMARK SUITE          "
echo "========================================================="
echo ""

# 1. NumPy Baseline (using venv)
echo "[1/3] Running NumPy Baseline (Standard CPU)..."
source venv/bin/activate
python simulations/cavity_flow/cavity_flow.py --benchmark "$@"
deactivate
echo ""

# 2. JAX CPU JIT (using venv)
echo "[2/3] Running JAX CPU (JIT Compilation)..."
source venv/bin/activate
# Force CPU for this run in case environment has Metal
# Note: In current setup venv only has CPU JAX anyway.
export JAX_PLATFORMS=cpu
python simulations/cavity_flow/cavity_flow_jax.py --benchmark "$@"
deactivate
echo ""

# 3. JAX GPU Metal (using venv_metal)
echo "[3/3] Running JAX Metal (Apple Silicon GPU)..."
unset JAX_PLATFORMS
if [ -d "venv_metal" ]; then
    source venv_metal/bin/activate
    # Ensure Metal is enabled (default in this venv, but explicit is good)
    # We rely on the script's internal config block which enables Metal if available
    python simulations/cavity_flow/cavity_flow_jax.py --benchmark "$@"
    deactivate
else
    echo "Skipping Metal Benchmark: venv_metal not found."
fi
echo ""

# 4. Compare Solutions
echo "[4/4] Generating Combined Solution Plot..."
source venv/bin/activate
python simulations/cavity_flow/compare_solutions.py
deactivate
echo ""

echo "========================================================="
echo "                  BENCHMARK COMPLETE                     "
echo "========================================================="
