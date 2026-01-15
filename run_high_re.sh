#!/bin/bash

# Configuration
RE=1000
NX=201
NT=5000
DT=0.0001 # Critical for stability at this Re/Grid

echo "========================================================="
echo "       HIGH REYNOLDS ($RE) BENCHMARK SUITE ($NX x $NX)   "
echo "========================================================="
echo ""

# 1. NumPy (Baseline)
echo "[1/3] Running NumPy (Refined Grid)..."
source venv/bin/activate
python simulations/cavity_flow/cavity_flow.py --re $RE --nx $NX --nt $NT --dt $DT --benchmark
deactivate

# 2. JAX CPU
echo "[2/3] Running JAX CPU..."
source venv/bin/activate
export JAX_PLATFORMS=cpu
python simulations/cavity_flow/cavity_flow_jax.py --re $RE --nx $NX --nt $NT --dt $DT --benchmark
deactivate

# 3. JAX Metal
echo "[3/3] Running JAX Metal..."
if [ -d "venv_metal" ]; then
    source venv_metal/bin/activate
    unset JAX_PLATFORMS # Let it auto-detect metal
    python simulations/cavity_flow/cavity_flow_jax.py --re $RE --nx $NX --nt $NT --dt $DT --benchmark
    deactivate
else
    echo "Skipping Metal (venv_metal not found)"
fi

# 4. Compare
echo "[4/4] Generating High-Re Comparison Plot..."
source venv/bin/activate
python simulations/cavity_flow/compare_solutions.py --output solutions_comparison_re1000.png
deactivate

echo ""
echo "Done. Result saved to simulations/cavity_flow/solutions_comparison_re1000.png"
