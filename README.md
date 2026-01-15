# Numerical Fluid Dynamics

A repository for learning and implementing numerical solutions to fluid flow problems, with a focus on performance comparison between NumPy and JAX (CPU/Metal) implementations.

## Project Overview

This project implements classical computational fluid dynamics (CFD) problems using the Finite Difference Method. Each simulation includes:
- **NumPy baseline** implementation
- **JAX-accelerated** versions (CPU and Apple Metal GPU)
- **Comprehensive benchmarking** and performance analysis
- **Detailed mathematical documentation**

## Current Simulations

### Lid-Driven Cavity Flow
2D incompressible Navier-Stokes simulation following Lorena Barba's "12 Steps to Navier-Stokes" (Step 11).

- **Location**: [`simulations/cavity_flow/`](simulations/cavity_flow/)
- **Documentation**: [Theory](docs/cavity_flow_theory.md) | [Usage](simulations/cavity_flow/README.md)
- **Features**:
  - Multiple Reynolds number configurations
  - Stability analysis (Re up to 7500)
  - CFL number tracking
  - Performance benchmarking

## Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/numerical_fluid_dynamics.git
cd numerical_fluid_dynamics

# Install dependencies in virtual environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
deactivate

# Optional: Install JAX with Metal support (Apple Silicon only)
python -m venv venv_metal
source venv_metal/bin/activate
pip install -r requirements.txt
pip install jax-metal
deactivate
```

### Using Virtual Environments

**For NumPy or JAX CPU simulations:**
```bash
source venv/bin/activate
python simulations/cavity_flow/cavity_flow.py
deactivate
```

**For JAX Metal (GPU) simulations:**
```bash
source venv_metal/bin/activate
python simulations/cavity_flow/cavity_flow_jax.py
deactivate
```

> [!TIP]
> The benchmark scripts (`run_benchmark.sh`, `run_high_re.sh`) automatically handle venv activation/deactivation.

### Run a Simulation
```bash
# Basic cavity flow (NumPy)
python simulations/cavity_flow/cavity_flow.py

# High Reynolds number (Re=1000, 201x201 grid)
python simulations/cavity_flow/cavity_flow.py --re 1000 --nx 201 --nt 5000 --dt 0.0001

# JAX version (auto-detects backend)
python simulations/cavity_flow/cavity_flow_jax.py
```

### Run Benchmarks
```bash
# Compare NumPy, JAX CPU, and JAX Metal
./run_benchmark.sh

# High-Re benchmark (Re=1000)
./run_high_re.sh
```

## Repository Structure

```
numerical_fluid_dynamics/
├── simulations/
│   └── cavity_flow/
│       ├── outputs/          # Simulation data (gitignored)
│       │   ├── baseline/     # Default benchmark data
│       │   └── high_re/      # High Reynolds data
│       ├── plots/            # Visualizations (tracked)
│       ├── cavity_flow.py    # NumPy implementation
│       ├── cavity_flow_jax.py # JAX implementation
│       └── README.md
├── docs/                     # Mathematical theory
│   └── cavity_flow_theory.md
├── run_benchmark.sh          # Baseline benchmark suite
├── run_high_re.sh            # High-Re benchmark
└── requirements.txt
```

## Performance

On Apple M1 (41×41 grid, 500 steps):
- **NumPy**: ~0.4s
- **JAX (CPU)**: ~0.07s (**5.7x faster**)
- **JAX (Metal)**: ~1.2s (overhead dominates for small grids)

On larger grids (201×201, 5000 steps, Re=1000):
- **NumPy**: ~39s
- **JAX (CPU)**: ~11s (**3.5x faster**)
- **JAX (Metal)**: ~22s

## Documentation

- **Theory**: [`docs/`](docs/) - Mathematical derivations and discretization schemes
- **Usage**: Each simulation folder contains a detailed README
- **Agent Instructions**: [`AGENT_INSTRUCTIONS.md`](AGENT_INSTRUCTIONS.md) - For AI agents working on this repo

## Contributing

When adding new simulations:
1. Create a folder in `simulations/`
2. Add mathematical documentation to `docs/`
3. Include visualization outputs in `plots/`
4. Follow the structure of existing simulations

## License

MIT License - see LICENSE file for details
