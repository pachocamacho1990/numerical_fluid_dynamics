import numpy as np
import matplotlib.pyplot as plt
import os

def load_data(filename):
    if os.path.exists(filename):
        return np.loadtxt(filename, delimiter=",")
    return None

def main():
    base_dir = "simulations/cavity_flow"
    
    # Load data
    numpy_cfl = load_data(os.path.join(base_dir, "cfl_numpy.csv"))
    jax_cpu_cfl = load_data(os.path.join(base_dir, "cfl_jax_cpu.csv"))
    jax_metal_cfl = load_data(os.path.join(base_dir, "cfl_jax_metal.csv"))
    
    plt.figure(figsize=(10, 6), dpi=100)
    
    if numpy_cfl is not None:
        plt.plot(numpy_cfl, label="NumPy", linewidth=2, alpha=0.8)
    
    if jax_cpu_cfl is not None:
        plt.plot(jax_cpu_cfl, label="JAX (CPU)", linestyle="--", linewidth=2, alpha=0.8)
        
    if jax_metal_cfl is not None:
        plt.plot(jax_metal_cfl, label="JAX (Metal)", linestyle=":", linewidth=2, alpha=0.8)
        
    plt.xlabel("Time Step")
    plt.ylabel("CFL Number")
    plt.title("CFL Number Evolution per Time Step")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_file = os.path.join(base_dir, "cfl_comparison.png")
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    main()
