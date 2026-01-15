import jax
import jax.numpy as jnp
from jax import jit, lax
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import argparse

# --- JIT Compiled Functions ---

@jit
def build_up_b(rho, dt, u, v, dx, dy):
    b = jnp.zeros_like(u)
    
    # Slices for vectorization
    u_i_plus_1 = u[1:-1, 2:]
    u_i_minus_1 = u[1:-1, 0:-2]
    u_i = u[1:-1, 1:-1]
    
    v_j_plus_1 = v[2:, 1:-1]
    v_j_minus_1 = v[0:-2, 1:-1]
    v_j = v[1:-1, 1:-1]
    
    # Cross terms need slightly different indices to match "u[2:, 1:-1]" etc relative to the centered grid
    # But strictly following the numpy implementation logic:
    
    # Term 1: (u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx)
    du_dx = (u_i_plus_1 - u_i_minus_1) / (2 * dx)
    
    # Term 2: (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)
    dv_dy = (v_j_plus_1 - v_j_minus_1) / (2 * dy)
    
    # Term squares and cross
    # u_y: (u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy)
    du_dy = (u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy)
    
    # v_x: (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx)
    dv_dx = (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx)
    
    
    b = b.at[1:-1, 1:-1].set(
        rho * (1 / dt * (du_dx + dv_dy) -
               du_dx**2 -
               2 * (du_dy * dv_dx) -
               dv_dy**2)
    )
    return b

def pressure_poisson_step(i, p, dx, dy, b):
    # This function defines a single iteration of the Poisson solver
    # suitable for use with lax.fori_loop
    
    pn = p
    
    # Calculate new pressure at inner points
    new_p_inner = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 + 
                    (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                   (2 * (dx**2 + dy**2)) -
                   dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * 
                   b[1:-1, 1:-1])
    
    p = p.at[1:-1, 1:-1].set(new_p_inner)
    
    # Boundary Conditions
    p = p.at[:, -1].set(p[:, -2]) # dp/dx = 0 at x = 2
    p = p.at[0, :].set(p[1, :])   # dp/dy = 0 at y = 0
    p = p.at[:, 0].set(p[:, 1])   # dp/dx = 0 at x = 0
    p = p.at[-1, :].set(0)        # p = 0 at y = 2 (Lid)
    
    return p

@jit
def solve_pressure(p, dx, dy, b, nit):
    # Use lax.fori_loop to keep the iterations inside the compiled XLA kernel
    # body_fun takes (i, val) -> val
    body_fun = lambda i, p_val: pressure_poisson_step(i, p_val, dx, dy, b)
    p_final = lax.fori_loop(0, nit, body_fun, p)
    return p_final

@jit
def time_step(u, v, p, dt, dx, dy, rho, nu, nit):
    # 1. Calculate B
    b = build_up_b(rho, dt, u, v, dx, dy)
    
    # 2. Solve Pressure Poisson
    p = solve_pressure(p, dx, dy, b, nit)
    
    # 3. Update Velocities
    un = u
    vn = v
    
    # Velocity update terms
    u_inner = (un[1:-1, 1:-1] -
               un[1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
               vn[1:-1, 1:-1] * dt / dy * (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
               dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
               nu * (dt / dx**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                     dt / dy**2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))

    v_inner = (vn[1:-1, 1:-1] -
               un[1:-1, 1:-1] * dt / dx * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
               vn[1:-1, 1:-1] * dt / dy * (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
               dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
               nu * (dt / dx**2 * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                     dt / dy**2 * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

    u = u.at[1:-1, 1:-1].set(u_inner)
    v = v.at[1:-1, 1:-1].set(v_inner)
    
    # 4. Boundary Conditions
    u = u.at[0, :].set(0)
    u = u.at[:, 0].set(0)
    u = u.at[:, -1].set(0)
    u = u.at[-1, :].set(1) # Lid velocity
    
    v = v.at[0, :].set(0)
    v = v.at[-1, :].set(0)
    v = v.at[:, 0].set(0)
    v = v.at[:, -1].set(0)
    
    v = v.at[:, -1].set(0)
    
    # Calculate CFL (Max velocity)
    cfl = dt * (jnp.max(jnp.abs(u))/dx + jnp.max(jnp.abs(v))/dy)
    
    return u, v, p, cfl

def main():
    parser = argparse.ArgumentParser(description="JAX Cavity Flow Simulation")
    parser.add_argument("--benchmark", action="store_true", help="Run in benchmark mode (minimal output)")
    parser.add_argument("--nx", type=int, default=41, help="Grid points in X and Y")
    parser.add_argument("--nt", type=int, default=500, help="Number of time steps")
    parser.add_argument("--re", type=float, default=None, help="Reynolds number (overrides nu)")
    parser.add_argument("--dt", type=float, default=0.001, help="Time step")
    args = parser.parse_args()

    # Parameters
    nx = args.nx
    ny = args.nx
    nt = args.nt
    nit = 50
    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)
    
    rho = 1.0
    dt = args.dt
    
    if args.re:
        nu = 2.0 / args.re
        print(f"Calculated nu={nu} for Re={args.re}")
    else:
        nu = 0.1

    # Initialization (JAX arrays)
    u = jnp.zeros((ny, nx))
    v = jnp.zeros((ny, nx))
    p = jnp.zeros((ny, nx))
    
    backend = jax.default_backend()
    print(f"JAX Backend: {backend}")
    print(f"Starting Cavity Flow (JAX) - Grid: {nx}x{ny}, Steps: {nt}")

    # Warmup / Compilation
    print("Compiling JIT functions...")
    start_warmup = time.time()
    _, _, _, _ = time_step(u, v, p, dt, dx, dy, rho, nu, nit)
    u = u.block_until_ready() # Force execution
    end_warmup = time.time()
    print(f"Compilation finished in {end_warmup - start_warmup:.4f}s")

    # Main Loop
    start_time = time.time()
    cfl_history = []
    
    # We can iterate in Python, because the heavy lifting (nit=50 loops) is done inside the JIT function
    iter_range = range(nt) if args.benchmark else tqdm(range(nt), desc="JAX Time Stepping")
    
    for n in iter_range:
        u, v, p, cfl = time_step(u, v, p, dt, dx, dy, rho, nu, nit)
        cfl_history.append(cfl)
        # We don't block_until_ready inside the loop to allow async dispatch, 
        # unless profiling explicitly.
    
    u = u.block_until_ready() # Sync at the end
    end_time = time.time()
    
    # Convert CFL history to numpy and save
    cfl_history = np.array(cfl_history)
    output_csv = f"simulations/cavity_flow/cfl_jax_{backend}.csv"
    output_npz = f"simulations/cavity_flow/solution_jax_{backend}.npz"
    
    if backend == "METAL": 
        output_csv = "simulations/cavity_flow/cfl_jax_metal.csv"
        output_npz = "simulations/cavity_flow/solution_jax_metal.npz"
    elif backend == "cpu":
        output_csv = "simulations/cavity_flow/cfl_jax_cpu.csv"
        output_npz = "simulations/cavity_flow/solution_jax_cpu.npz"
        
    np.savetxt(output_csv, cfl_history, delimiter=",")
    print(f"CFL history saved to {output_csv}")
    
    # Save Raw Solution (NPZ) - Convert to numpy first
    np.savez(output_npz, u=np.array(u), v=np.array(v), p=np.array(p))
    print(f"Solution saved to {output_npz}")
    
    duration = end_time - start_time
    print(f"Simulation complete. Duration: {duration:.4f}s ({nt/duration:.2f} steps/s)")

    if not args.benchmark:
        # Visualization (Convert back to Numpy)
        print("Generating plot...")
        u_np = np.array(u)
        v_np = np.array(v)
        p_np = np.array(p)
        
        x = np.linspace(0, 2, nx)
        y = np.linspace(0, 2, ny)
        X, Y = np.meshgrid(x, y)
        
        fig = plt.figure(figsize=(11, 7), dpi=100)
        plt.contourf(X, Y, p_np, alpha=0.5, cmap='viridis')
        plt.colorbar()
        plt.streamplot(X, Y, u_np, v_np, color='k')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Cavity Flow (JAX {backend})')
        
        output_file = "simulations/cavity_flow/cavity_flow_jax_result.png"
        plt.savefig(output_file)
        print(f"Result saved to {output_file}")

if __name__ == "__main__":
    main()
