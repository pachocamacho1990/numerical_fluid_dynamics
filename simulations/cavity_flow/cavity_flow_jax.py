import jax
import jax.numpy as jnp
import numpy as np
import time
import os
import matplotlib.pyplot as plt

# Enable Metal backend (if available) or fallback to CPU
try:
    jax.config.update("jax_enable_x64", False) # Metal prefers float32
    print(f"JAX Default Backend: {jax.default_backend()}")
    print(f"JAX Devices: {jax.devices()}")
except Exception as e:
    print(f"Warning: Could not configure JAX backend/devices: {e}")

def build_up_b(b, rho, dt, u, v, dx, dy):
    """
    Calculates source term 'b' using JAX arrays.
    """
    # Note: Intermediate terms are calculated for the interior points [1:-1, 1:-1]
    
    # Slice views for readability and performance
    u_center = u[1:-1, 1:-1]
    u_right  = u[1:-1, 2:]
    u_left   = u[1:-1, 0:-2]
    u_up     = u[2:, 1:-1]
    u_down   = u[0:-2, 1:-1]
    
    v_center = v[1:-1, 1:-1]
    v_right  = v[1:-1, 2:]
    v_left   = v[1:-1, 0:-2]
    v_up     = v[2:, 1:-1]
    v_down   = v[0:-2, 1:-1]

    # Term 1: Div(u)/dt
    term1 = ((u_right - u_left) / (2 * dx) + 
             (v_up - v_down) / (2 * dy)) / dt
            
    # Term 2: (du/dx)^2
    term2 = ((u_right - u_left) / (2 * dx))**2
    
    # Term 3: 2 * (du/dy) * (dv/dx)
    term3 = 2 * ((u_up - u_down) / (2 * dy) * 
                 (v_right - v_left) / (2 * dx))
                 
    # Term 4: (dv/dy)^2
    term4 = ((v_up - v_down) / (2 * dy))**2

    b_interior = rho * (term1 - term2 - term3 - term4)
    
    # Update interior of b
    return b.at[1:-1, 1:-1].set(b_interior)

@jax.jit
def poisson_step_jit(p, b, dx, dy):
    """
    Single step of Pressure Poisson solver.
    Unrolled loops or scan are better for JAX, but for simple porting we JIT the whole step.
    Actually, we can put the loop inside the JIT function using jax.lax.fori_loop for max speed.
    """
    # For simplicity and speed comparison, let's keep the iterative structure but 
    # write the update as a single tensor operation which JAX loves.
    
    p_right = p[1:-1, 2:]
    p_left  = p[1:-1, 0:-2]
    p_up    = p[2:, 1:-1]
    p_down  = p[0:-2, 1:-1]
    
    p_new_interior = (((p_right + p_left) * dy**2 + 
                       (p_up + p_down) * dx**2) /
                      (2 * (dx**2 + dy**2)) -
                      dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * 
                      b[1:-1, 1:-1])
    
    p = p.at[1:-1, 1:-1].set(p_new_interior)
    
    # Boundary Conditions
    p = p.at[:, -1].set(p[:, -2])  # x = 2
    p = p.at[0, :].set(p[1, :])    # y = 0
    p = p.at[:, 0].set(p[:, 1])    # x = 0
    p = p.at[-1, :].set(0)         # y = 2
    
    return p

def pressure_poisson(p, dx, dy, b, nit=50):
    """
    Solves PPE. We'll JIT the loop body or use lax.scan.
    Using python loop inside JIT unrolls it, which is fine for nit=50.
    """
    for _ in range(nit):
        p = poisson_step_jit(p, b, dx, dy)
    return p

@jax.jit
def velocity_step_jit(u, v, p, dt, dx, dy, rho, nu):
    """
    JIT-compiled single time step for the whole system.
    Returns updated u, v, p.
    """
    
    # 1. Calculate B
    b = jnp.zeros_like(p)
    b = build_up_b(b, rho, dt, u, v, dx, dy)
    
    # 2. Solve Pressure (iterative)
    # We embed the loop here appropriately. 
    # Since 'pressure_poisson' calls 'poisson_step_jit', and we are inside a JIT function,
    # the whole thing gets fused.
    
    # Note: We must convert the python loop in pressure_poisson to something JIT-friendly or unroll.
    # A simple python loop unrolls with static nit.
    
    nit = 50
    # Function to run one pressure step
    def body_fun(i, p_val):
        return poisson_step_jit(p_val, b, dx, dy)
    
    # Use jax.lax.fori_loop for efficient compiling of fixed loops without unrolling overhead
    p = jax.lax.fori_loop(0, nit, body_fun, p)

    # 3. Update Velocity
    un = u
    vn = v
    
    # Stencils
    u_center, v_center = un[1:-1, 1:-1], vn[1:-1, 1:-1]
    
    u_right, u_left = un[1:-1, 2:], un[1:-1, 0:-2]
    u_up, u_down    = un[2:, 1:-1], un[0:-2, 1:-1]
    
    v_right, v_left = vn[1:-1, 2:], vn[1:-1, 0:-2]
    v_up, v_down    = vn[2:, 1:-1], vn[0:-2, 1:-1]
    
    p_right, p_left = p[1:-1, 2:], p[1:-1, 0:-2]
    p_up, p_down    = p[2:, 1:-1], p[0:-2, 1:-1] # For v-momentum term (p_up - p_down) which is (p[i,j+1]-p[i,j-1]) in y-dir? 
    # Wait, original code:
    # u-mom: p[1:-1, 2:] - p[1:-1, 0:-2] (dx)
    # v-mom: p[2:, 1:-1] - p[0:-2, 1:-1] (dy) -> This corresponds to y-partial
    
    # Explicit updates
    u_new = (u_center -
             u_center * dt / dx * (u_center - u_left) -
             v_center * dt / dy * (u_center - u_down) - # Note: original code v term uses u_down? 
             # Original u-mom: - v[1:-1, 1:-1] * dt / dy * (u[1:-1, 1:-1] - u[0:-2, 1:-1])  -> Yes, u[0:-2] is (i-1 in y?) standard notation usually (i,j-1). In numpy 0:-2 is 'down' (lower index).
             dt / (2 * rho * dx) * (p_right - p_left) +
             nu * (dt / dx**2 * (u_right - 2 * u_center + u_left) +
                   dt / dy**2 * (u_up - 2 * u_center + u_down)))
                   
    v_new = (v_center -
             u_center * dt / dx * (v_center - v_left) -
             v_center * dt / dy * (v_center - v_down) -
             dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
             nu * (dt / dx**2 * (v_right - 2 * v_center + v_left) +
                   dt / dy**2 * (v_up - 2 * v_center + v_down)))

    u = u.at[1:-1, 1:-1].set(u_new)
    v = v.at[1:-1, 1:-1].set(v_new)
    
    # Boundary Conditions
    u = u.at[0, :].set(0)
    u = u.at[:, 0].set(0)
    u = u.at[:, -1].set(0)
    u = u.at[-1, :].set(1)
    
    v = v.at[0, :].set(0)
    v = v.at[-1, :].set(0)
    v = v.at[:, 0].set(0)
    v = v.at[:, -1].set(0)
    
    return u, v, p


def run_simulation_jax(nt=700, nx=41, ny=41, dt=0.001):
    # Parameters
    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)
    rho = 1.0
    nu = 0.1

    # Initialization (JAX arrays)
    u = jnp.zeros((ny, nx), dtype=jnp.float32)
    v = jnp.zeros((ny, nx), dtype=jnp.float32)
    p = jnp.zeros((ny, nx), dtype=jnp.float32)
    
    print(f"Starting JAX simulation with {nt} steps...")
    print(f"Configuration: Grid={nx}x{ny}, dt={dt}")
    
    # Warmup / Compilation
    print("Compiling JIT function (warmup)...")
    start_compile = time.time()
    u, v, p = velocity_step_jit(u, v, p, dt, dx, dy, rho, nu)
    u.block_until_ready() # Force execution
    print(f"Compilation finished in {time.time() - start_compile:.4f}s")
    
    # Benchmark
    start = time.time()
    # Using lax.scan is the most 'JAX' way to loop, but python loop is easier to read/debug for comparison
    # We will use a python loop calling the JIT function to mimic the numpy structure
    
    # Optional: We could scan the whole NT loop for ultimate speed, but this is fast enough to show diff
    
    # Let's try scan for maximum "Apple Silicon" flex
    def scan_body(carry, _):
        u, v, p = carry
        u, v, p = velocity_step_jit(u, v, p, dt, dx, dy, rho, nu)
        return (u, v, p), None

    (u, v, p), _ = jax.lax.scan(scan_body, (u, v, p), jnp.arange(nt))
    
    # For fair timing vs Python loop, we block until ready
    jax.block_until_ready(u)
    
    end = time.time()
    total_time = end - start
    print(f"JAX Simulation complete.")
    print(f"Steps: {nt}")
    print(f"Time: {total_time:.4f} s")
    print(f"Speed: {nt/total_time:.2f} it/s")
    
    return u, v, p

def check_stability(nx, ny, dt, rho, nu, dx, dy):
    """
    Checks CFL and Diffusion stability conditions.
    """
    # 1. Diffusion Limit
    sigma_x = nu * dt / dx**2
    sigma_y = nu * dt / dy**2
    sigma_max = max(sigma_x, sigma_y)
    
    # 2. CFL Condition
    u_max = 1.0 
    cfl_x = u_max * dt / dx
    cfl_y = u_max * dt / dy
    
    # Memory Estimation (JAX typically uses float32)
    bytes_per_float = 4 
    total_elements = nx * ny
    num_arrays = 3 # u, v, p (plus internal copies)
    total_memory_mb = (total_elements * bytes_per_float * num_arrays * 2) / (1024 * 1024) # *2 safety factor
    
    print("-" * 40)
    print("RESOURCE & STABILITY CHECK (JAX)")
    print("-" * 40)
    print(f"Grid: {nx}x{ny}  |  dt: {dt}")
    print(f"Memory (Est): {total_memory_mb:.2f} MB")
    print("-" * 40)
    print(f"Viscous Diffusion Sigma: {sigma_max:.6f} (Limit: 0.5)")
    print(f"CFL Number: {max(cfl_x, cfl_y):.6f} (Limit: 1.0)")
    
    is_stable = True
    if sigma_max > 0.5:
        print("❌ UNSTABLE: dt too large (Sigma > 0.5)")
        print(f"   Recommendation: dt < {0.5 * dx**2 / nu:.6f}")
        is_stable = False
    elif max(cfl_x, cfl_y) > 1.0:
        print("❌ UNSTABLE: dt too large (CFL > 1.0)")
        is_stable = False
    else:
        print("✅ PARAMETERS STABLE")
    print("-" * 40)
    return is_stable

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Lid-Driven Cavity Flow Simulation (JAX)")
    parser.add_argument('--check', action='store_true', help="Check stability and resources without running")
    parser.add_argument('--nx', type=int, default=41, help="Grid points in x direction")
    parser.add_argument('--ny', type=int, default=41, help="Grid points in y direction")
    parser.add_argument('--nt', type=int, default=700, help="Number of time steps")
    parser.add_argument('--dt', type=float, default=0.001, help="Time step size")
    args = parser.parse_args()

    if args.check:
        dx = 2 / (args.nx - 1)
        dy = 2 / (args.ny - 1)
        check_stability(args.nx, args.ny, args.dt, 1.0, 0.1, dx, dy)
        exit(0)

    u, v, p = run_simulation_jax(nt=args.nt, nx=args.nx, ny=args.ny, dt=args.dt)
    
    # Save a simple plot to verify correctness
    u = np.array(u) # Convert back to CPU numpy for plotting
    p = np.array(p)
    v = np.array(v)
    
    nx = args.nx
    ny = args.ny
    x = np.linspace(0, 2, nx)
    y = np.linspace(0, 2, ny)
    X, Y = np.meshgrid(x, y)
    
    fig = plt.figure(figsize=(11, 7), dpi=100)
    plt.contourf(X, Y, p, alpha=0.5, cmap='viridis')
    plt.colorbar()
    plt.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2]) 
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "cavity_flow_jax_result.png")
    plt.savefig(output_path)
    print(f"Result saved to {output_path}")
