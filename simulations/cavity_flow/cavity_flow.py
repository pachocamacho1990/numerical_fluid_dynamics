import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from tqdm import tqdm

def build_up_b(b, rho, dt, u, v, dx, dy):
    """
    Calculates the source term 'b' for the Pressure Poisson Equation.
    
    The equation being represented is the RHS of the Poisson equation:
                                  
    b = rho * [ (1/dt) * ( (du/dx) + (dv/dy) )
              - (du/dx)^2
              - 2 * (du/dy) * (dv/dx)
              - (dv/dy)^2 ]
              
    This term forces the velocity field to likely be divergence-free after the pressure correction.
    """
    b[1:-1, 1:-1] = (rho * (1 / dt * 
                    ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx) + 
                     (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                    ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 -
                    2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                         (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx)) -
                    ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2))
    return b

def pressure_poisson(p, dx, dy, b):
    """
    Solves the Pressure Poisson Equation using pseudo-time stepping / iterative relaxation.
    
    The discrete equation being solved is (rearranged for p[i,j]):
    
    p[i,j] = ( (p[i+1,j] + p[i-1,j]) * dy^2 + (p[i,j+1] + p[i,j-1]) * dx^2 - b[i,j] * dx^2 * dy^2 )
             -------------------------------------------------------------------------------------
                                        2 * (dx^2 + dy^2)
    """
    pn = np.empty_like(p)
    nit = 50  # Number of iterations for pressure solver approximation
    
    for q in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 + 
                          (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                          (2 * (dx**2 + dy**2)) -
                          dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * 
                          b[1:-1, 1:-1])

        # Wall Boundary Conditions for Pressure
        p[:, -1] = p[:, -2]  # dp/dx = 0 at x = 2
        p[0, :] = p[1, :]    # dp/dy = 0 at y = 0
        p[:, 0] = p[:, 1]    # dp/dx = 0 at x = 0
        p[-1, :] = 0         # p = 0 at y = 2
        
    return p

def velocity_step(u, v, p, dt, dx, dy, rho, nu):
    """
    Performs one time-step update for velocity and pressure.
    """
    un = u.copy()
    vn = v.copy()
    b = np.zeros_like(p)
    
    b = build_up_b(b, rho, dt, u, v, dx, dy)
    p = pressure_poisson(p, dx, dy, b)
    
    # Momentum Equation for u
    u[1:-1, 1:-1] = (un[1:-1, 1:-1]-
                     un[1:-1, 1:-1] * dt / dx *
                    (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                     vn[1:-1, 1:-1] * dt / dy *
                    (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                     dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                     nu * (dt / dx**2 *
                    (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                     dt / dy**2 *
                    (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))

    # Momentum Equation for v
    v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                     un[1:-1, 1:-1] * dt / dx *
                    (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                     vn[1:-1, 1:-1] * dt / dy *
                    (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                     dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                     nu * (dt / dx**2 *
                    (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                     dt / dy**2 *
                    (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

    # Boundary Conditions for Velocity
    u[0, :]  = 0
    u[:, 0]  = 0
    u[:, -1] = 0
    u[-1, :] = 1  # Lid moving at velocity 1
    
    v[0, :]  = 0
    v[-1, :] = 0
    v[:, 0]  = 0
    v[:, -1] = 0
    
    return u, v, p

def run_simulation(nt, u, v, dt, dx, dy, p, rho, nu, animate=False):
    """
    Runs the simulation loop. If animate=True, generates a GIF.
    """
    
    if not animate:
        print("Running static simulation...")
        for n in tqdm(range(nt), desc="Time Stepping"):
            u, v, p = velocity_step(u, v, p, dt, dx, dy, rho, nu)
        return u, v, p
    else:
        print("Running animation...")
        fig = plt.figure(figsize=(8, 6), dpi=100)
        ax = plt.gca()
        
        # We will skip frames to make animation faster to generate and view
        skip_frames = 5
        frames = nt // skip_frames
        
        def update(frame):
            nonlocal u, v, p
            # Run multiple steps per frame for speed
            for _ in range(skip_frames):
                 u, v, p = velocity_step(u, v, p, dt, dx, dy, rho, nu)
            
            ax.clear()
            # Contour
            contour = ax.contourf(X, Y, p, alpha=0.5, cmap='viridis')
            # Quiver
            # Stride the quiver plot to avoid clutter
            ax.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2]) 
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title(f'Cavity Flow - Step {frame*skip_frames}/{nt}')
            return ax,

        anim = animation.FuncAnimation(fig, update, frames=frames, interval=50, blit=False)
        
        # Save to the same directory as this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, "cavity_flow_evolution.gif")
        
        print(f"Saving animation to {output_path} (this may take a moment)...")
        anim.save(output_path, writer='pillow', fps=15)
        print("Animation saved.")
        return u, v, p

def check_stability(nx, ny, dt, rho, nu, dx, dy):
    """
    Checks CFL and Diffusion stability conditions.
    Returns True if stable, False if unstable.
    """
    # 1. Diffusion Limit (viscous stability)
    sigma_x = nu * dt / dx**2
    sigma_y = nu * dt / dy**2
    sigma_max = max(sigma_x, sigma_y)
    
    # 2. CFL Condition (Courant-Friedrichs-Lewy)
    # Estimate max velocity (lid velocity = 1.0)
    u_max = 1.0 
    cfl_x = u_max * dt / dx
    cfl_y = u_max * dt / dy
    
    # Memory Estimation (Fields u, v, p, b, plus temps)
    # Approx 6-8 arrays of size Nx*Ny * 8 bytes (float64) or 4 bytes (float32)
    bytes_per_float = 8 # NumPy defaults to float64
    total_elements = nx * ny
    num_arrays = 6 # u, v, p, b, un, vn
    total_memory_mb = (total_elements * bytes_per_float * num_arrays) / (1024 * 1024)
    
    print("-" * 40)
    print("RESOURCE & STABILITY CHECK")
    print("-" * 40)
    print(f"Grid: {nx}x{ny}  |  Cells: {total_elements}")
    print(f"Memory (Est): {total_memory_mb:.2f} MB")
    print(f"dt: {dt}")
    print(f"dx: {dx:.6f}, dy: {dy:.6f}")
    print("-" * 40)
    print(f"Viscous Diffusion Sigma: {sigma_max:.6f} (Limit: 0.5)")
    print(f"CFL Number: {max(cfl_x, cfl_y):.6f} (Limit: 1.0)")
    
    is_stable = True
    if sigma_max > 0.5:
        print("❌ UNSTABLE: Time step too large for viscosity (Sigma > 0.5)")
        print(f"   Recommendation: dt < {0.5 * dx**2 / nu:.6f}")
        is_stable = False
    
    if max(cfl_x, cfl_y) > 1.0:
        print("❌ UNSTABLE: Time step too large for velocity (CFL > 1.0)")
        print(f"   Recommendation: dt < {dx / u_max:.6f}")
        is_stable = False
        
    if is_stable:
        print("✅ PARAMETERS STABLE")
        
    print("-" * 40)
    return is_stable

if __name__ == "__main__":
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description="Lid-Driven Cavity Flow Simulation (NumPy)")
    parser.add_argument('--benchmark', action='store_true', help="Run in benchmark mode (no animation, print stats)")
    parser.add_argument('--check', action='store_true', help="Check stability and resources without running")
    parser.add_argument('--nx', type=int, default=41, help="Grid points in x direction")
    parser.add_argument('--ny', type=int, default=41, help="Grid points in y direction")
    parser.add_argument('--nt', type=int, default=700, help="Number of time steps")
    parser.add_argument('--dt', type=float, default=0.001, help="Time step size")
    args = parser.parse_args()

    # Parameters from args
    nx = args.nx
    ny = args.ny
    nt = args.nt
    dt = args.dt
    
    # Fixed/Derived Parameters
    nit = 50
    c = 1
    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)
    x = np.linspace(0, 2, nx)
    y = np.linspace(0, 2, ny)
    X, Y = np.meshgrid(x, y)
    
    rho = 1
    nu = .1

    # Check Mode
    if args.check:
        check_stability(nx, ny, dt, rho, nu, dx, dy)
        exit(0)

    # Run Simulation
    if args.benchmark:
        print(f"Configuration: Grid={nx}x{ny}, dt={dt}, nt={nt}, Backend=NumPy(CPU)")
        print(f"Starting simulation...")
        start_time = time.time()
        
        # Initialization
        u = np.zeros((ny, nx))
        v = np.zeros((ny, nx))
        p = np.zeros((ny, nx))
        
        run_simulation(nt, u, v, dt, dx, dy, p, rho, nu, animate=False)
        
        end_time = time.time()
        total_time = end_time - start_time
        print(f"NumPy Simulation complete.")
        print(f"Steps: {nt}")
        print(f"Time: {total_time:.4f} s")
        print(f"Speed: {nt/total_time:.2f} it/s")
        
    else:
        print(f"Configuration: Grid={nx}x{ny}, dt={dt}, nt={nt} (Animation Mode)")
        # Reset for animation
        u = np.zeros((ny, nx))
        v = np.zeros((ny, nx))
        p = np.zeros((ny, nx))
        run_simulation(nt, u, v, dt, dx, dy, p, rho, nu, animate=True)
