import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import os

def build_up_b(b, rho, dt, u, v, dx, dy):
    b[1:-1, 1:-1] = (rho * (1 / dt * 
                    ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx) + 
                     (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                    ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 -
                    2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                         (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx)) -
                    ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2))
    return b

def pressure_poisson(p, dx, dy, b):
    pn = np.empty_like(p)
    nit = 50
    for q in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 + 
                          (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                          (2 * (dx**2 + dy**2)) -
                          dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * 
                          b[1:-1, 1:-1])

        # Boundary Conditions for Pressure
        p[:, -1] = p[:, -2] # dp/dx = 0 at x = 2
        p[0, :] = p[1, :]   # dp/dy = 0 at y = 0
        p[:, 0] = p[:, 1]   # dp/dx = 0 at x = 0
        p[-1, :] = 0        # p = 0 at y = 2
        
    return p

def cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu, verbose=False):
    un = np.empty_like(u)
    vn = np.empty_like(v)
    b = np.zeros((ny, nx))
    
    # Monitoring data
    cfl_history = []
    monitoring_data = []  # For verbose CSV output

    # Progress display
    if verbose:
        print("\n" + "="*80)
        print("REAL-TIME STABILITY MONITORING")
        print("="*80)
        print(f"{'Step':<8} {'Time':<10} {'u_max':<8} {'v_max':<8} {'CFL':<8} {'D_num':<8} {'Status':<10}")
        print("-"*80)
        iter_range = range(nt)
    else:
        iter_range = tqdm(range(nt), desc="Time Stepping")

    for n in iter_range:
        un = u.copy()
        vn = v.copy()
        
        b = build_up_b(b, rho, dt, u, v, dx, dy)
        p = pressure_poisson(p, dx, dy, b)
        
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
        u[-1, :] = 1    # Lid velocity
        v[0, :]  = 0
        v[-1, :] = 0
        v[:, 0]  = 0
        v[:, -1] = 0
        
        # Calculate monitoring metrics
        u_max = np.max(np.abs(u))
        v_max = np.max(np.abs(v))
        cfl = dt * (u_max / dx + v_max / dy)
        diffusion_num = nu * dt / (dx**2)  # Should be ≤ 0.25 for stability
        
        cfl_history.append(cfl)
        
        # Verbose monitoring
        if verbose and (n % 50 == 0 or n == nt - 1):  # Print every 50 steps
            current_time = n * dt
            
            # Determine stability status
            if cfl > 1.0 or diffusion_num > 0.25:
                status = "✗ UNSTABLE"
            elif cfl > 0.8 or diffusion_num > 0.2:
                status = "⚠ WARNING"
            else:
                status = "✓ STABLE"
            
            print(f"{n:<8} {current_time:<10.4f} {u_max:<8.4f} {v_max:<8.4f} {cfl:<8.4f} {diffusion_num:<8.4f} {status:<10}")
            
            # Store monitoring data
            monitoring_data.append({
                'step': n,
                'time': current_time,
                'u_max': u_max,
                'v_max': v_max,
                'cfl': cfl,
                'diffusion_num': diffusion_num,
                'stable': cfl <= 1.0 and diffusion_num <= 0.25
            })
    
    if verbose:
        print("="*80 + "\n")
        
    return u, v, p, cfl_history, monitoring_data if verbose else None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cavity Flow Simulation (NumPy)")
    parser.add_argument("--nx", type=int, default=41, help="Grid points in X and Y")
    parser.add_argument("--nt", type=int, default=500, help="Number of time steps")
    parser.add_argument("--re", type=float, default=None, help="Reynolds number (overrides nu)")
    parser.add_argument("--dt", type=float, default=0.001, help="Time step")
    parser.add_argument("--benchmark", action="store_true", help="Minimal output for benchmarking")
    parser.add_argument("--verbose", action="store_true", help="Show real-time stability monitoring")
    parser.add_argument("--output-dir", type=str, default="outputs/baseline", help="Output directory for data files")
    args = parser.parse_args()
    
    # Create output directories
    output_dir = os.path.join("simulations/cavity_flow", args.output_dir)
    plots_dir = "simulations/cavity_flow/plots"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Parameters
    nx = args.nx
    ny = args.nx # Square grid
    nt = args.nt
    nit = 50
    c = 1
    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)
    x = np.linspace(0, 2, nx)
    y = np.linspace(0, 2, ny)
    X, Y = np.meshgrid(x, y)

    rho = 1
    dt = args.dt
    
    # Calculate nu based on Re or default
    # Default Barba Step 11: nu = 0.1 -> Re = 20
    if args.re:
        # Re = U * L / nu -> nu = U * L / Re. U=1, L=2
        nu = 2.0 / args.re
        print(f"Calculated nu={nu} for Re={args.re}")
    else:
        nu = .1

    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.zeros((ny, nx))
    b = np.zeros((ny, nx))

    if not args.benchmark:
        print(f"Starting Cavity Flow Simulation (Barba Step 11)")
        print(f"Grid: {nx}x{ny}, Re: {1*2/nu:.2f}")
        if args.verbose:
            # Pre-flight stability check
            dt_max_diffusion = dx**2 / (4 * nu)
            print(f"\nPre-flight Stability Check:")
            print(f"  Max dt (diffusion): {dt_max_diffusion:.6f}")
            print(f"  Your dt: {dt:.6f}")
            print(f"  Safety factor: {dt_max_diffusion/dt:.2f}x")
    
    result = cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu, verbose=args.verbose)
    if args.verbose:
        u, v, p, cfl_history, monitoring_data = result
    else:
        u, v, p, cfl_history = result[:4]
        monitoring_data = None
    
    # Save CFL data
    cfl_path = os.path.join(output_dir, "cfl_numpy.csv")
    np.savetxt(cfl_path, cfl_history, delimiter=",")
    if not args.benchmark:
        print(f"CFL history saved to {cfl_path}")
    
    # Save detailed monitoring data if verbose
    if args.verbose and monitoring_data:
        import csv
        monitor_path = os.path.join(output_dir, "monitoring_numpy.csv")
        with open(monitor_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['step', 'time', 'u_max', 'v_max', 'cfl', 'diffusion_num', 'stable'])
            writer.writeheader()
            writer.writerows(monitoring_data)
        print(f"Monitoring data saved to {monitor_path}")
    
    # Save Raw Solution (NPZ)
    solution_path = os.path.join(output_dir, "solution_numpy.npz")
    np.savez(solution_path, u=u, v=v, p=p)
    if not args.benchmark:
        print(f"Solution saved to {solution_path}")
    
    print("Simulation complete. Generating plot...")

    fig = plt.figure(figsize=(11, 7), dpi=100)
    # Contour plot of pressure
    plt.contourf(X, Y, p, alpha=0.5, cmap='viridis')
    plt.colorbar()
    # Streamplot to visualize vortex
    plt.streamplot(X, Y, u, v, color='k')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Cavity Flow Pressure and Streamlines')
    
    output_file = os.path.join(plots_dir, "cavity_flow_result.png")
    plt.savefig(output_file)
    print(f"Result saved to {output_file}")
