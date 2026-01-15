import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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

def cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu):
    un = np.empty_like(u)
    vn = np.empty_like(v)
    b = np.zeros((ny, nx))
    
    # CFL History
    cfl_history = []

    for n in tqdm(range(nt), desc="Time Stepping"):
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
        
        # Calculate CFL
        cfl = dt * (np.max(np.abs(u)) / dx + np.max(np.abs(v)) / dy)
        cfl_history.append(cfl)
        
    return u, v, p, cfl_history

if __name__ == "__main__":
    # Parameters
    nx = 41
    ny = 41
    nt = 500
    nit = 50
    c = 1
    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)
    x = np.linspace(0, 2, nx)
    y = np.linspace(0, 2, ny)
    X, Y = np.meshgrid(x, y)

    rho = 1
    nu = .1
    dt = .001

    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.zeros((ny, nx))
    b = np.zeros((ny, nx))

    print(f"Starting Cavity Flow Simulation (Barba Step 11)")
    print(f"Grid: {nx}x{ny}, Re: {1*2/nu} (approx based on lid vel=1, L=2)")
    
    u, v, p, cfl_history = cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu)
    
    # Save CFL data
    np.savetxt("simulations/cavity_flow/cfl_numpy.csv", cfl_history, delimiter=",")
    print("CFL history saved to simulations/cavity_flow/cfl_numpy.csv")
    
    # Save Raw Solution (NPZ)
    np.savez("simulations/cavity_flow/solution_numpy.npz", u=u, v=v, p=p)
    print("Solution saved to simulations/cavity_flow/solution_numpy.npz")
    
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
    
    output_file = "simulations/cavity_flow/cavity_flow_result.png"
    plt.savefig(output_file)
    print(f"Result saved to {output_file}")
