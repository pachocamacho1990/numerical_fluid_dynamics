
import jax.numpy as jnp
from jax import jit

@jit
def compute_wall_distance_3d(mask, dx, dy, dz):
    """
    Compute approximate wall distance field d(x,y,z).
    
    For complex geometries, this usually requires solving the Eikonal equation
    or brute-force search.
    For the backward-facing step or channel, we can compute it analytically or 
    assume simple boundaries.
    
    For now, we implement a simplified analytical distance for channel-like bounds.
    If 'mask' is provided (1=fluid, 0=solid), we can estimate distance.
    
    Args:
        mask: Boolean/Binary mask (nz, ny, nx) where 1 is fluid.
              Or can be specialized for specific geometry in future.
        dx, dy, dz: Grid spacing
        
    Returns:
        d: Distance field (nz, ny, nx)
    """
    # Placeholder: Euclidean distance to y=0 and y=Ly (top/bottom walls)
    # Assumes simple channel geometry for now.
    
    nz, ny, nx = mask.shape
    
    # Grid coordinates
    y = jnp.linspace(0.0, (ny-1)*dy, ny)
    z = jnp.linspace(0.0, (nz-1)*dz, nz)
    
    # Broadcast to 3D
    Y = jnp.broadcast_to(y[None, :, None], (nz, ny, nx))
    Z = jnp.broadcast_to(z[:, None, None], (nz, ny, nx))
    
    # Distance to bottom wall (y=0) and top wall (y=Ly)
    Ly = (ny - 1) * dy
    dist_y = jnp.minimum(Y, Ly - Y)
    # Also z walls if spanwise constrained? (Periodic usually in z for channel)
    # For now assume walls at y=0, y=Ly.
    
    return dist_y

@jit
def van_driest_damping(nu_t_raw, y_plus, A_plus=25.0):
    """
    Apply Van Driest damping to eddy viscosity.
    
    D = (1 - exp(-y+/A+))^2
    nu_t = nu_t_raw * D
    
    Args:
        nu_t_raw: Undamped eddy viscosity
        y_plus: Wall units y*u_tau/nu
        A_plus: Van Driest constant (default 25)
        
    Returns:
        nu_t_damped
    """
    damping = (1.0 - jnp.exp(-y_plus / A_plus))**2
    return nu_t_raw * damping

@jit
def compute_friction_velocity(u, nu, dy):
    """
    Estimate friction velocity u_tau from simplified wall gradients.
    u_tau = sqrt(tau_w / rho)
    tau_w = mu * du/dy|_wall ~= rho * nu * u_first / dy
    
    Args:
        u: Velocity component parallel to wall (nz, ny, nx)
        nu: Kinematic viscosity
        dy: Grid spacing normal to wall
    
    Returns:
        u_tau field (approximate)
    """
    # Assuming first grid point is at y=dy (since y=0 is wall with u=0)
    # tau_w = nu * (u[1] - u[0]) / dy = nu * u[1] / dy
    
    # This is rough and assumes known wall location.
    # We operate on the whole field or just extract near wall?
    # Usually we want a field of u_tau that is prolonged into the domain 
    # or computed locally based on nearest wall.
    
    # Simplified approach: Use local velocity gradient at the wall,
    # and extend this u_tau to the whole boundary layer for y+ calculation.
    
    # For channel flow validation (walls at y=0, y=H):
    # u_tau_bottom = sqrt(nu * u[:, 1, :] / dy)
    # u_tau_top = sqrt(nu * u[:, -2, :] / dy)
    
    # We need a field of u_tau to compute y+ everywhere.
    # y+ = d * u_tau / nu.
    
    # Let's return a field of u_tau that varies in x/z but is constant in y (?)
    # or just return the separate values.
    # To keep it general and simple for now:
    
    u_grad_bot = jnp.abs(u[:, 1, :]) / dy
    u_tau_bot = jnp.sqrt(nu * u_grad_bot)
    
    u_grad_top = jnp.abs(u[:, -2, :]) / dy
    u_tau_top = jnp.sqrt(nu * u_grad_top)
    
    # Broadcast to interior? 
    # Simple linear interpolation or just use nearest.
    # For channel flow, correct u_tau is critical. 
    # Let's assume bottom half uses bottom u_tau, top half uses top.
    
    return u_tau_bot, u_tau_top
