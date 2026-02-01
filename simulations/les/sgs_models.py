import jax
import jax.numpy as jnp
from jax import jit

@jit
def compute_strain_rate_tensor_3d(u, v, w, dx, dy, dz):
    """
    Compute the 6 independent components of the strain rate tensor S_ij.
    
    S_ij = 0.5 * (dv_i/dx_j + dv_j/dx_i)
    
    Args:
        u, v, w: Velocity fields of shape (nz, ny, nx)
        dx, dy, dz: Grid spacings (scalers)
        
    Returns:
        Tuple of (S11, S22, S33, S12, S13, S23) each of shape (nz, ny, nx)
        where indices 1,2,3 correspond to x,y,z
    """
    
    # 1. Diagonal components (normal strains)
    # S11 = du/dx
    S11 = (u[..., 2:] - u[..., :-2]) / (2 * dx)
    # Pad to maintain shape, or rely on internal points only.
    # To keep shapes consistent with input, we need to handle boundaries.
    # For LES, usually we care about internal points for SGS. 
    # Let's pad with zeros or replicate boundary values to return same shape.
    # A simple approach for separate gradients is to pad the result.
    # However, standard practice in this codebase (from context) suggests 
    # operations often shrink or we need to be careful. 
    # Let's use jnp.gradient-like behavior but explicitly with central diffs 
    # where possible, and one-sided at boundaries, OR simply return the 
    # valid internal domain if that's the convention.
    # Looking at the plan, it expects (nz, ny, nx). 
    # We will use jnp.gradient for simplicity and correctness at boundaries if uniform grid.
    # But strictly following the plan's snippet which used slicing:
    # S11 = (u[..., 2:] - u[..., :-2]) / (2 * dx)  <-- this reduces size by 2 in x
    
    # Let's implement full shape return using jnp.gradient for now as it handles boundaries,
    # or implement explicit central diffs with padding.
    # Explicit central diffs are faster/more controlled in JAX often.
    
    def central_diff_x(f, d):
        df = (f[..., 2:] - f[..., :-2]) / (2 * d)
        # Pad left and right to restore shape
        return jnp.pad(df, ((0,0), (0,0), (1,1)), mode='edge')

    def central_diff_y(f, d):
        df = (f[:, 2:, :] - f[:, :-2, :]) / (2 * d)
        return jnp.pad(df, ((0,0), (1,1), (0,0)), mode='edge')
    
    def central_diff_z(f, d):
        df = (f[2:, ...] - f[:-2, ...]) / (2 * d)
        return jnp.pad(df, ((1,1), (0,0), (0,0)), mode='edge')

    S11 = central_diff_x(u, dx)
    S22 = central_diff_y(v, dy)
    S33 = central_diff_z(w, dz)
    
    # 2. Off-diagonal components (shear strains)
    # S12 = 0.5 * (du/dy + dv/dx)
    du_dy = central_diff_y(u, dy)
    dv_dx = central_diff_x(v, dx)
    S12 = 0.5 * (du_dy + dv_dx)
    
    # S13 = 0.5 * (du/dz + dw/dx)
    du_dz = central_diff_z(u, dz)
    dw_dx = central_diff_x(w, dx)
    S13 = 0.5 * (du_dz + dw_dx)
    
    # S23 = 0.5 * (dv/dz + dw/dy)
    dv_dz = central_diff_z(v, dz)
    dw_dy = central_diff_y(w, dy)
    S23 = 0.5 * (dv_dz + dw_dy)
    
    return S11, S22, S33, S12, S13, S23

@jit
def compute_strain_magnitude(S11, S22, S33, S12, S13, S23):
    """
    Compute the magnitude of the strain rate tensor |S| = sqrt(2 * S_ij * S_ij).
    """
    # Contract tensor: S_ij S_ij
    # S11^2 + S22^2 + S33^2 + 2*S12^2 + 2*S13^2 + 2*S23^2
    # The factor of 2 for off-diagonals accounts for symmetry (S12=S21, etc.)
    
    S_contracted = (S11**2 + S22**2 + S33**2 + 
                    2.0 * S12**2 + 2.0 * S13**2 + 2.0 * S23**2)
    
    return jnp.sqrt(2.0 * S_contracted)

@jit
def compute_smagorinsky_viscosity(S11, S22, S33, S12, S13, S23, cs, delta):
    """
    Compute the Smagorinsky eddy viscosity nu_t.
    
    nu_t = (Cs * delta)^2 * |S|
    
    Args:
        S11-S23: Strain rate components
        cs: Smagorinsky constant
        delta: Filter width (grid scale)
        
    Returns:
        nu_t: Eddy viscosity field
    """
    S_mag = compute_strain_magnitude(S11, S22, S33, S12, S13, S23)
    nu_t = (cs * delta)**2 * S_mag
    
    # Ensure non-negative (mathematically it fits squares, but good to be safe if numerical noise)
    return jnp.maximum(nu_t, 0.0)

@jit
def compute_filter_width(dx, dy, dz):
    """
    Compute filter width delta = (dx * dy * dz)^(1/3)
    """
    return (dx * dy * dz)**(1.0/3.0)
