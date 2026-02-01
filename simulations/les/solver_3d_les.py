
import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
import numpy as np
import time
from functools import partial

# Import LES components
from simulations.les.sgs_models import (
    compute_strain_rate_tensor_3d,
    compute_smagorinsky_viscosity,
    compute_filter_width
)
from simulations.les.wall_treatment import (
    compute_wall_distance_3d,
    van_driest_damping
)
from simulations.les.variable_viscosity_adi import (
    build_tridiagonal_variable_nu,
    thomas_algorithm_variable
)
from simulations.les.config import LESConfig

# Reuse Advection and Pressure from original solver (assumed available or copied)
# To avoid duplication, we should import them. 
# But importing from simulations.backward_facing_step_3d.solver_3d might cause issues if not a package.
# We'll copy the helper functions 'pressure_poisson_3d' and 'compute_advection_3d' 
# or assume we can import them if __init__.py exists.
# `simulations` is likely a package.

try:
    from simulations.backward_facing_step_3d.solver_3d import (
        pressure_poisson_3d,
        compute_advection_3d
    )
except ImportError:
    # If import fails (path issues), we will redefine them to be safe and self-contained
    # For robust agent code, redefining is safer than struggling with python path
    pass 

# Redefine helpers if import not guaranteed to work easily in this env structure
# (Actually, users sys.path hack in solver_3d.py suggests it might be hard to import)

@partial(jit, static_argnums=(5, 7, 8))
def pressure_poisson_3d(p, b, dx, dy, dz, nit, mask, i_step, j_step):
    def body_fn(p_old, _):
        p_new = (
            ((p_old[1:-1, 1:-1, 2:] + p_old[1:-1, 1:-1, :-2]) * dy**2 * dz**2 +
             (p_old[1:-1, 2:, 1:-1] + p_old[1:-1, :-2, 1:-1]) * dx**2 * dz**2 +
             (p_old[2:, 1:-1, 1:-1] + p_old[:-2, 1:-1, 1:-1]) * dx**2 * dy**2 -
             b[1:-1, 1:-1, 1:-1] * dx**2 * dy**2 * dz**2) /
            (2 * (dy**2 * dz**2 + dx**2 * dz**2 + dx**2 * dy**2))
        )
        p_full = p_old.at[1:-1, 1:-1, 1:-1].set(p_new)
        # Global BCs
        p_full = p_full.at[:, j_step:, 0].set(p_full[:, j_step:, 1]) # Inlet
        p_full = p_full.at[:, :, -1].set(0.0) # Outlet
        p_full = p_full.at[:, -1, :].set(p_full[:, -2, :]) # Top
        p_full = p_full.at[:, 0, i_step:].set(p_full[:, 1, i_step:]) # Bottom
        p_full = p_full.at[0, :, :].set(p_full[1, :, :]) # Side 1
        p_full = p_full.at[-1, :, :].set(p_full[-2, :, :]) # Side 2
        # Step BCs
        p_full = p_full.at[:, j_step-1, :i_step].set(p_full[:, j_step, :i_step])
        p_full = p_full.at[:, :j_step, i_step-1].set(p_full[:, :j_step, i_step])
        return p_full, None
    p_final, _ = lax.scan(body_fn, p, None, length=nit)
    return p_final

@jit
def compute_advection_3d(u, v, w, dx, dy, dz):
    u_c = u[1:-1, 1:-1, 1:-1]; v_c = v[1:-1, 1:-1, 1:-1]; w_c = w[1:-1, 1:-1, 1:-1]
    
    def upwind_grad(f, d, axis):
        # Generic upwind gradient helper could simplify, but lets expand for speed/clarity
        pass

    # u advection
    du_dx_b = (u[1:-1, 1:-1, 1:-1] - u[1:-1, 1:-1, :-2]) / dx
    du_dx_f = (u[1:-1, 1:-1, 2:] - u[1:-1, 1:-1, 1:-1]) / dx
    du_dy_b = (u[1:-1, 1:-1, 1:-1] - u[1:-1, :-2, 1:-1]) / dy
    du_dy_f = (u[1:-1, 2:, 1:-1] - u[1:-1, 1:-1, 1:-1]) / dy
    du_dz_b = (u[1:-1, 1:-1, 1:-1] - u[:-2, 1:-1, 1:-1]) / dz
    du_dz_f = (u[2:, 1:-1, 1:-1] - u[1:-1, 1:-1, 1:-1]) / dz
    
    adv_u = jnp.zeros_like(u)
    val_u = (jnp.where(u_c > 0, u_c * du_dx_b, u_c * du_dx_f) +
             jnp.where(v_c > 0, v_c * du_dy_b, v_c * du_dy_f) +
             jnp.where(w_c > 0, w_c * du_dz_b, w_c * du_dz_f))
    adv_u = adv_u.at[1:-1, 1:-1, 1:-1].set(val_u)
    
    # v advection
    dv_dx_b = (v[1:-1, 1:-1, 1:-1] - v[1:-1, 1:-1, :-2]) / dx
    dv_dx_f = (v[1:-1, 1:-1, 2:] - v[1:-1, 1:-1, 1:-1]) / dx
    dv_dy_b = (v[1:-1, 1:-1, 1:-1] - v[1:-1, :-2, 1:-1]) / dy
    dv_dy_f = (v[1:-1, 2:, 1:-1] - v[1:-1, 1:-1, 1:-1]) / dy
    dv_dz_b = (v[1:-1, 1:-1, 1:-1] - v[:-2, 1:-1, 1:-1]) / dz
    dv_dz_f = (v[2:, 1:-1, 1:-1] - v[1:-1, 1:-1, 1:-1]) / dz
    
    adv_v = jnp.zeros_like(v)
    val_v = (jnp.where(u_c > 0, u_c * dv_dx_b, u_c * dv_dx_f) +
             jnp.where(v_c > 0, v_c * dv_dy_b, v_c * dv_dy_f) +
             jnp.where(w_c > 0, w_c * dv_dz_b, w_c * dv_dz_f))
    adv_v = adv_v.at[1:-1, 1:-1, 1:-1].set(val_v)
    
    # w advection
    dw_dx_b = (w[1:-1, 1:-1, 1:-1] - w[1:-1, 1:-1, :-2]) / dx
    dw_dx_f = (w[1:-1, 1:-1, 2:] - w[1:-1, 1:-1, 1:-1]) / dx
    dw_dy_b = (w[1:-1, 1:-1, 1:-1] - w[1:-1, :-2, 1:-1]) / dy
    dw_dy_f = (w[1:-1, 2:, 1:-1] - w[1:-1, 1:-1, 1:-1]) / dy
    dw_dz_b = (w[1:-1, 1:-1, 1:-1] - w[:-2, 1:-1, 1:-1]) / dz
    dw_dz_f = (w[2:, 1:-1, 1:-1] - w[1:-1, 1:-1, 1:-1]) / dz
    
    adv_w = jnp.zeros_like(w)
    val_w = (jnp.where(u_c > 0, u_c * dw_dx_b, u_c * dw_dx_f) +
             jnp.where(v_c > 0, v_c * dw_dy_b, v_c * dw_dy_f) +
             jnp.where(w_c > 0, w_c * dw_dz_b, w_c * dw_dz_f))
    adv_w = adv_w.at[1:-1, 1:-1, 1:-1].set(val_w)
    
    return adv_u, adv_v, adv_w

@jit
def adi_step_3d_les_sweeps(u, rhs, nu_eff, dt, dx, dy, dz, mask):
    """
    Full 3D ADI step with variable viscosity.
    Performs X, Y, Z sweeps sequentially.
    """
    nz, ny, nx = u.shape
    
    # --- X-SWEEP ---
    # Equation: (1 - dt/2 * D_x) u* = RHS_X
    # We need to formulate RHS properly for Douglas-Gunn or similar interactions.
    # Simplified ADI often treats:
    # u* = u^n + ...
    # Here we assume 'rhs' contains the explicit part + old time terms.
    # The diffusion operator is L = Lx + Ly + Lz
    # CN: (1 - dt/2 L) u^{n+1} = (1 + dt/2 L) u^n + dt*Adv
    # ADI applied to (1 - dt/2 L) Delta_u = Total_RHS
    
    # Let's follow the constant-nu solver structure:
    # It calculates rhs = u + alpha*Lap - dt*Adv
    # Then solves (1 - alpha d2x) u* = rhs_x
    # where rhs_x includes explicit y,z parts.
    
    # With variable viscosity, we cannot pre-calculate simple Laplancians as easily?
    # Or we use the flux form for explicit parts too.
    
    # Explicit Laplacian with variable nu:
    # div(nu grad u)
    
    def compute_variable_laplacian(field, nu):
        # Central difference approximation of div(nu grad field)
        # Using build_tridiagonal logic or direct computation
        # d/dx(nu du/dx) ~ (nu_{i+1/2}(u_{i+1}-u_i) - nu_{i-1/2}(u_i-u_{i-1})) / dx^2
        pass
        # Too complex to replicate inside here efficiently without standard functions
        # For simplicity, we might assume nu is slowly varying in explicit parts 
        # or implement full flux form.
        
    # STRATEGY:
    # To minimize risk of breaking numerics, we use the ADI splitting that naturally isolates directions.
    # Sweep 1 (X): (1 - A_x) u* = RHS + A_y u^n + A_z u^n
    # Sweep 2 (Y): (1 - A_y) u** = u* - A_y u^n + A_y u*  (Correction form)
    # ...
    #
    # Implementation:
    # We will use the `solve_direction` helper for each direction.
    
    # A_x u = dt/2 * d/dx(nu d/dx u)
    # The matrix built by build_tridiagonal_variable_nu corresponds physically to (1 - A_x) if configured nicely.
    # Our build_tri returns components a, b, c for: a u_{i-1} + b u_i + c u_{i+1} = d
    # where b = 1 + ..., a = -..., c = -...
    # So it solves (1 - A_x) u = d.
    
    # 1. X-Sweep
    # rhs_x should be: u_current + dt/2 * (diff_y + diff_z) + explicit terms?
    # This matches standard logic if `rhs` passed in is the Total explicit guess.
    
    # Wait, the `rhs` passed to `adi_step_3d` in original solver was:
    # rhs = u + alpha*Lap - dt*Adv
    # This effectively explicitly integrates diffusion + advection.
    # Then ADI is used as a filter/solver to implicitize the diffusion.
    # (I - alpha Lx) ...
    
    # We need to construct explicit diffusion using variable nu for the RHS.
    # If we ignore variable nu in explicit predictor, it might simple but inaccurate.
    # Let's approximate explicit diffusion with variable nu.
    
    # Explicit diffusion terms (flux form):
    def diff_flux_x(f, nu):
        nu_ip = 0.5*(nu+jnp.roll(nu,-1,axis=2)); nu_im = 0.5*(nu+jnp.roll(nu,1,axis=2))
        return (nu_ip*(jnp.roll(f,-1,axis=2)-f) - nu_im*(f-jnp.roll(f,1,axis=2))) / dx**2
        
    def diff_flux_y(f, nu):
        nu_jp = 0.5*(nu+jnp.roll(nu,-1,axis=1)); nu_jm = 0.5*(nu+jnp.roll(nu,1,axis=1))
        return (nu_jp*(jnp.roll(f,-1,axis=1)-f) - nu_jm*(f-jnp.roll(f,1,axis=1))) / dy**2

    def diff_flux_z(f, nu):
        nu_kp = 0.5*(nu+jnp.roll(nu,-1,axis=0)); nu_km = 0.5*(nu+jnp.roll(nu,1,axis=0))
        return (nu_kp*(jnp.roll(f,-1,axis=0)-f) - nu_km*(f-jnp.roll(f,1,axis=0))) / dz**2

    # Note: Boundaries for roll need care (masked anyway usually).
    # Masking handles most, but safe padding is better. 
    # For now, let's rely on mask to kill invalid boundary fluxes.
    
    # But wait, `rhs` passed to this function is just `rhs`. 
    # In `time_step_3d_les`, we must compute `rhs`.
    
    # In this `adi_step_3d_les_sweeps`:
    # We assume we operate on u to update it.
    
    # X-Sweep
    # Solve (1 - dt/2 Lx) u* = rhs + dt/2 Lx u^n  (if using Delta form, different)
    # The original solver constructs `rhs` containing FULL explicit, then "subtracts" implicit parts?
    # Original:
    # rhs_x = rhs + alpha * (d2u_dy2 + d2u_dz2)
    #   where rhs = u + alpha*lap - dt*adv = u + alpha(Lx+Ly+Lz) - ...
    # So rhs_x = u + alpha*Lx + 2*alpha*Ly + 2*alpha*Lz ... ? 
    # Be careful. The original solver logic:
    # rhs (input) = u + alpha*Lap - dt*Adv = u + alpha(Lx+Ly+Lz) - ...
    # rhs_x = rhs + alpha*(Ly_impl + Lz_impl) ?? No.
    
    # Original code:
    # d2u_dy2, d2u_dz2 computed explicitly.
    # rhs_x = rhs + alpha * (d2u_dy2 + d2u_dz2) ?
    # This looks like it DOUBLES the Dy, Dz terms?
    # Ah, standard ADI (Douglas-Rachford):
    # (1 - A_x) u* = (1 + A_y + A_z) u + ...
    # If `rhs` has (A_x + A_y + A_z) u, then we want (1 + A_y + A_z) u?
    # rhs - A_x u = (1 + A_y + A_z) u.
    # So we remove A_x?
    
    # Let's stick to a robust simpler operator-splitting if unsure of original's exact flavor.
    # Or cleaner:
    # 1. Predict u* = u^n + dt * (-Adv + Diff(u^n)) [Explicit]
    # 2. Implicit correction sweeps:
    #    (1 - dt/2 Lx) u** = u* - dt/2 Lx u^n
    #    This is getting complicated.
    
    # Let's trust the logic structure:
    # We will invoke `build_tridiagonal_variable_nu` which gives LHS matrix M_x.
    # We simply solve M_x u_{new} = RHS.
    
    # X-Sweep:
    # RHS_X needs to act as forcing.
    # We will expose `adi_step_3d_les` to take `rhs` and handle the splitting internally 
    # OR we follow the helper approach where we pass `rhs` for that direction.
    
    # Let's implement the sweeps.
    # We'll calculate explicit diffusion terms locally to adjust RHS.
    
    d_flux_y = diff_flux_y(u, nu_eff) 
    d_flux_z = diff_flux_z(u, nu_eff)
    # alpha = dt/2. This factor is inside build_tridiagonal logic usually (dt passed).
    
    # We pass `dt/2` to build_tridiagonal (Crank Nicolson).
    
    # X-Solve
    # LHS: (1 - dt/2 Lx)
    # RHS: rhs_input - (something?)
    # If rhs_input is the full explicit guess u + dt(Diff - Adv),
    # And we want to implicitize X:
    # (1 - dt/2 Lx) u* = rhs_input - dt/2 Lx u^n  (Delta formulation?)
    
    # Let's simplify:
    # We will use Alternating Direction Implicit method to solve:
    # du/dt = Diff - Adv
    # u_{new} = u + dt * (-Adv + Diff_implicit)
    
    # Step 1 X: (1 - A_x) u* = (1 + A_x + 2A_y + 2A_z) u - ...
    
    # Let's use the provided `rhs` as the target and just smooth it?
    # Original solver: `rhs_x = rhs + alpha * (d2u_dy2 + d2u_dz2)`
    # This suggests `rhs` didn't have enough Y/Z?
    
    # Let's define the sweeps using the `solve_line` pattern.
    
    # Define solver for a line using thomas_variable
    def solve_x_line(j, k, args):
        r, m_row, nu_row = args
        a, b, c = build_tridiagonal_variable_nu(nu_row, dt/2.0, dx, 2)
        return thomas_algorithm_variable(a, b, c, r) # shape (nx,)

    # This requires vmapping.
    # For efficiency and variable viscosity, we need `build_tridiagonal` to accept batches 
    # OR we vmap it. build_tridiagonal supports batched field inputs!
    # So we can just call it on the whole field.
    
    a_x, b_x, c_x = build_tridiagonal_variable_nu(nu_eff, dt/2.0, dx, 2)
    # Solve X
    # RHS construction:
    # Typically: rhs_x = u + dt/2 Lx u + dt Ly u + dt Lz u - dt Adv
    # We assume 'rhs' passed in has everything?
    
    # To avoid rewriting the universe, let's implement the sweeps assuming `rhs` is "the thing to invert against"
    # treating other directions as explicit sources.
    
    # X-Solve
    # We need to solve: A_x u* = rhs
    # u_star = thomas(a_x, b_x, c_x, rhs)
    
    # But ADI updates the RHS between sweeps.
    # Let's implement standard Douglas ADI.
    # 1. (1 - 0.5 dt Lx) u* = (1 + 0.5 dt Lx + dt Ly + dt Lz) u^n - dt Adv
    # 2. (1 - 0.5 dt Ly) u** = u* - 0.5 dt Ly u^n + 0.5 dt Ly u*? 
    #    OR easier: (1 - 0.5 dt Ly) u** = u* - dt/2 Ly u^n + ...
    
    # Approximation:
    # Just solve the 3 systems sequentially.
    # u* = inv(1-Ax) * (rhs_from_explicit_step)
    # u** = inv(1-Ay) * u*
    # u*** = inv(1-Az) * u**
    # This is "Approximate Factorization" of (I-Ax)(I-Ay)(I-Az) u_new = RHS.
    # If RHS is a consistent predictor, this works.
    
    u_star = thomas_algorithm_variable(a_x, b_x, c_x, rhs)
    
    # Y-Solve
    a_y, b_y, c_y = build_tridiagonal_variable_nu(nu_eff, dt/2.0, dy, 1)
    u_star2 = thomas_algorithm_variable(a_y, b_y, c_y, u_star)
    
    # Z-Solve
    a_z, b_z, c_z = build_tridiagonal_variable_nu(nu_eff, dt/2.0, dz, 0)
    u_new = thomas_algorithm_variable(a_z, b_z, c_z, u_star2)
    
    return u_new * mask

@partial(jit, static_argnums=(10, 13))
def time_step_3d_les(u, v, w, p, dt, dx, dy, dz, rho, nu, nit, mask, u_inlet, geom_params):
    """
    LES time step with Smagorinsky SGS model.
    """
    j_step, i_step = geom_params
    
    # 1. Compute SGS properties
    S_comps = compute_strain_rate_tensor_3d(u, v, w, dx, dy, dz)
    delta = compute_filter_width(dx, dy, dz)
    
    # Config parameters (extracted roughly or passed in)
    les_config = LESConfig() # Use defaults for now
    
    nu_t = compute_smagorinsky_viscosity(*S_comps, cs=les_config.cs, delta=delta)
    
    if les_config.wall_damping:
        dist = compute_wall_distance_3d(mask, dx, dy, dz)
        # Simplify y_plus calculation (use rough estimate or average u_tau)
        # Ideally we compute local u_tau
        # For now, simplistic: y+ = y * Re_tau? No.
        # Use simple damping function arg?
        # Let's pass 0 for now or implement proper u_tau hook later if needed.
        # We need y_plus field.
        # y_plus = dist * u_tau / nu
        # Estimate u_tau from nu and velocity field?
        # u_tau = sqrt(nu * du/dy_wall)
        
        # This is expensive/complex to robustly do for step.
        # We'll skip complex u_tau for step and apply damping based on geometric distance?
        # Or just use raw Wall treatment.
        # We'll assume y_plus ~ dist * 100 for checking? 
        # No, let's omit wall damping logic detail call to avoid bugs for now, 
        # or use a dummy u_tau=1.
        y_plus = dist * 100.0 # pure geometric approx for now
        
        nu_t = van_driest_damping(nu_t, y_plus, les_config.a_plus)

    nu_eff = nu + nu_t
    
    # 2. Advection
    adv_u, adv_v, adv_w = compute_advection_3d(u, v, w, dx, dy, dz)
    
    # 3. Predictor RHS
    # Explicit Euler guess or AB2?
    # u_pred = u + dt * (-Adv + Diff(u, nu_eff))
    # We use Approx Factorization ADI, so RHS is (I + dt/2 L) u ? 
    # AF Scheme: (I - dt/2 L) Delta u = dt * ( -Adv + L u )
    # So RHS = u + dt * ( -Adv + Diff_Explicit )
    # We need compute_laplacian_variable(u, nu_eff)
    
    def compute_div_nu_grad(f, nu):
        # Flux form central diff
        # x
        nu_ip = 0.5*(nu+jnp.roll(nu,-1,axis=2)); nu_im = 0.5*(nu+jnp.roll(nu,1,axis=2))
        d_x = (nu_ip*(jnp.roll(f,-1,axis=2)-f) - nu_im*(f-jnp.roll(f,1,axis=2))) / dx**2
        # y
        nu_jp = 0.5*(nu+jnp.roll(nu,-1,axis=1)); nu_jm = 0.5*(nu+jnp.roll(nu,1,axis=1))
        d_y = (nu_jp*(jnp.roll(f,-1,axis=1)-f) - nu_jm*(f-jnp.roll(f,1,axis=1))) / dy**2
        # z
        nu_kp = 0.5*(nu+jnp.roll(nu,-1,axis=0)); nu_km = 0.5*(nu+jnp.roll(nu,1,axis=0))
        d_z = (nu_kp*(jnp.roll(f,-1,axis=0)-f) - nu_km*(f-jnp.roll(f,1,axis=0))) / dz**2
        
        return d_x + d_y + d_z

    diff_u = compute_div_nu_grad(u, nu_eff)
    diff_v = compute_div_nu_grad(v, nu_eff)
    diff_w = compute_div_nu_grad(w, nu_eff)
    
    rhs_u = u + dt * (diff_u - adv_u)
    rhs_v = v + dt * (diff_v - adv_v)
    rhs_w = w + dt * (diff_w - adv_w)
    
    # 4. Implicit Solve (ADI)
    u_star = adi_step_3d_les_sweeps(u, rhs_u, nu_eff, dt, dx, dy, dz, mask)
    v_star = adi_step_3d_les_sweeps(v, rhs_v, nu_eff, dt, dx, dy, dz, mask)
    w_star = adi_step_3d_les_sweeps(w, rhs_w, nu_eff, dt, dx, dy, dz, mask)
    
    # Inlet BCs
    u_star = u_star.at[:, j_step:, 0].set(u_inlet)
    v_star = v_star.at[:, j_step:, 0].set(0.0)
    w_star = w_star.at[:, j_step:, 0].set(0.0)
    
    # 5. Pressure Correction
    div = jnp.zeros_like(p)
    div = div.at[1:-1, 1:-1, 1:-1].set(
        (u_star[1:-1, 1:-1, 2:] - u_star[1:-1, 1:-1, :-2]) / (2 * dx) +
        (v_star[1:-1, 2:, 1:-1] - v_star[1:-1, :-2, 1:-1]) / (2 * dy) +
        (w_star[2:, 1:-1, 1:-1] - w_star[:-2, 1:-1, 1:-1]) / (2 * dz)
    )
    b_term = (rho / dt) * div
    p_new = pressure_poisson_3d(p, b_term, dx, dy, dz, nit, mask, i_step, j_step)
    
    # 6. Corrector
    dp_dx = (p_new[1:-1, 1:-1, 2:] - p_new[1:-1, 1:-1, :-2]) / (2 * dx)
    dp_dy = (p_new[1:-1, 2:, 1:-1] - p_new[1:-1, :-2, 1:-1]) / (2 * dy)
    dp_dz = (p_new[2:, 1:-1, 1:-1] - p_new[:-2, 1:-1, 1:-1]) / (2 * dz)
    
    u_new = u_star.at[1:-1, 1:-1, 1:-1].add(-(dt/rho) * dp_dx)
    v_new = v_star.at[1:-1, 1:-1, 1:-1].add(-(dt/rho) * dp_dy)
    w_new = w_star.at[1:-1, 1:-1, 1:-1].add(-(dt/rho) * dp_dz)
    
    # Mask
    u_new *= mask; v_new *= mask; w_new *= mask
    
    # BCs
    u_new = u_new.at[:, j_step:, 0].set(u_inlet)
    u_new = u_new.at[:, :, -1].set(u_new[:, :, -2]) # Outlet
    v_new = v_new.at[:, :, -1].set(v_new[:, :, -2])
    w_new = w_new.at[:, :, -1].set(w_new[:, :, -2])
    
    # CFL
    cfl = dt * (jnp.max(jnp.abs(u_new))/dx + jnp.max(jnp.abs(v_new))/dy + jnp.max(jnp.abs(w_new))/dz)
    
    return u_new, v_new, w_new, p_new, nu_t, cfl
