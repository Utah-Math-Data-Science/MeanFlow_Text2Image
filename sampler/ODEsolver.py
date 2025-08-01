import torch
from torchdiffeq import odeint 
################################################################################
# Sampling (Euler & RK38 ODE integrators)
################################################################################
@torch.no_grad()
def euler_sampler(model, txt_tok: torch.Tensor, steps: int = 25):
    dt = 1.0 / steps
    z = txt_tok.clone()
    for i in range(steps):
        t = torch.full((z.size(0),), (i + 0.5) * dt, device=z.device, dtype=z.dtype)
        v = model(z, t)
        z = z + dt * v
    return z

@torch.no_grad()
def rk38_sampler(model, txt_tok: torch.Tensor, steps: int = 20):
    dt = 1.0 / steps
    z = txt_tok.clone()
    for i in range(steps):
        t0 = torch.full((z.size(0),), i * dt, device=z.device, dtype=z.dtype)
        k1 = model(z, t0)
        k2 = model(z + dt * k1 / 3, t0 + dt / 3)
        k3 = model(z + dt * (-k1/3 + k2), t0 + 2*dt/3)
        k4 = model(z + dt * (k1 - k2 + k3), t0 + dt)
        z = z + dt * (k1 + 3*k2 + 3*k3 + k4) / 8
    return z


@torch.no_grad()
def dopri5_sampler(
    model:      torch.nn.Module,          # FlowTokLite:  v = model(z, t_vec)
    txt_tok:    torch.Tensor,       # (B,4,32,32)  or (B,4096)
    t0: float = 0.0,
    t1: float = 1.0,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    max_steps: int = 1000,
    initial_dt: float | None = None,  # optional: hint for the first step
):
    """
    Adaptive Dormand–Prince (torchdiffeq) sampler.
    Returns the latent at t = t1.

    • `atol`, `rtol` – usual ODE tolerances (tune as you like).
    • `max_steps`    – fails fast if the solver would need absurdly many steps.
    • `initial_dt`   – set if you already know a good first step size.
    """
    device, dtype = txt_tok.device, txt_tok.dtype

    # torchdiffeq expects  func(t, y)  →  dy/dt
    def ode_func(t_scalar: torch.Tensor, z_tensor: torch.Tensor):
        # Broadcast scalar t → (B,) vector so your model API stays the same
        t_vec = torch.full(
            (z_tensor.size(0),), t_scalar.item(), device=device, dtype=dtype
        )
        return model(z_tensor, t_vec)

    t_span = torch.tensor([t0, t1], device=device, dtype=dtype)

    options = {"max_num_steps": max_steps}
    if initial_dt is not None:
        options["first_step"] = initial_dt

    # Integrate; result shape (2, B, …) – first row t0, second row t1
    z_traj = odeint(
        ode_func,
        txt_tok,
        t_span,
        method="dopri5",
        atol=atol,
        rtol=rtol,
        options=options,
    )

    return z_traj[-1]                # z at t = t1