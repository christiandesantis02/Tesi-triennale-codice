import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.optim.lr_scheduler import StepLR
import time

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ═══════════════════════════════════════════════════════════════
# IPERPARAMETRI
# ═══════════════════════════════════════════════════════════════

RE           = 500.0
T_END        = 1.0
N_COLLOC     = 5000
N_IC         = 1000
N_BC         = 400

LAYERS       = 6
NEURONS      = 64

LR           = 1e-3
EPOCHS_ADAM  = 3000
EPOCHS_LBFGS = 1200

LAM_NS       = 1.0
LAM_CONT     = 1.0
LAM_IC       = 5.0
LAM_BC       = 1.0

LOG_EVERY    = 200

T_PLOTS      = [0.5]


# ═══════════════════════════════════════════════════════════════
# ARCHITETTURA DELLA RETE NEURALE
# ═══════════════════════════════════════════════════════════════

class PINN(nn.Module):
    def __init__(self, layers: int = LAYERS, neurons: int = NEURONS):
        super().__init__()
        dims = [3] + [neurons] * layers + [3]
        self.net = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.net.append(nn.Linear(dims[i], dims[i + 1]))

        for layer in self.net[:-1]:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, y: torch.Tensor,
                t: torch.Tensor) -> tuple:
        inp = torch.stack([x, y, t], dim=1)
        out = inp
        for layer in self.net[:-1]:
            out = torch.tanh(layer(out))
        out = self.net[-1](out)
        return out[:, 0], out[:, 1], out[:, 2]  # u, v, p


# ═══════════════════════════════════════════════════════════════
# SOLUZIONE ANALITICA
# ═══════════════════════════════════════════════════════════════

def exact_solution(x: torch.Tensor, y: torch.Tensor,
                   t: torch.Tensor, Re: float = RE):
    decay = torch.exp(-2.0 * t / Re)
    u =  torch.sin(x) * torch.cos(y) * decay
    v = -torch.cos(x) * torch.sin(y) * decay
    p = +0.25 * (torch.cos(2.0 * x) + torch.cos(2.0 * y)) * decay**2
    return u, v, p


# ═══════════════════════════════════════════════════════════════
# DERIVATE VIA AUTOMATIC DIFFERENTIATION
# ═══════════════════════════════════════════════════════════════

def grad(output: torch.Tensor, inp: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        output, inp,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
        retain_graph=True
    )[0]


def pde_residuals(model: PINN, x: torch.Tensor, y: torch.Tensor,
                  t: torch.Tensor, Re: float = RE) -> tuple:
    u, v, p = model(x, y, t)

    u_t = grad(u, t);  u_x = grad(u, x);  u_y = grad(u, y)
    v_t = grad(v, t);  v_x = grad(v, x);  v_y = grad(v, y)
    p_x = grad(p, x);  p_y = grad(p, y)

    u_xx = grad(u_x, x);  u_yy = grad(u_y, y)
    v_xx = grad(v_x, x);  v_yy = grad(v_y, y)

    r_u = u_t + u * u_x + v * u_y + p_x - (u_xx + u_yy) / Re
    r_v = v_t + u * v_x + v * v_y + p_y - (v_xx + v_yy) / Re
    r_c = u_x + v_y

    return r_u, r_v, r_c


# ═══════════════════════════════════════════════════════════════
# CAMPIONAMENTO DEI PUNTI
# ═══════════════════════════════════════════════════════════════

def sample_collocation(n: int) -> tuple:
    x = torch.rand(n) * 2 * np.pi
    y = torch.rand(n) * 2 * np.pi
    t = torch.rand(n) * T_END
    return (x.to(device).requires_grad_(True),
            y.to(device).requires_grad_(True),
            t.to(device).requires_grad_(True))


def sample_ic(n: int) -> tuple:
    x = torch.rand(n) * 2 * np.pi
    y = torch.rand(n) * 2 * np.pi
    t = torch.zeros(n)
    return (x.to(device).requires_grad_(True),
            y.to(device).requires_grad_(True),
            t.to(device).requires_grad_(True))


def sample_bc(n_per_side: int) -> tuple:
    s    = torch.rand(n_per_side) * 2 * np.pi
    t_bc = torch.rand(n_per_side) * T_END

    x0 = torch.zeros(n_per_side)
    x1 = torch.full((n_per_side,), 2 * np.pi)
    y0 = torch.zeros(n_per_side)
    y1 = torch.full((n_per_side,), 2 * np.pi)

    def to_dev(*tensors):
        return [t_.to(device).requires_grad_(True) for t_ in tensors]

    left  = to_dev(x0, s.clone(), t_bc.clone())
    right = to_dev(x1, s.clone(), t_bc.clone())
    bot   = to_dev(s.clone(), y0, t_bc.clone())
    top   = to_dev(s.clone(), y1, t_bc.clone())

    return left, right, bot, top


# ═══════════════════════════════════════════════════════════════
# FUNZIONE DI LOSS
# ═══════════════════════════════════════════════════════════════

def compute_loss(model: PINN,
                 col_pts: tuple,
                 ic_pts:  tuple,
                 bc_pts:  tuple,
                 lam:     tuple = (LAM_NS, LAM_CONT, LAM_IC, LAM_BC)) -> tuple:
    lam_ns, lam_cont, lam_ic, lam_bc = lam

    # ── L_ns + L_cont ─────────────────────────────────────────────
    r_u, r_v, r_c = pde_residuals(model, *col_pts)
    L_ns   = (r_u ** 2 + r_v ** 2).mean()
    L_cont = (r_c ** 2).mean()

    # ── L_ic ──────────────────────────────────────────────────────
    u0_ex, v0_ex, p0_ex = exact_solution(*ic_pts)
    u0_pr, v0_pr, p0_pr = model(*ic_pts)
    L_ic = ((u0_pr - u0_ex) ** 2 +
            (v0_pr - v0_ex) ** 2 +
            (p0_pr - p0_ex) ** 2).mean()

    # ── L_bc ──────────────────────────────────────────────────────
    left, right, bot, top = bc_pts
    u_l, v_l, p_l = model(*left)
    u_r, v_r, p_r = model(*right)
    u_b, v_b, p_b = model(*bot)
    u_t, v_t, p_t = model(*top)
    L_bc = ((u_l - u_r) ** 2 + (v_l - v_r) ** 2 + (p_l - p_r) ** 2 +
            (u_b - u_t) ** 2 + (v_b - v_t) ** 2 + (p_b - p_t) ** 2).mean()

    # ── Loss totale pesata ────────────────────────────────────────
    loss = (lam_ns   * L_ns   +
            lam_cont * L_cont +
            lam_ic   * L_ic   +
            lam_bc   * L_bc)

    components = {
        "L_ns":   L_ns.item(),
        "L_cont": L_cont.item(),
        "L_ic":   L_ic.item(),
        "L_bc":   L_bc.item(),
        "total":  loss.item(),
    }
    return loss, components


# ═══════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════

def train():
    model = PINN(LAYERS, NEURONS).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parametri rete: {n_params:,}")
    print(f"Re = {RE},  dominio [0,2π]² x [0,{T_END}]\n")

    optimizer_adam = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler      = StepLR(optimizer_adam, step_size=1000, gamma=0.5)

    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(), lr=0.1,
        max_iter=15,
        history_size=20,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        line_search_fn="strong_wolfe"
    )

    col_pts = sample_collocation(N_COLLOC)
    ic_pts  = sample_ic(N_IC)
    bc_pts  = sample_bc(N_BC)

    history = {"total": [], "L_ns": [], "L_cont": [], "L_ic": [], "L_bc": []}
    t0 = time.time()

    # ── Fase 1: Adam ─────────────────────────────────────────────
    print("── Fase Adam ──────────────────────────────")
    for ep in range(1, EPOCHS_ADAM + 1):
        optimizer_adam.zero_grad()
        loss, comp = compute_loss(model, col_pts, ic_pts, bc_pts)
        loss.backward()
        optimizer_adam.step()
        scheduler.step()

        for k in history:
            history[k].append(comp[k])

        if ep % LOG_EVERY == 0 or ep == 1:
            elapsed = time.time() - t0
            print(f"  ep {ep:5d} | loss {comp['total']:.4e} | "
                  f"NS {comp['L_ns']:.4e} | cont {comp['L_cont']:.4e} | "
                  f"IC {comp['L_ic']:.4e} | BC {comp['L_bc']:.4e} | "
                  f"{elapsed:.1f}s")

    # ── Fase 2: L-BFGS ───────────────────────────────────────────
    print("\n── Fase L-BFGS ────────────────────────────")
    for ep in range(1, EPOCHS_LBFGS + 1):
        def closure():
            optimizer_lbfgs.zero_grad()
            loss, _ = compute_loss(model, col_pts, ic_pts, bc_pts)
            loss.backward()
            return loss

        optimizer_lbfgs.step(closure)
        _, comp = compute_loss(model, col_pts, ic_pts, bc_pts)
        for k in history:
            history[k].append(comp[k])

        if ep % (LOG_EVERY // 4) == 0 or ep == 1:
            print(f"  ep {ep:4d} | loss {comp['total']:.4e}")

    print(f"\nTraining completato in {time.time()-t0:.1f}s")
    return model, history


# ═══════════════════════════════════════════════════════════════
# VALUTAZIONE
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model: PINN, t_plot: float, N: int = 100) -> dict:
    x_ = torch.linspace(0, 2 * np.pi, N)
    y_ = torch.linspace(0, 2 * np.pi, N)
    X, Y = torch.meshgrid(x_, y_, indexing="ij")
    x_flat = X.reshape(-1).to(device)
    y_flat = Y.reshape(-1).to(device)
    t_flat = torch.full_like(x_flat, t_plot)

    u_pr, v_pr, p_pr = model(x_flat, y_flat, t_flat)
    u_ex, v_ex, p_ex = exact_solution(x_flat, y_flat, t_flat)

    def l2_rel(pred, exact):
        return (torch.norm(pred - exact) / torch.norm(exact)).item()

    return {
        "t":      t_plot,
        "L2_u":   l2_rel(u_pr, u_ex),
        "L2_v":   l2_rel(v_pr, v_ex),
        "L2_p":   l2_rel(p_pr, p_ex),
        "X": X.numpy(), "Y": Y.numpy(),
        "u_pr": u_pr.cpu().reshape(N, N).numpy(),
        "v_pr": v_pr.cpu().reshape(N, N).numpy(),
        "p_pr": p_pr.cpu().reshape(N, N).numpy(),
        "u_ex": u_ex.cpu().reshape(N, N).numpy(),
        "v_ex": v_ex.cpu().reshape(N, N).numpy(),
        "p_ex": p_ex.cpu().reshape(N, N).numpy(),
    }


# ═══════════════════════════════════════════════════════════════
# DECADIMENTO ENERGIA CINETICA
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def energy_decay(model: PINN, Re: float, T_end: float = T_END,
                 n_times: int = 30, N: int = 50) -> tuple:
    times = np.linspace(0, T_end, n_times)
    E_pinn, E_exact = [], []

    x_ = torch.linspace(0, 2 * np.pi, N)
    y_ = torch.linspace(0, 2 * np.pi, N)
    X, Y = torch.meshgrid(x_, y_, indexing="ij")
    x_flat = X.reshape(-1).to(device)
    y_flat = Y.reshape(-1).to(device)

    for t_val in times:
        t_flat = torch.full_like(x_flat, t_val)
        u_pr, v_pr, _ = model(x_flat, y_flat, t_flat)
        u_ex, v_ex, _ = exact_solution(x_flat, y_flat, t_flat, Re)
        E_pinn.append((0.5 * (u_pr**2 + v_pr**2)).mean().item())
        E_exact.append((0.5 * (u_ex**2 + v_ex**2)).mean().item())

    return times, np.array(E_pinn), np.array(E_exact)


# ═══════════════════════════════════════════════════════════════
# VISUALIZZAZIONE
# ═══════════════════════════════════════════════════════════════

def plot_results(model: PINN, history: dict, metrics_list: list,
                 Re: float = RE):
    # ── Figura 1: andamento loss ──────────────────────────────────
    fig1, ax_loss = plt.subplots(figsize=(8, 5))
    epochs = np.arange(1, len(history["total"]) + 1)
    ax_loss.semilogy(epochs, history["total"],  label="loss totale", lw=1.5)
    ax_loss.semilogy(epochs, history["L_ns"],   label="L_NS",   lw=1.0, ls="--")
    ax_loss.semilogy(epochs, history["L_cont"], label="L_cont", lw=1.0, ls="--")
    ax_loss.semilogy(epochs, history["L_ic"],   label="L_IC",   lw=1.0, ls="--")
    ax_loss.semilogy(epochs, history["L_bc"],   label="L_BC",   lw=1.0, ls="--")
    ax_loss.set_xlabel("epoca"); ax_loss.set_ylabel("loss")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend(fontsize=8)
    fig1.tight_layout()

    # ── Figura 2: campi u, v, p (una figura per ogni istante) ─────
    for metrics in metrics_list:
        t_val = metrics["t"]
        X, Y  = metrics["X"], metrics["Y"]

        fig2, axs = plt.subplots(3, 3, figsize=(15, 15))
        for row, (name, label) in enumerate([("u", "u"), ("v", "v"), ("p", "p")]):
            ex, pr = metrics[f"{name}_ex"], metrics[f"{name}_pr"]
            vmin, vmax = min(ex.min(), pr.min()), max(ex.max(), pr.max())
            kw = dict(cmap="RdBu_r", vmin=vmin, vmax=vmax, shading="auto")

            for col, (data, title, extra) in enumerate([
                (ex, f"Soluzione analitica ({label})", kw),
                (pr, f"Soluzione PINN ({label})",   kw),
                (np.abs(pr - ex), f"Errore assoluto ({label})",
                 dict(cmap="hot_r", shading="auto")),
            ]):
                ax = axs[row, col]
                c = ax.pcolormesh(X, Y, data, **extra)
                plt.colorbar(c, ax=ax, shrink=0.8)
                ax.set_title(title)
                ax.set_aspect("equal")
                ax.set_xlabel("x"); ax.set_ylabel("y")
                ax.set_xticks([0, np.pi, 2*np.pi]); ax.set_xticklabels(["0", "π", "2π"])
                ax.set_yticks([0, np.pi, 2*np.pi]); ax.set_yticklabels(["0", "π", "2π"])


        fig2.tight_layout(h_pad=3.0)
        fig2.savefig(f"taylor_green_campi_t{t_val}.png", dpi=150, bbox_inches="tight")

    # ── Figura 3: decadimento energia cinetica ────────────────────
    times, E_pinn, E_exact = energy_decay(model, Re)
    fig3, ax_e = plt.subplots(figsize=(15, 3))
    ax_e.semilogy(times, E_exact, 'k-',  lw=2, label='esatta')
    ax_e.semilogy(times, E_pinn,  'r--', lw=2, label='PINN')
    ax_e.set_xlabel('t'); ax_e.set_ylabel('E(t)')

    ax_e.legend(); ax_e.grid(True, alpha=0.3)
    fig3.tight_layout()

    plt.show()

# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    model, history = train()

    metrics_list = []
    for t_val in T_PLOTS:
        m = evaluate(model, t_plot=t_val)
        metrics_list.append(m)
        print(f"\nt = {t_val}:")
        print(f"  L2 — u: {m['L2_u']:.3e}  v: {m['L2_v']:.3e}  "
              f"p: {m['L2_p']:.3e}")

    plot_results(model, history, metrics_list, Re=RE)

    torch.save(model.state_dict(), "taylor_green_pinn.pt")
    print("\nModello salvato: taylor_green_pinn.pt")
