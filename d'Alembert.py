import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Definizione parametri
R = 1.0
R_max = 4.0
U = 1.0
hidden_layers = 5
neurons = 64
lr = 1e-3
lr_lbfgs = 1.0
epochs = 1000
epochs_lbfgs = 800
max_iter_lbfgs = 5
history_size_lbfgs = 20
N_f = 3000
N_inner_bc = 500
N_outer_bc = 500
w_inner = 20.0
w_outer = 10.0
w_pde = 1.0
N_test = 250
seed = 1

# Struttura rete neurale
class PINN(nn.Module):
    def __init__(self, hidden_layers, neurons):
        super().__init__()
        layers = [nn.Linear(2, neurons), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(neurons, neurons), nn.Tanh()])
        layers.append(nn.Linear(neurons, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# Calcolo del laplaciano per il residuo pde
def laplacian(phi, x):
    grads = torch.autograd.grad(
        phi, x, grad_outputs=torch.ones_like(phi), create_graph=True)[0]
    phi_xx = torch.autograd.grad(
        grads[:, 0:1], x, grad_outputs=torch.ones_like(grads[:, 0:1]), create_graph=True)[0][:, 0:1]
    phi_yy = torch.autograd.grad(
        grads[:, 1:2], x, grad_outputs=torch.ones_like(grads[:, 1:2]), create_graph=True)[0][:, 1:2]
    return phi_xx + phi_yy

# Campionamento punti di collocazione e boundary
def sample_points(n, R, Rmax):
    n_near = n // 2
    n_far = n - n_near
    r_near = R + torch.rand(n_near, 1) * R
    th_near = torch.rand(n_near, 1) * 2 * torch.pi
    r_far = 2 * R + torch.rand(n_far, 1) * (Rmax - 2 * R)
    th_far = torch.rand(n_far, 1) * 2 * torch.pi
    r = torch.cat([r_near, r_far])
    th = torch.cat([th_near, th_far])  
    x = r * torch.cos(th)
    y = r * torch.sin(th)
    return torch.cat([x / R, y / R], dim=1)

def circle_boundary(n, R):
    theta = torch.rand(n, 1) * 2 * torch.pi
    x = R * torch.cos(theta)
    y = R * torch.sin(theta)
    return torch.cat([x / R, y / R], dim=1)

def outer_boundary(n, Rmax, R, U):
    theta = torch.rand(n, 1) * 2 * torch.pi
    x = Rmax * torch.cos(theta)
    y = Rmax * torch.sin(theta)
    phi_exact = U * (Rmax + R**2 / Rmax) * torch.cos(theta)
    return torch.cat([x / R, y / R], dim=1), phi_exact

# Soluzione analitica per confronto
def phi_analytical(x, y, U=1.0, R=1.0):
    r = torch.sqrt(x**2 + y**2) + 1e-8
    theta = torch.atan2(y, x)
    return U * (r + (R**2) / r) * torch.cos(theta)

# Funzione di perdita
def loss_fn(model, collocation, inner_bc, outer_bc_norm, outer_bc_phi, R, U, lam_pde, lam_inner, lam_outer):
    collocation.requires_grad_(True)
    phi = model(collocation)
    lap = laplacian(phi, collocation)
    loss_pde = torch.mean(lap**2)
    inner_bc.requires_grad_(True)
    phi_b = model(inner_bc)
    grads = torch.autograd.grad(
        phi_b, inner_bc, grad_outputs=torch.ones_like(phi_b), create_graph=True)[0]    
    x_b, y_b = inner_bc[:, 0:1], inner_bc[:, 1:2]
    r_b = torch.sqrt(x_b**2 + y_b**2) + 1e-8
    nx, ny = x_b / r_b, y_b / r_b    
    dphi_dn = (grads[:, 0:1] * nx + grads[:, 1:2] * ny) / R
    loss_inner = torch.mean(dphi_dn**2)
    phi_out = model(outer_bc_norm)
    loss_outer = torch.mean((phi_out - outer_bc_phi)**2)
    loss_total = lam_pde * loss_pde + lam_inner * loss_inner + lam_outer * loss_outer
    return loss_total, loss_pde, loss_inner, loss_outer

# Fase di addestramento (training)
def training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    model = PINN(hidden_layers=hidden_layers, neurons=neurons).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history = {
        "total": [],
        "pde": [],
        "inner": [],
        "outer": []
    }

    print("Fase 1: Adam")
    for epoch in range(1, epochs + 1):
        collocation = sample_points(N_f, R, R_max).to(device)
        inner_bc = circle_boundary(N_inner_bc, R).to(device)

        outer_bc_norm, outer_bc_phi = outer_boundary(N_outer_bc, R_max, R, U)
        outer_bc_norm, outer_bc_phi = outer_bc_norm.to(device), outer_bc_phi.to(device)

        loss, loss_pde, loss_inner, loss_outer = loss_fn(
            model, collocation, inner_bc, outer_bc_norm, outer_bc_phi,
            R, U, w_pde, w_inner, w_outer
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history["total"].append(loss.item())
        loss_history["pde"].append(loss_pde.item())
        loss_history["inner"].append(loss_inner.item())
        loss_history["outer"].append(loss_outer.item())

        if epoch % 200 == 0:
            print(f"Adam Epoca {epoch:4d}: Loss = {loss.item():.6e}")

    print("\nFase 2: L-BFGS")
    lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=lr_lbfgs,
        max_iter=max_iter_lbfgs,
        history_size=history_size_lbfgs
    )

    collocation_f = sample_points(N_f, R, R_max).to(device)
    inner_bc_f = circle_boundary(N_inner_bc, R).to(device)
    outer_bc_norm_f, outer_bc_phi_f = outer_boundary(N_outer_bc, R_max, R, U)
    outer_bc_norm_f, outer_bc_phi_f = outer_bc_norm_f.to(device), outer_bc_phi_f.to(device)

    def closure():
        lbfgs.zero_grad()
        loss, _, _, _ = loss_fn(
            model, collocation_f, inner_bc_f,
            outer_bc_norm_f, outer_bc_phi_f,
            R, U, w_pde, w_inner, w_outer
        )
        loss.backward()
        return loss

    for ep in range(1, epochs_lbfgs + 1):
        lbfgs.step(closure)

        loss, loss_pde, loss_inner, loss_outer = loss_fn(
            model, collocation_f, inner_bc_f,
            outer_bc_norm_f, outer_bc_phi_f,
            R, U, w_pde, w_inner, w_outer
        )

        loss_history["total"].append(loss.item())
        loss_history["pde"].append(loss_pde.item())
        loss_history["inner"].append(loss_inner.item())
        loss_history["outer"].append(loss_outer.item())

        if ep % 10 == 0:
            print(f"L-BFGS Epoca {ep:3d}: Loss = {loss.item():.6e}")

    print(f"\nL-BFGS completato | Loss finale: {loss_history['total'][-1]:.6e}")

    return model, device, loss_history

# Valutazione della velocità
def evaluate_velocity(model, R, Rmax, N, device):
    x = torch.linspace(-Rmax, Rmax, N)
    y = torch.linspace(-Rmax, Rmax, N)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    pts_phys = torch.cat([X.reshape(-1, 1), Y.reshape(-1, 1)], dim=1)  
    r = torch.sqrt(pts_phys[:, 0:1]**2 + pts_phys[:, 1:2]**2)
    mask = (r.squeeze() >= R) & (r.squeeze() <= Rmax)
    pts_phys = pts_phys[mask].to(device)
    pts_norm = (pts_phys / R).requires_grad_(True)    
    phi_pred = model(pts_norm)
    grads_pred = torch.autograd.grad(
        phi_pred, pts_norm, grad_outputs=torch.ones_like(phi_pred), create_graph=False)[0]
    u_pred, v_pred = grads_pred[:, 0:1] / R, grads_pred[:, 1:2] / R
    pts_phys_grad = pts_phys.clone().requires_grad_(True)
    phi_true = phi_analytical(pts_phys_grad[:, 0:1], pts_phys_grad[:, 1:2], U=U, R=R)
    grads_true = torch.autograd.grad(
        phi_true, pts_phys_grad, grad_outputs=torch.ones_like(phi_true), create_graph=False)[0]
    u_true, v_true = grads_true[:, 0:1], grads_true[:, 1:2]
    l2_u = torch.norm(u_pred - u_true) / torch.norm(u_true)
    l2_v = torch.norm(v_pred - v_true) / torch.norm(v_true)   
    print(f"\nErrore relativo L2_u: {l2_u.item():.6e}")
    print(f"Errore relativo L2_v: {l2_v.item():.6e}")   
    return (u_pred.detach(), v_pred.detach(),
            u_true.detach(), v_true.detach(),
            X, Y, mask, l2_u.item(), l2_v.item())

# Plot dei risultati
def plot_velocity_results(u_pred, v_pred, u_true, v_true, X, Y, mask,
                          loss_history, l2_u, l2_v, R, R_max, N):
    def to_grid(values, mask, N):
        grid = torch.full((N * N,), float('nan'))
        grid[mask] = values.squeeze().cpu()
        return grid.reshape(N, N).numpy()

    u_pinn_grid = to_grid(u_pred, mask, N)
    v_pinn_grid = to_grid(v_pred, mask, N)
    u_true_grid = to_grid(u_true, mask, N)
    v_true_grid = to_grid(v_true, mask, N)
    err_u_grid = to_grid((u_pred - u_true).abs(), mask, N)
    err_v_grid = to_grid((v_pred - v_true).abs(), mask, N)

    x_ax = torch.linspace(-R_max, R_max, N).numpy()
    y_ax = torch.linspace(-R_max, R_max, N).numpy()

    fig = plt.figure(figsize=(16, 16.6))
    gs = fig.add_gridspec(
        3, 3,
        height_ratios=[2.0, 1.3, 1.3],
        hspace=0.5,
        wspace=0.3
    )

    ax_loss = fig.add_subplot(gs[0, :])
    ax_loss.plot(loss_history["total"], label="Loss totale", color="black", linewidth=2)
    ax_loss.plot(loss_history["pde"], label="PDE", color="tab:blue")
    ax_loss.plot(loss_history["inner"], label="BC interna", color="tab:orange")
    ax_loss.plot(loss_history["outer"], label="BC esterna", color="tab:green")
    ax_loss.legend()
    ax_loss.set_yscale("log")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, alpha=0.5)

    fig.suptitle(
        f"Campo di velocità attorno a un cilindro\n"
        f"L2 relativo: u = {l2_u:.4f}, v = {l2_v:.4f}",
        fontsize=13,
        fontweight="bold",
        y=0.92
    )

    rows = [
        ("u", u_pinn_grid, u_true_grid, err_u_grid),
        ("v", v_pinn_grid, v_true_grid, err_v_grid),
    ]

    titles = ["Soluzione Analitica", "Soluzione PINN", "Errore assoluto"]
    cmaps = ["RdBu_r", "RdBu_r", "hot_r"]

    for row_idx, (comp, pinn_grid, true_grid, err_grid) in enumerate(rows, start=1):
        vmin = np.nanmin(true_grid)
        vmax = np.nanmax(true_grid)
        grids = [true_grid, pinn_grid, err_grid]

        for i, (title, grid, cmap) in enumerate(zip(titles, grids, cmaps)):
            ax = fig.add_subplot(gs[row_idx, i])

            if i < 2:
                im = ax.pcolormesh(
                    x_ax, y_ax, grid.T,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    shading="auto"
                )
            else:
                im = ax.pcolormesh(
                    x_ax, y_ax, grid.T,
                    cmap=cmap,
                    shading="auto"
                )

            circle = plt.Circle((0, 0), R, color="white", zorder=5)
            border = plt.Circle((0, 0), R, color="black", fill=False, linewidth=1.5, zorder=6)
            ax.add_patch(circle)
            ax.add_patch(border)

            ax.set_title(f"{title} ({comp})", fontsize=11)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_aspect("equal")
            ax.set_xlim(-R_max, R_max)
            ax.set_ylim(-R_max, R_max)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.show()

# Creazione del modello, addestramento, valutazione e plot finale
model, device, loss_history = training()

u_pred, v_pred, u_true, v_true, X, Y, mask, l2_u, l2_v = evaluate_velocity(
    model, R, R_max, N_test, device)

plot_velocity_results(
    u_pred, v_pred, u_true, v_true, X, Y, mask,
    loss_history, l2_u, l2_v, R, R_max, N_test)
