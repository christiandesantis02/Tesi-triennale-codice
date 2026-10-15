import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Definizione parametri
R = 1.0
n_mode = 2
omega = float(np.sqrt(n_mode * (n_mode**2 - 1)))
epsilon = 0.05
T = 2 * np.pi / omega
hidden_layers = 6
neurons = 64
epochs = 3000
epochs_lbfgs = 500
N_f = 4000
N_bc = 1000
N_ic = 500
N_test = 250
w_pde = 1.0
w_kin = 50.0
w_dyn = 50.0
w_ic = 10.0
SEED = 0              
ALPHA = 0.95         
UPDATE_EVERY = 100   

# Struttura rete neurale
class PINN(nn.Module):
    def __init__(self, hidden_layers, neurons):
        super().__init__()
        layers = []
        layers.append(nn.Linear(3, neurons))
        layers.append(nn.Tanh())
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(neurons, neurons))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(neurons, 1))
        self.net = nn.Sequential(*layers)
        linears = [m for m in self.net if isinstance(m, nn.Linear)]
        for lin in linears[:-1]:
            nn.init.xavier_normal_(lin.weight, gain=1.0)
            nn.init.zeros_(lin.bias)
        nn.init.xavier_normal_(linears[-1].weight, gain=0.1)
        nn.init.zeros_(linears[-1].bias)

    def forward(self, x):
        t_norm = 2.0 * x[:, 2:3] / T - 1.0
        x = torch.cat([x[:, 0:2], t_norm], dim=1)
        return self.net(x)

# Calcolo del laplaciano per il residuo pde
def laplacian(phi, x):
    grads = torch.autograd.grad(phi, x,
        grad_outputs=torch.ones_like(phi),
        create_graph=True)[0]
    phi_x = grads[:, 0:1]
    phi_y = grads[:, 1:2]
    phi_xx = torch.autograd.grad(phi_x, x,
        grad_outputs=torch.ones_like(phi_x),
        create_graph=True)[0][:, 0:1]
    phi_yy = torch.autograd.grad(phi_y, x,
        grad_outputs=torch.ones_like(phi_y),
        create_graph=True)[0][:, 1:2]
    return phi_xx + phi_yy

# Campionamento collocation points
def sample_points(n, R, T):
    r = R * torch.sqrt(torch.rand(n, 1))
    th = torch.rand(n, 1) * 2 * torch.pi
    x = r * torch.cos(th)
    y = r * torch.sin(th)
    t = torch.rand(n, 1) * T
    return torch.cat([x, y, t], dim=1)

def circle_boundary(n, R, T):
    theta = torch.linspace(0, 2 * torch.pi, n + 1)[:-1].reshape(-1, 1)
    x = R * torch.cos(theta)
    y = R * torch.sin(theta)
    t = torch.rand(n, 1) * T
    return torch.cat([x, y, t], dim=1)

def initial_condition(n, R):
    r = R * torch.sqrt(torch.rand(n, 1))
    th = torch.rand(n, 1) * 2 * torch.pi
    x = r * torch.cos(th)
    y = r * torch.sin(th)
    t = torch.zeros(n, 1)
    return torch.cat([x, y, t], dim=1)

# Soluzione analitica per confronto
def phi_analytical(x, y, t, epsilon=0.05, omega=1.0, n_mode=2):
    r = torch.sqrt(x**2 + y**2) + 1e-8
    theta = torch.atan2(y, x)
    return -(epsilon * omega / n_mode) * r**n_mode * torch.cos(n_mode * theta) * torch.sin(omega * t)

# Struttura della funzione di perdita
def loss_fn(model, collocation, bc, ic, R, epsilon, omega, lam_pde, lam_kin, lam_dyn, lam_ic):
    collocation.requires_grad_(True)
    phi = model(collocation)
    lap = laplacian(phi, collocation)
    loss_pde = torch.mean(lap**2)
    bc.requires_grad_(True)
    phi_b = model(bc)
    grads = torch.autograd.grad(
        phi_b, bc,
        grad_outputs=torch.ones_like(phi_b),
        create_graph=True)[0]
    x_b, y_b = bc[:, 0:1], bc[:, 1:2]
    t_b = bc[:, 2:3]
    r_b = torch.sqrt(x_b**2 + y_b**2) + 1e-8
    nx, ny = x_b / r_b, y_b / r_b
    dphi_dn = grads[:, 0:1] * nx + grads[:, 1:2] * ny
    dphi_dt = grads[:, 2:3]
    theta_b = torch.atan2(y_b, x_b)
    eta_dot = -epsilon * omega * torch.cos(n_mode * theta_b) * torch.sin(omega * t_b)
    loss_kin = torch.mean((dphi_dn - eta_dot)**2)
    eta_b = epsilon * torch.cos(n_mode * theta_b) * torch.cos(omega * t_b)
    loss_dyn = torch.mean((dphi_dt + (n_mode**2 - 1) * eta_b)**2)
    phi_ic = model(ic)
    loss_ic = torch.mean(phi_ic**2)
    loss_total = lam_pde * loss_pde + lam_kin * loss_kin + lam_dyn * loss_dyn + lam_ic * loss_ic
    return loss_total, loss_pde, loss_kin, loss_dyn, loss_ic

# Inverse Dirichlet Weighting: aggiornamento dei pesi della loss
def update_weights(epoch, losses, model, lam, alpha=ALPHA, every=UPDATE_EVERY):   
    if epoch % every != 0:                      
        return lam
    params = list(model.parameters())    
    stds = torch.stack([torch.cat([g.flatten() for g in torch.autograd.grad(
        l, params, retain_graph=True)]).std() for l in losses])
    
    new = stds.max() / (stds + 1e-8)
    return (alpha * lam + (1 - alpha) * new).detach()

# Addestramento del modello
def training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    model = PINN(hidden_layers=hidden_layers, neurons=neurons).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
    collocation = sample_points(N_f, R, T).to(device)
    bc = circle_boundary(N_bc, R, T).to(device)
    ic = initial_condition(N_ic, R).to(device)
    lam = torch.tensor([w_pde, w_kin, w_dyn, w_ic], device=device)   
    # Fase 1: Adam
    print(f"\nFase 1: Adam ({epochs} epoche)")
    for epoch in range(epochs):
        loss, loss_pde, loss_kin, loss_dyn, loss_ic = loss_fn(model, collocation, bc, ic, R, epsilon, omega, *lam)
        # IDW: per usare pesi fissi basta commentare la riga seguente
        lam = update_weights(epoch, [loss_pde, loss_kin, loss_dyn, loss_ic], model, lam)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch + 1:4d}: Loss = {loss.item():.3e}  pde={loss_pde.item():.3e}  kin={loss_kin.item():.3e}  dyn={loss_dyn.item():.3e}  ic={loss_ic.item():.3e}  lam={[round(v, 2) for v in lam.tolist()]}")
    # Fase 2: L-BFGS (pesi fissi ai valori raggiunti al termine di Adam)
    lbfgs = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=5, max_eval=None, history_size=50,
        tolerance_grad=1e-9, tolerance_change=1e-11, line_search_fn='strong_wolfe')
    def closure():
        loss, _, _, _, _ = loss_fn(model, collocation, bc, ic, R, epsilon, omega, *lam)
        lbfgs.zero_grad()
        loss.backward()
        return loss
    print(f"\nFase 2: L-BFGS ({epochs_lbfgs} epoche)")
    for ep in range(1, epochs_lbfgs + 1):
        lbfgs.step(closure)
        loss, loss_pde, loss_kin, loss_dyn, loss_ic = loss_fn(model, collocation, bc, ic, R, epsilon, omega, *lam)
        if ep % 100 == 0:
            print(f"L-BFGS epoch {ep:4d}: Loss = {loss.item():.3e}  pde={loss_pde.item():.3e}  kin={loss_kin.item():.3e}  dyn={loss_dyn.item():.3e}  ic={loss_ic.item():.3e}")
    print(f"\nL-BFGS completato, Loss finale: {loss.item():.3e}")
    return model, device

# Valutazione del modello
def evaluate_velocity(model, device, R, N, t_val):
    x = torch.linspace(-R, R, N)
    y = torch.linspace(-R, R, N)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    pts = torch.cat([X.reshape(-1, 1), Y.reshape(-1, 1)], dim=1)
    r = torch.sqrt(pts[:, 0:1]**2 + pts[:, 1:2]**2)
    mask = r.squeeze() <= R
    pts = pts[mask]
    pts = torch.cat([pts, torch.full((pts.shape[0], 1), t_val)], dim=1).to(device)
    pts.requires_grad_(True)
    phi_pred = model(pts)
    grads_pred = torch.autograd.grad(
        phi_pred, pts,
        grad_outputs=torch.ones_like(phi_pred),
        create_graph=False)[0]
    u_pred, v_pred = grads_pred[:, 0:1], grads_pred[:, 1:2]
    pts_true = pts.detach().clone().requires_grad_(True)
    phi_true = phi_analytical(pts_true[:, 0:1], pts_true[:, 1:2], pts_true[:, 2:3],
                              epsilon=epsilon, omega=omega, n_mode=n_mode)
    grads_true = torch.autograd.grad(
        phi_true, pts_true,
        grad_outputs=torch.ones_like(phi_true),
        create_graph=False)[0]
    u_true, v_true = grads_true[:, 0:1], grads_true[:, 1:2]
    l2_phi = torch.norm(phi_pred - phi_true) / torch.norm(phi_true)
    l2_u = torch.sqrt(torch.sum((u_pred - u_true)**2 + (v_pred - v_true)**2)
                      / torch.sum(u_true**2 + v_true**2))
    return (phi_pred.detach(), phi_true.detach(),
            u_pred.detach(), v_pred.detach(),
            u_true.detach(), v_true.detach(),
            mask, l2_phi.item(), l2_u.item())

def print_errors(results):
    print("\nErrori L2 relativi")
    for label, (l2_phi, l2_u) in results.items():
        print(f"  {label:5s}  phi={l2_phi:.3e}  |u|={l2_u:.3e}")
    phi_mean = np.mean([e[0] for e in results.values()])
    u_mean = np.mean([e[1] for e in results.values()])
    print(f"  media  phi={phi_mean:.3e}  |u|={u_mean:.3e}")

# Plot dei risultati
def plot_results(phi_pred, phi_true, u_pred, v_pred, u_true, v_true, mask, R, N, t_label):
    def to_grid(values, mask, N):
        grid = torch.full((N * N,), float('nan'))
        grid[mask] = values.squeeze().cpu()
        return grid.reshape(N, N).numpy()
    phi_pinn_grid = to_grid(phi_pred, mask, N)
    phi_true_grid = to_grid(phi_true, mask, N)
    err_phi_grid = to_grid((phi_pred - phi_true).abs(), mask, N)
    u_pinn_grid = to_grid(u_pred, mask, N)
    v_pinn_grid = to_grid(v_pred, mask, N)
    u_true_grid = to_grid(u_true, mask, N)
    v_true_grid = to_grid(v_true, mask, N)
    mod_pinn_grid = np.hypot(u_pinn_grid, v_pinn_grid)
    mod_true_grid = np.hypot(u_true_grid, v_true_grid)
    err_mod_grid = np.hypot(u_pinn_grid - u_true_grid, v_pinn_grid - v_true_grid)
    x_ax = torch.linspace(-R, R, N).numpy()
    y_ax = torch.linspace(-R, R, N).numpy()
    X, Y = np.meshgrid(x_ax, y_ax, indexing="ij")
    th = np.linspace(0, 2 * np.pi, 300)
    cx, cy = R * np.cos(th), R * np.sin(th)
    fig, axs = plt.subplots(2, 3, figsize=(17, 10))
    vmin, vmax = np.nanmin(phi_true_grid), np.nanmax(phi_true_grid)
    for ax, data, title in zip(axs[0], (phi_true_grid, phi_pinn_grid), ('Soluzione analitica (φ)', 'Soluzione PINN (φ)')):
        im = ax.contourf(X, Y, data, levels=150, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        ax.contour(X, Y, data, levels=10, colors='k', linewidths=0.3, alpha=0.3)
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
    im = axs[0, 2].contourf(X, Y, err_phi_grid, levels=150, cmap='hot_r')
    axs[0, 2].set_title('Errore assoluto (|Δφ|)')
    fig.colorbar(im, ax=axs[0, 2], format='%.1e')
    vm = np.nanmax(mod_true_grid)
    s = 12
    fields = [(mod_true_grid, u_true_grid, v_true_grid, 'Soluzione analitica (|u|)'),
              (mod_pinn_grid, u_pinn_grid, v_pinn_grid, 'Soluzione PINN (|u|)')]
    for ax, (mod, u, v, title) in zip(axs[1], fields):
        im = ax.contourf(X, Y, mod, levels=100, cmap='viridis', vmin=0, vmax=vm)
        ax.quiver(X[::s, ::s], Y[::s, ::s], u[::s, ::s], v[::s, ::s],
                  color='w', scale=vm * 25, width=0.004)
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
    im = axs[1, 2].contourf(X, Y, err_mod_grid, levels=100, cmap='hot_r')
    axs[1, 2].set_title('Errore assoluto (|Δu|)')
    fig.colorbar(im, ax=axs[1, 2], format='%.1e')
    for ax in axs.ravel():
        ax.plot(cx, cy, 'k-', lw=1.5)
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    fig.suptitle(f"$t={t_label}$", fontsize=13)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.35, top=0.92)

# Creazione del modello, addestramento, valutazione e plot finale
model, device = training()
istanti = {'T/4': T / 4, '3T/4': 3 * T / 4}
risultati = {lab: evaluate_velocity(model, device, R, N_test, t) for lab, t in istanti.items()}
print_errors({lab: (r[7], r[8]) for lab, r in risultati.items()})
for lab, r in risultati.items():
    plot_results(*r[:7], R, N_test, lab)
plt.show()
