# seed = 0, IDW (alpha = 0.95)

import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# ── Configurazione ────────────────────────────────────────────────────────────
N_MODE = 2
OMEGA = float(np.sqrt(N_MODE * (N_MODE**2 - 1)))
EPSILON = 0.05
T_END = 2 * np.pi / OMEGA

HIDDEN, LAYERS = 64, 6
N_ADAM, N_LBFGS = 5000, 500
N_INT, N_BC, N_IC = 4000, 1000, 500
NG = 250
seed = 0

torch.manual_seed(seed)
np.random.seed(seed)


# ── Modello ───────────────────────────────────────────────────────────────────
class PINN(nn.Module):

  def __init__(self, hidden, layers):
    super().__init__()
    sizes = [3] + [hidden] * layers + [1]
    self.linears = nn.ModuleList(
        [nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
    )
    for lin in self.linears[:-1]:
      nn.init.xavier_normal_(lin.weight, gain=1.0)
      nn.init.zeros_(lin.bias)
    nn.init.xavier_normal_(self.linears[-1].weight, gain=0.1)
    nn.init.zeros_(self.linears[-1].bias)

  def forward(self, x, y, t):
    tn = 2.0 * t / T_END - 1.0
    u = torch.stack([x, y, tn], dim=-1)
    for lin in self.linears[:-1]:
      u = torch.tanh(lin(u))
    return self.linears[-1](u).squeeze(-1)


KEYS = ('pde', 'kin', 'dyn', 'ic')  # ordine fisso dei termini
LAMBDA0 = [1.0, 3.0, 8.0, 3.0]  # pesi di partenza
UPDATE_EVERY = 100
ALPHA = 0.95


def update_weights(terms, model, lam, alpha=ALPHA):
  params = list(model.parameters())
  stds = []
  for k in KEYS:
    grads = torch.autograd.grad(
        terms[k],
        params,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    flat = torch.cat([
        (torch.zeros_like(p) if g is None else g).flatten()
        for g, p in zip(grads, params)
    ])
    stds.append(flat.std())
  stds = torch.stack(stds)
  new = stds.max() / (stds + 1e-8)
  return (alpha * lam + (1 - alpha) * new).detach()


# ── Soluzione analitica ───────────────────────────────────────────────────────
def phi_exact_np(x, y, t):
  r = np.sqrt(x**2 + y**2)
  theta = np.arctan2(y, x)
  return (
      -(EPSILON * OMEGA / N_MODE)
      * r**N_MODE
      * np.cos(N_MODE * theta)
      * np.sin(OMEGA * t)
  )


def vel_exact_np(x, y, t):
  """u,v analitici: gradiente di phi."""
  z = (x + 1j * y) ** (N_MODE - 1)
  A = EPSILON * OMEGA * np.sin(OMEGA * t)
  return -A * z.real, A * z.imag


# ── Punti di collocazione ─────────────────────────────────────────────────────
def to_tensors(x, y, t):
  return tuple(
      torch.tensor(a, dtype=torch.float32, device=DEVICE, requires_grad=True)
      for a in (x, y, t)
  )


def make_interior(n):
  r = np.sqrt(np.random.rand(n))
  th = np.random.rand(n) * 2 * np.pi
  return to_tensors(
      r * np.cos(th), r * np.sin(th), np.random.rand(n) * T_END
  )


def make_boundary(n):
  th = np.linspace(0, 2 * np.pi, n, endpoint=False)
  return to_tensors(
      np.cos(th), np.sin(th), np.random.rand(n) * T_END
  )


def make_ic(n):
  r = np.sqrt(np.random.rand(n))
  th = np.random.rand(n) * 2 * np.pi
  return to_tensors(r * np.cos(th), r * np.sin(th), np.zeros(n))


def make_points():
  return dict(
      int=make_interior(N_INT), bc=make_boundary(N_BC), ic=make_ic(N_IC)
  )


# ── Loss ──────────────────────────────────────────────────────────────────────
def grad(f, x):
  return torch.autograd.grad(f.sum(), x, create_graph=True)[0]


def compute_loss(model, pts, lam):
  xi, yi, ti = pts['int']
  xb, yb, tb = pts['bc']
  xic, yic, tic = pts['ic']

  # PDE: Laplaciano = 0
  phi_i = model(xi, yi, ti)
  lap = grad(grad(phi_i, xi), xi) + grad(grad(phi_i, yi), yi)
  loss_pde = (lap**2).mean()

  # Bordo: condizione cinematica e dinamica
  phi_b = model(xb, yb, tb)
  dphi_dr = xb * grad(phi_b, xb) + yb * grad(phi_b, yb)
  theta_b = torch.atan2(yb, xb)

  eta_dot = (
      -EPSILON * OMEGA * torch.cos(N_MODE * theta_b) * torch.sin(OMEGA * tb)
  )
  loss_kin = ((dphi_dr - eta_dot) ** 2).mean()

  eta_b = EPSILON * torch.cos(N_MODE * theta_b) * torch.cos(OMEGA * tb)
  loss_dyn = ((grad(phi_b, tb) + (N_MODE**2 - 1) * eta_b) ** 2).mean()

  # Condizione iniziale
  loss_ic = (model(xic, yic, tic) ** 2).mean()

  terms = dict(pde=loss_pde, kin=loss_kin, dyn=loss_dyn, ic=loss_ic)
  total = sum(lam[i] * terms[k] for i, k in enumerate(KEYS))
  return total, terms


def log_history(history, total, terms):
  history['total'].append(total.item())
  for k, v in terms.items():
    history[k].append(v.item())


def fmt_losses(total, terms):
  return f'total={total.item():.3e}  ' + '  '.join(
      f'{k}={v.item():.3e}' for k, v in terms.items()
  )


# ── Training ──────────────────────────────────────────────────────────────────
def train_adam(model, pts, history):
  opt = torch.optim.Adam(model.parameters(), lr=1e-3)
  sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1000, gamma=0.5)
  lam = torch.tensor(LAMBDA0, dtype=torch.float32, device=DEVICE)
  history['lam'] = []
  print(f'\nFase 1: Adam ({N_ADAM} step)')

  for i in range(N_ADAM):
    opt.zero_grad()
    total, terms = compute_loss(model, pts, lam)

    if i % UPDATE_EVERY == 0:
      lam = update_weights(terms, model, lam)
      total = sum(
          lam[j] * terms[k] for j, k in enumerate(KEYS)
      )  # ricalcolo col peso nuovo

    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    log_history(history, total, terms)
    history['lam'].append(lam.cpu().numpy().copy())
    if (i + 1) % 500 == 0:
      print(f'  step {i+1:5d}  {fmt_losses(total, terms)}')
      print(f'            lam = {lam.cpu().numpy().round(2)}')
  return lam


def train_lbfgs(model, pts, history, lam):
  opt = torch.optim.LBFGS(
      model.parameters(),
      lr=1.0,
      max_iter=5,
      max_eval=None,
      history_size=50,
      tolerance_grad=1e-9,
      tolerance_change=1e-11,
      line_search_fn='strong_wolfe',
  )

  def closure():
    opt.zero_grad()
    total, _ = compute_loss(model, pts, lam)
    total.backward()
    return total

  print(f'\nFase 2: L-BFGS ({N_LBFGS} epoche)')
  for i in range(N_LBFGS):
    opt.step(closure)
    total, terms = compute_loss(
        model, pts, lam
    )  # valutazione separata per la history
    log_history(history, total, terms)
    if (i + 1) % 100 == 0:
      print(f'  epoca {i+1:5d}  {fmt_losses(total, terms)}')


def train(model, pts):
  history = {k: [] for k in ('total',) + KEYS}
  lam = train_adam(model, pts, history)
  # train_lbfgs(model, pts, history, lam)
  return history, lam


# ── Valutazione ───────────────────────────────────────────────────────────────
def make_grid():
  x1d = np.linspace(-1, 1, NG)
  Xg, Yg = np.meshgrid(x1d, x1d)
  return Xg, Yg, np.sqrt(Xg**2 + Yg**2) <= 1.0


def rel_err(pn, ex):
  return np.sqrt(np.nansum((pn - ex) ** 2)) / np.sqrt(np.nansum(ex**2))


def evaluate(model, grid, t_val):
  """Restituisce dict con phi, u, v (exact, pinn) e modulo velocità."""
  Xg, Yg, inside = grid
  mask = lambda a: np.where(inside, a, np.nan)

  phi_ex = mask(phi_exact_np(Xg, Yg, t_val))
  u_ex, v_ex = (mask(a) for a in vel_exact_np(Xg, Yg, t_val))

  xf = torch.tensor(
      Xg[inside], dtype=torch.float32, device=DEVICE, requires_grad=True
  )
  yf = torch.tensor(
      Yg[inside], dtype=torch.float32, device=DEVICE, requires_grad=True
  )
  tf = torch.full_like(xf, t_val)  # eredita già il device di xf
  phi = model(xf, yf, tf)
  u_p = (
      torch.autograd.grad(phi.sum(), xf, retain_graph=True)[0]
      .detach()
      .cpu()
      .numpy()
  )
  v_p = torch.autograd.grad(phi.sum(), yf)[0].detach().cpu().numpy()

  def to_grid(a):
    g = np.full_like(Xg, np.nan)
    g[inside] = a
    return g

  phi_pn, u_pn, v_pn = (
      to_grid(phi.detach().cpu().numpy()),
      to_grid(u_p),
      to_grid(v_p),
  )

  return dict(
      phi=(phi_ex, phi_pn, np.abs(phi_pn - phi_ex)),
      u=(u_ex, u_pn),
      v=(v_ex, v_pn),
      mod=(
          np.hypot(u_ex, v_ex),
          np.hypot(u_pn, v_pn),
          np.hypot(u_pn - u_ex, v_pn - v_ex),
      ),
      err=dict(
          phi=rel_err(phi_pn, phi_ex),
          u=np.sqrt(
              np.nansum((u_pn - u_ex) ** 2 + (v_pn - v_ex) ** 2)
              / np.nansum(u_ex**2 + v_ex**2)
          ),
      ),
  )


def print_errors(results):
  print('\nErrori L2 relativi')
  for label, ev in results.items():
    e = ev['err']
    print(f"  {label:5s}  phi={e['phi']:.3e}  |u|={e['u']:.3e}")
  phi_mean = np.mean([ev['err']['phi'] for ev in results.values()])
  u_mean = np.mean([ev['err']['u'] for ev in results.values()])
  print(f'  media  phi={phi_mean:.3e}  |u|={u_mean:.3e}')


# ── Plot ──────────────────────────────────────────────────────────────────────


def plot_loss(history, n_adam=N_ADAM):
  fig, ax = plt.subplots(figsize=(9, 5))
  steps = np.arange(len(history['total']))

  stili = [
      ('total', 'tab:blue', '-', 'Total', 0.8),
      ('pde', 'tab:orange', '--', 'PDE', 1.0),
      ('kin', 'tab:red', '--', 'BC cinematica', 1.0),
      ('dyn', 'tab:green', '--', 'BC dinamica', 1.0),
      ('ic', 'tab:purple', '--', 'IC', 1.0),
  ]
  for key, color, ls, label, lw in stili:
    ax.semilogy(
        steps, history[key], color=color, linestyle=ls, lw=lw, label=label
    )

  ax.set_xlabel('Iterazione')
  ax.set_ylabel('Loss')
  ax.set_title('Convergenza loss')
  ax.legend(fontsize=8)
  ax.grid(True, alpha=0.5)
  fig.tight_layout()
  return fig


def plot_istante(grid, ev, t_label):
  Xg, Yg, _ = grid
  x1d = Xg[0]
  th = np.linspace(0, 2 * np.pi, 300)
  cx, cy = np.cos(th), np.sin(th)

  fig, axs = plt.subplots(2, 3, figsize=(17, 10))

  # Riga 0: potenziale
  phi_ex, phi_pn, phi_er = ev['phi']
  vmin, vmax = np.nanmin(phi_ex), np.nanmax(phi_ex)
  for ax, data, title in zip(
      axs[0],
      (phi_ex, phi_pn),
      ('Soluzione analitica (φ)', 'Soluzione PINN (φ)'),
  ):
    im = ax.contourf(
        Xg, Yg, data, levels=150, cmap='RdBu_r', vmin=vmin, vmax=vmax
    )
    ax.contour(Xg, Yg, data, levels=10, colors='k', linewidths=0.3, alpha=0.3)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
  im = axs[0, 2].contourf(
      Xg,
      Yg,
      phi_er,
      levels=150,
      cmap='hot_r',
      vmin=0,
      vmax=1.3 * np.nanmax(phi_er),
  )
  axs[0, 2].set_title('Errore assoluto (|Δφ|)')
  fig.colorbar(im, ax=axs[0, 2], format='%.1e')

  # Riga 1: velocità (modulo + frecce)
  mod_ex, mod_pn, mod_er = ev['mod']
  vm = np.nanmax(mod_ex)
  s = 12
  for ax, mod, (u, v), title in zip(
      axs[1],
      (mod_ex, mod_pn),
      (ev['u'][0:1] + ev['v'][0:1], ev['u'][1:2] + ev['v'][1:2]),
      ('Soluzione analitica (|u|)', 'Soluzione PINN (|u|)'),
  ):
    im = ax.contourf(Xg, Yg, mod, levels=100, cmap='viridis', vmin=0, vmax=vm)
    ax.quiver(
        Xg[::s, ::s],
        Yg[::s, ::s],
        u[::s, ::s],
        v[::s, ::s],
        color='w',
        scale=vm * 25,
        width=0.004,
    )
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
  im = axs[1, 2].contourf(
      Xg,
      Yg,
      mod_er,
      levels=100,
      cmap='hot_r',
      vmin=0,
      vmax=1.3 * np.nanmax(mod_er),
  )
  axs[1, 2].set_title('Errore assoluto (|Δu|)')
  fig.colorbar(im, ax=axs[1, 2], format='%.1e')

  for ax in axs.ravel():
    ax.plot(cx, cy, 'k-', lw=1.5)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

  e = ev['err']
  fig.suptitle(f'$t={t_label}$', fontsize=13)
  fig.tight_layout()
  fig.subplots_adjust(hspace=0.35, top=0.92)
  return fig


# ── Main ──────────────────────────────────────────────────────────────────────

print(f'omega = {OMEGA:.4f},  T = {T_END:.4f}')
pts = make_points()
model = PINN(HIDDEN, LAYERS).to(DEVICE)
history, lam = train(model, pts)
plot_loss(history)

grid = make_grid()
istanti = {'T/4': T_END / 4, '3T/4': 3 * T_END / 4}
results = {lab: evaluate(model, grid, t) for lab, t in istanti.items()}

print_errors(results)
for lab, ev in results.items():
  plot_istante(grid, ev, lab)
plt.show()
