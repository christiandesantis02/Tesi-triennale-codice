import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(15)

h = 1.0
U = 1.0

N_f = 100

class PINN(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )

    def forward(self, y):
        return self.net(y)

model = PINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def pde_residual(y):
    u = model(y)
    du = torch.autograd.grad(
        u, y,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]
    d2u = torch.autograd.grad(
        du, y,
        grad_outputs=torch.ones_like(du),
        create_graph=True
    )[0]
    return d2u

loss_history = []

for epoch in range(4000):
    optimizer.zero_grad()

    y_f = torch.rand(N_f, 1).to(device)
    y_f.requires_grad_(True)

    res = pde_residual(y_f)
    loss_pde = torch.mean(res**2)

    y0 = torch.tensor([[0.0]], requires_grad=True).to(device)
    y1 = torch.tensor([[h]], requires_grad=True).to(device)

    u0 = model(y0)
    u1 = model(y1)

    loss_bc = (u0 - 0.0)**2 + (u1 - U)**2

    loss = loss_pde + 5 * loss_bc
    loss.backward()

    optimizer.step()

    loss_history.append([
        loss.item(),
        loss_pde.item(),
        loss_bc.item()
    ])

    if epoch % 500 == 0:
        print(f"Epoca {epoch}: Loss = {loss.item():.6e}")

# =========================
# Estrazione delle loss
# =========================
loss_total = [x[0] for x in loss_history]

# =========================
# Calcolo soluzione PINN
# =========================

y_test = torch.linspace(0, h, 100).view(-1, 1).to(device)

u_pred = model(y_test).detach().cpu().numpy()
y_test_np = y_test.cpu().numpy()

u_exact = U * y_test_np / h

error = np.linalg.norm(u_pred - u_exact, 2) / np.linalg.norm(u_exact, 2)


# =========================
# Plot 1: Loss
# =========================

epochs = range(1, len(loss_history) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, loss_total, label="Loss totale", linewidth=0.5)

plt.xlabel("Epoca")
plt.ylabel("Loss")
plt.title("Andamento della Loss")
plt.yscale("log")
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================
# Plot 2: Soluzione
# =========================

plt.figure(figsize=(8, 5))
plt.plot(y_test_np, u_exact, label="Analitica", linewidth=2)
plt.plot(y_test_np, u_pred, '--', label="PINN", linewidth=2)

plt.xlabel("y")
plt.ylabel("u")
plt.title(f"Campo di velocità \n Errore L2 relativo: {error:.5f}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
