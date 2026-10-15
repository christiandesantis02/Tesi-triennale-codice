import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Definizione parametri
h = 1.0 
U = 1.0 
neurons = 20 
hidden_layers = 3 
epochs = 4000
lr = 1e-3
N_f = 100 
seed = 15
w_pde = 1.0
w_bc = 5.0
N_test = 100 

# Struttura rete neurale
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Linear(1, neurons), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(neurons, neurons), nn.Tanh()])
        layers.append(nn.Linear(neurons, 1))
        self.net = nn.Sequential(*layers)
        
    def forward(self, y):
        return self.net(y)

# Struttura del residuo per la pde
def pde_residual(y, model):
    u = model(y)
    du = torch.autograd.grad(
        u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    d2u = torch.autograd.grad(
        du, y, grad_outputs=torch.ones_like(du), create_graph=True)[0]
    return d2u

# Addestramento del modello
def training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    model = PINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  
    y0 = torch.tensor([[0.0]], device=device)
    y1 = torch.tensor([[h]], device=device)  
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        y_f = (torch.rand(N_f, 1, device=device) * h).requires_grad_(True)
        res = pde_residual(y_f, model)
        loss_pde = torch.mean(res**2)
        u0 = model(y0)
        u1 = model(y1)
        loss_bc = torch.mean((u0 - 0.0)**2) + torch.mean((u1 - U)**2)        
        loss = w_pde * loss_pde + w_bc * loss_bc     
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0 or epoch == 1:
            print(f"Epoca {epoch:4d}: Loss = {loss.item():.6e}")
    return model, device

# Valutazione del modello
def evaluate(model, device, N_test):
    model.eval()
    with torch.no_grad():
        y_test = torch.linspace(0, h, N_test, device=device).view(-1, 1)
        u_pred = model(y_test).cpu().numpy()
        y_test_np = y_test.cpu().numpy()
        
    u_exact = U * y_test_np / h
    error = np.linalg.norm(u_pred - u_exact, 2) / np.linalg.norm(u_exact, 2)
    return y_test_np, u_pred, u_exact, error

# Plot dei risultati
def plot_results(y_test_np, u_pred, u_exact, error):
    plt.figure(figsize=(6, 4))
    plt.plot(y_test_np, u_exact, label="Analitica", linewidth=2)
    plt.plot(y_test_np, u_pred, '--', label="PINN", linewidth=2)
    plt.xlabel("y")
    plt.ylabel("u")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title(f"Flusso di Couette\nErrore L2 relativo: {error:.4e}")
    plt.show()

# Creazione del modello, training, valutazione e plot finale
model, device = training()
y_test_np, u_pred, u_exact, error = evaluate(model, device, N_test)
plot_results(y_test_np, u_pred, u_exact, error)
