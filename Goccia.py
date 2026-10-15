import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(15)
# parametri fisici
h = 1.0
U = 1.0

# numero punti collocation
N_f = 100

# rete neurale
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

    def forward(self, y):   # questo metodo definisce il passaggio in avanti dei dati attraverso la NN.
        return self.net(y)   # in pratica prende y, un tensore di dati in ingresso, e calcola l'output predetto dalla rete

model = PINN().to(device)  # crea un istanza della classe PINN, è l'oggetto "rete neurale" (.to(device) sposta l'oggetto dalla memoria cpu a quella impostata, cuda se presente)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # crea un istanza della classe Adam, contenuta nel modulo torch.optim; è l'oggetto "ottimizzatore",
                                                          # che contiene metodi come optimizer.step() o optimizer.zero_grad(), che verranno usati in seguito

# funzione residuo PDE
def pde_residual(y):
    u = model(y) # passa l'input y alla rete neurale, ottenendo la predizione della rete u
    du = torch.autograd.grad( # calcola la derivata di u rispetto ad y  
                                
        u, y,  # stabilisce la variabile da derivare secondo quale altra ( u rispetto ad y)
        grad_outputs=torch.ones_like(u),  # è un tecnicismo vettoriale di PyTorch necessario per avviare il calcolo del gradiente con la stessa forma di u.
        create_graph=True # dice a PyTorch di memorizzare la struttura di questa derivata, perché in seguito dovremo derivarla una seconda volta.
                           # ( in pratica non distrugge il grafo computazionale creato alla fine del calcolo
    )[0] # [0] serve perchè il risultato restituisce una tupla, e ne prendiamo solo il primo elemento che contiene il valore effettivo della derivata
              # (La funzione torch.autograd.grad è progettata per poter calcolare i gradienti rispetto a molti input contemporaneamente, non solo uno)
    d2u = torch.autograd.grad( # stesso procedimento di calcolo per la derivata prima
        du, y,
        grad_outputs=torch.ones_like(du),
        create_graph=True
    )[0]
    return d2u

# training
for epoch in range(5000):
    optimizer.zero_grad()

    # ricampionamento punti interni
    y_f = torch.rand(N_f, 1).to(device) * h   #  A ogni singola epoca, la rete genera N_f punti casuali compresi tra 0 e h.
    y_f.requires_grad_(True)   # tiene traccia delle operazioni eseguite su questi punti

    # residuo PDE
    res = pde_residual(y_f)   # calcola il residuo invocando la funzione pde_residual con input y_f (di fatto gli arriva d2u)
    loss_pde = torch.mean(res**2)   # calcola il MSE, cioè l'errore quadratico medio

    # condizioni al contorno
    y0 = torch.tensor([[0.0]], requires_grad=True).to(device)  # crea una matrice 1x1 con il valore al contorno scelto (i dati sono scritti da prassi come tensori)
    y1 = torch.tensor([[h]], requires_grad=True).to(device)    
    u0 = model(y0) # calcola il valore predetto dalla rete nel punto al contorno
    u1 = model(y1)
    loss_bc = (u0 - 0.0)**2 + (u1 - U)**2   # costruisce il termine di loss al contorno

    loss = loss_pde + loss_bc
    loss.backward() # avvia la backpropagation: ripercorre all'indietro l'intero grafo computazionale, partendo dal valore numerico finale della loss
                    # fino ad arrivare ai singoli pesi e bias interni del modello, dopodichè calcola il gradiente per ogni parametro della rete.
                    # Il gradiente è un vettore matematico che indica in quale direzione e di quanto bisogna modificare ciascun peso per far diminuire l'errore.
                    #  I gradienti calcolati vengono salvati temporaneamente dentro una memoria interna a ogni parametro del modello (nell'attributo .grad di ciascun peso).
                    
    optimizer.step() # applica l'algoritmo matematico scelto(qui Adam) per aggiornare e modificare i parametri del modello (pesi e bias) sulla base dei gradienti appena calcolati
                      # i gradienti di questa epoca verranno poi cancellati in quella successiva all'inizio con il comando optimizer.zero_grad()
    if epoch % 500 == 0:
        print(f"Epoca {epoch}: Loss = {loss.item():.6e}")

# valutazione
y_test = torch.linspace(0, h, 100).view(-1, 1).to(device) # crea un vettore 1d di 100 punti, poi con .view lo converte in matrice da 100 righe e 1 colonna,
                                                         # infine sposta tutto sulla gpu
u_pred = model(y_test).detach().cpu().numpy() # da in pasto i 100 punti creati alla pinn per ottenere u, poi con .detach scollega il tensore risultante dal grafo computazionale di pytorch
                                              # .cpu serve a spostare il tensore dalla gpu alla cpu, infine .numpy a convertirlo in un array numpy
y_test_np = y_test.cpu().numpy() # prende le 100 coordinate spaziali create prima, le sposta sulla cpu e le converte in un array numpy
                                 # (servirà come asse delle ascisse nel grafico)

# soluzione analitica
u_exact = U * y_test_np / h   # calcola la soluzione esatta direttamente sui punti della griglia di test
                               # (poichè y_test_np è un array numpy, la formula è applicata elemento per elemento e il risultato, u_exact, è di nuovo un array numpy da 100 elementi)
                            
# errore L2 relativo
error = np.linalg.norm(u_pred - u_exact, 2) / np.linalg.norm(u_exact, 2) # calcola una metrica standard quantitativa per misurare l'errore globale del modelli PINN.
                                                                         # La funzione np.linalg.norm calcola la norma di un vettore, con il 2 che indica la norma euclidea
                                                                         # (essendo un vettore l'argomento, corrisponde alla somma dei quadrati di tutti i suoi elementi)
                                                                         # il numeratore calcola quindi la differenza tra modello e soluzione esatta e somma questi termini al quadrato
                                                                         # ottenendo l'errore quadratico medio non mediato sui 100 punti della griglia.
                                                                         # Infine lo divide per la norma assoluta della soluzione esatta.
                                                                         # In questo modo si ottiene l'errore L2 relativo, indipendente dalla scala delle variabili.

# grafico soluzione
plt.figure(figsize=(6,4)) # crea una nuova finestra per il grafico (figssize imposta 6 pollici di larghezza e 4 di lunghezza)
plt.plot(y_test_np, u_exact, label="Analitica", linewidth=2) # disegna la soluzione esatta, con y_test_np come punti delle ascisse e u_exact come ordinate.
plt.plot(y_test_np, u_pred, '--', label="PINN", linewidth=2) # disegna la soluzione predetta dalla PINN, con linea tratteggiata (di default è continua)
plt.xlabel("y") # inserisce l'etichetta all'asse x
plt.ylabel("u") # inserisce l'etichetta sull'asse y
plt.legend() # mostra la legenda nel grafico (contenente Analitica, PINN e le relative curve, se tratteggiate o continue)
plt.title(f"Flusso di Couette\nL2 relativo: {error:.4e}") # inserisce il titolo principale sopra al grafico, usando la formattazione con f-string.
                                              # ( :.4e fa visualizzare il risultato in notazione scientifica con 4 cifre decimali)
plt.show() # comando finale che renderizza e mostra il grafico, bloccando l'esecuzione del codice finchè l'immagine non viene chiusa
