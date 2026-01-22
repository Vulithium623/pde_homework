import torch
import torch.nn as nn
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import requests
import os
import time

np.random.seed(42)
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on {device}")

N_train = 2000
epochs_adam = 10000
layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1] 

def download_data_cn(file_name):
    if os.path.exists(file_name):
        return
    url = "https://mirror.ghproxy.com/https://github.com/maziarraissi/PINNs/raw/master/appendix/Data/burgers_shock.mat"
    print(f"Downloading {file_name}...")
    try:
        r = requests.get(url, timeout=30)
        with open(file_name, 'wb') as f:
            f.write(r.content)
    except:
        print("Download failed.")
        exit()

class PhysicsInformedNN(nn.Module):
    def __init__(self, layers):
        super(PhysicsInformedNN, self).__init__()
        
        self.depth = len(layers) - 1
        self.activation = nn.Tanh()
        
        layer_list = []
        for i in range(self.depth):
            layer_list.append(nn.Linear(layers[i], layers[i+1]))
            nn.init.xavier_normal_(layer_list[-1].weight)
            nn.init.zeros_(layer_list[-1].bias)
        
        self.layers_list = nn.ModuleList(layer_list)
        
        self.lambda_1 = nn.Parameter(torch.tensor([0.0], device=device))
        
        self.lambda_2 = nn.Parameter(torch.tensor([-6.0], device=device))

    def forward(self, x, t):
        a = torch.cat([x, t], dim=1)
        for i in range(self.depth - 1):
            a = self.layers_list[i](a)
            a = self.activation(a)
        a = self.layers_list[-1](a)
        return a

    def physics_loss(self, x, t):
        x.requires_grad = True
        t.requires_grad = True
        
        u = self.forward(x, t)
        
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        
        f = u_t + self.lambda_1 * u * u_x - torch.exp(self.lambda_2) * u_xx
        
        return f

if __name__ == "__main__":
    if not os.path.exists('./data'): os.makedirs('./data')
    if not os.path.exists('./result/final'): os.makedirs('./result/final')
    
    file_name = "./data/burgers_shock.mat"
    download_data_cn(file_name)
    data = scipy.io.loadmat(file_name)
    
    t = data['t'].flatten()[:,None]
    x = data['x'].flatten()[:,None]
    Exact = np.real(data['usol']).T
    X, T = np.meshgrid(x, t)
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    u_star = Exact.flatten()[:,None]
    
    idx = np.random.choice(X_star.shape[0], N_train, replace=False)
    X_train = X_star[idx, :]
    u_train = u_star[idx, :]
    
    x_train_tensor = torch.tensor(X_train[:, 0:1], dtype=torch.float32, device=device)
    t_train_tensor = torch.tensor(X_train[:, 1:2], dtype=torch.float32, device=device)
    u_train_tensor = torch.tensor(u_train, dtype=torch.float32, device=device)
    
    model = PhysicsInformedNN(layers).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    loss_history = []
    lambda_1_history = []
    lambda_2_history = []
    
    print("\nStage 1: Adam Training (10000 epochs)...")
    start_time = time.time()
    
    for epoch in range(epochs_adam):
        optimizer.zero_grad()
        
        u_pred = model(x_train_tensor, t_train_tensor)
        loss_u = torch.mean((u_pred - u_train_tensor) ** 2)
        
        f_pred = model.physics_loss(x_train_tensor, t_train_tensor)
        loss_f = torch.mean(f_pred ** 2)
        
        loss = loss_u + loss_f
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        l1 = model.lambda_1.item()
        l2 = torch.exp(model.lambda_2).item()
        lambda_1_history.append(l1)
        lambda_2_history.append(l2)
        
        if epoch % 1000 == 0:
            print(f"Iter: {epoch}, Loss: {loss.item():.5f}, L1: {l1:.4f}, L2: {l2:.5f}")
            
    print("\n🔥 Stage 2: L-BFGS Fine-tuning (The Magic Step)...")
    
    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(), 
        lr=1.0, 
        history_size=50,
        max_iter=50000, 
        max_eval=50000,
        tolerance_grad=1e-5,
        tolerance_change=1.0 * np.finfo(float).eps,
        line_search_fn="strong_wolfe"
    )
    
    def closure():
        optimizer_lbfgs.zero_grad()
        u_pred = model(x_train_tensor, t_train_tensor)
        loss_u = torch.mean((u_pred - u_train_tensor) ** 2)
        f_pred = model.physics_loss(x_train_tensor, t_train_tensor)
        loss_f = torch.mean(f_pred ** 2)
        loss = loss_u + loss_f
        loss.backward()
        
        loss_history.append(loss.item())
        l1 = model.lambda_1.item()
        l2 = torch.exp(model.lambda_2).item()
        lambda_1_history.append(l1)
        lambda_2_history.append(l2)
        return loss

    optimizer_lbfgs.step(closure)
    
    elapsed = time.time() - start_time
    print(f"Training time: {elapsed:.2f}s")
    
    true_l1 = 1.0
    true_l2 = 0.01 / np.pi
    
    pred_l1 = lambda_1_history[-1]
    pred_l2 = lambda_2_history[-1]
    
    err_l1 = abs(pred_l1 - true_l1) / true_l1 * 100
    err_l2 = abs(pred_l2 - true_l2) / true_l2 * 100
    
    print(f"\nFinal Results:")
    print(f"Lambda 1: {pred_l1:.5f} (True: {true_l1}) -> Error: {err_l1:.4f}%")
    print(f"Lambda 2: {pred_l2:.5f} (True: {true_l2:.5f}) -> Error: {err_l2:.4f}%")
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1,2,1)
    plt.plot(lambda_1_history)
    plt.axhline(true_l1, color='r', linestyle='--')
    plt.title(f'Lambda 1 Convergence (Err: {err_l1:.2f}%)')
    
    plt.subplot(1,2,2)
    plt.plot(lambda_2_history)
    plt.axhline(true_l2, color='r', linestyle='--')
    plt.title(f'Lambda 2 Convergence (Err: {err_l2:.2f}%)')
    plt.savefig('./result/baseline/parameters.png')