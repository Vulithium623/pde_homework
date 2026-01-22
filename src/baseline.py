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
N_train = 2000
epochs = 10000


def download_data_cn(file_name):
    if os.path.exists(file_name):
        print(f"Data {file_name} exists")
        return

    url = "https://github.com/maziarraissi/PINNs/raw/master/appendix/Data/burgers_shock.mat"
    
    print(f"Downloading: {file_name} ...")
    try:
        r = requests.get(url, timeout=30)
        with open(file_name, 'wb') as f:
            f.write(r.content)
        print("Done!")
    except Exception as e:
        print(f"Download failed {e}")
        exit()


class PhysicsInformedNN(nn.Module):
    def __init__(self):
        super(PhysicsInformedNN, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(2, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 1)
        )
        
        self.lambda_1 = nn.Parameter(torch.tensor([0.0], device=device))
        self.lambda_2 = nn.Parameter(torch.tensor([0.0], device=device))

    def forward(self, x, t):
        input_data = torch.cat([x, t], dim=1)
        u = self.net(input_data)
        return u

    def physics_loss(self, x, t):
        x.requires_grad = True
        t.requires_grad = True
        
        u = self.forward(x, t)
        
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        
        f = u_t + self.lambda_1 * u * u_x - torch.exp(self.lambda_2) * u_xx 
        
        f_val = u_t + self.lambda_1 * u * u_x - self.lambda_2 * u_xx
        return f_val


if __name__ == "__main__":
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
    
    model = PhysicsInformedNN().to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    loss_history = []
    lambda_1_history = []
    lambda_2_history = []
    
    print("\nStart Training ... (Goal: lambda_1 -> 1.0, lambda_2 -> 0.00318)")
    
    start_time = time.time()
    
    for epoch in range(epochs):
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
        l2 = model.lambda_2.item()
        lambda_1_history.append(l1)
        lambda_2_history.append(l2)
        
        if epoch % 1000 == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item():.5f}, "
                  f"L1: {l1:.4f}, L2: {l2:.5f}")

    elapsed = time.time() - start_time
    print(f"\nTime: {elapsed:.2f}s")
    
    true_l1 = 1.0
    true_l2 = 0.01 / np.pi
    error_l1 = abs(true_l1 - lambda_1_history[-1]) / true_l1 * 100
    error_l2 = abs(true_l2 - lambda_2_history[-1]) / true_l2 * 100
    
    print("\n--- Final Results ---")
    print(f"Lambda_1 predict: {lambda_1_history[-1]:.5f} (ground truth: {true_l1}) -> err: {error_l1:.2f}%")
    print(f"Lambda_2 predict: {lambda_2_history[-1]:.5f} (ground truth: {true_l2:.5f}) -> err: {error_l2:.2f}%")
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(lambda_1_history, label='Predicted $\lambda_1$')
    plt.axhline(true_l1, color='r', linestyle='--', label='True $\lambda_1$')
    plt.title('Convergence of $\lambda_1$')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(lambda_2_history, label='Predicted $\lambda_2$')
    plt.axhline(true_l2, color='r', linestyle='--', label='True $\lambda_2$')
    plt.title('Convergence of $\lambda_2$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./result/baseline/parameters_baseline.png')

    x_star_tensor = torch.tensor(X_star[:, 0:1], dtype=torch.float32, device=device)
    t_star_tensor = torch.tensor(X_star[:, 1:2], dtype=torch.float32, device=device)
    
    with torch.no_grad():
        u_pred_star = model(x_star_tensor, t_star_tensor).cpu().numpy()
    
    U_pred = u_pred_star.reshape(X.shape)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.pcolormesh(T, X, Exact, shading='auto', cmap='jet')
    plt.title('Ground Truth u(x,t)')
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.pcolormesh(T, X, U_pred, shading='auto', cmap='jet')
    plt.title('PINN Prediction u(x,t)')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('./result/baseline/flow_field_baseline.png')
