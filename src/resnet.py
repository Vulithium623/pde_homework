import torch
import torch.nn as nn
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import requests
import os
import time
import warnings

np.random.seed(42)
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on {device}")
warnings.filterwarnings("ignore", message=".*Attempting to run cuBLAS.*")

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

# === 新增：残差块定义 ===
class ResidualBlock(nn.Module):
    def __init__(self, hidden_size):
        super(ResidualBlock, self).__init__()
        # 残差路径：Linear -> Tanh -> Linear -> Tanh
        self.layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
        # 权重初始化
        for m in self.layer:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # 核心：输出 = 变换后的x + 原始x
        return self.layer(x) + x

# === 修改：ResNet-PINN 主模型 ===
class ResNetPINN(nn.Module):
    def __init__(self, input_size=2, hidden_size=20, output_size=1, num_blocks=4):
        super(ResNetPINN, self).__init__()
        
        # 1. 输入映射层
        self.input_layer = nn.Linear(input_size, hidden_size)
        
        # 2. 堆叠残差块 (num_blocks=4 相当于深度增加)
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_size) for _ in range(num_blocks)
        ])
        
        # 3. 输出层
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        self.activation = nn.Tanh()
        
        # 初始化首尾层
        nn.init.xavier_normal_(self.input_layer.weight)
        nn.init.xavier_normal_(self.output_layer.weight)
        
        # === 物理参数 (保持之前的成功设置) ===
        # lambda_1 初始化为 0.0
        self.lambda_1 = nn.Parameter(torch.tensor([0.0], device=device))
        # lambda_2 初始化为 -6.0 (exp(-6) approx 0.0025)
        self.lambda_2 = nn.Parameter(torch.tensor([-6.0], device=device))

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        
        # 先做一次线性映射 + 激活
        out = self.activation(self.input_layer(inputs))
        
        # 通过所有残差块
        for block in self.res_blocks:
            out = block(out)
            
        # 输出
        out = self.output_layer(out)
        return out

    def physics_loss(self, x, t):
        x.requires_grad = True
        t.requires_grad = True
        
        u = self.forward(x, t)
        
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        
        # 保持 exp 以确保稳定性
        f = u_t + self.lambda_1 * u * u_x - torch.exp(self.lambda_2) * u_xx
        return f

if __name__ == "__main__":
    if not os.path.exists('./data'): os.makedirs('./data')
    if not os.path.exists('./result/resnet'): os.makedirs('./result/resnet')
    
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
    
    model = ResNetPINN(input_size=2, hidden_size=20, output_size=1, num_blocks=4).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    loss_history = []
    lambda_1_history = []
    lambda_2_history = []
    
    print("\nAdam Training...")
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
            
    print("\nL-BFGS Fine-tuning...")
    
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
    

    x_star_tensor = torch.tensor(X_star[:, 0:1], dtype=torch.float32, device=device)
    t_star_tensor = torch.tensor(X_star[:, 1:2], dtype=torch.float32, device=device)
    
    model.eval()
    with torch.no_grad():
        u_pred_star = model(x_star_tensor, t_star_tensor)
        u_pred_star = u_pred_star.cpu().numpy()
    
    U_pred = u_pred_star.reshape(X.shape)
    
    error_u = np.linalg.norm(Exact - U_pred, 2) / np.linalg.norm(Exact, 2)
    print(f"Error u (L2 Relative): {error_u * 100:.4f}%")

    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 3, 1)
    plt.pcolormesh(T, X, Exact, shading='auto', cmap='jet')
    plt.title('Ground Truth u(x,t)')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.colorbar()
    
    plt.subplot(1, 3, 2)
    plt.pcolormesh(T, X, U_pred, shading='auto', cmap='jet')
    plt.title(f'ResNet Prediction (Err: {error_u*100:.2f}%)')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.colorbar()
    
    plt.subplot(1, 3, 3)
    plt.pcolormesh(T, X, np.abs(Exact - U_pred), shading='auto', cmap='jet')
    plt.title('Absolute Error |u_true - u_pred|')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.colorbar()
    
    plt.tight_layout()
    save_path_flow = './result/resnet/flow_field_resnet.png'
    plt.savefig(save_path_flow, dpi=150)

    true_l1 = 1.0
    true_l2 = 0.01 / np.pi
    
    err_l1 = abs(lambda_1_history[-1] - true_l1) / true_l1 * 100
    err_l2 = abs(lambda_2_history[-1] - true_l2) / true_l2 * 100

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(lambda_1_history, label='Predicted $\lambda_1$', linewidth=2)
    plt.axhline(true_l1, color='r', linestyle='--', label='True $\lambda_1$', linewidth=2)
    plt.title(f'$\lambda_1$ Convergence (Final Err: {err_l1:.3f}%)')
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(lambda_2_history, label='Predicted $\lambda_2$', linewidth=2)
    plt.axhline(true_l2, color='r', linestyle='--', label='True $\lambda_2$', linewidth=2)
    plt.title(f'$\lambda_2$ Convergence (Final Err: {err_l2:.3f}%)')
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path_param = './result/resnet/parameters_resnet.png'
    plt.savefig(save_path_param, dpi=150)
