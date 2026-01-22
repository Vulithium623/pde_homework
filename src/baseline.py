import torch
import torch.nn as nn


class PhysicsInformedNN(nn.Module):
    def __init__(self, device, layers=[2, 20, 20, 20, 20, 20, 20, 20, 20, 1]):
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

        print("Using Baseline")

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
