import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size):
        super(ResidualBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
        for m in self.layer:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.layer(x) + x


class ResNetPINN(nn.Module):
    def __init__(self, device, input_size=2, hidden_size=20, output_size=1, num_blocks=4):
        super(ResNetPINN, self).__init__()

        self.input_layer = nn.Linear(input_size, hidden_size)

        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_size) for _ in range(num_blocks)
        ])

        self.output_layer = nn.Linear(hidden_size, output_size)
        
        self.activation = nn.Tanh()
        
        nn.init.xavier_normal_(self.input_layer.weight)
        nn.init.xavier_normal_(self.output_layer.weight)
        
        self.lambda_1 = nn.Parameter(torch.tensor([0.0], device=device))
        self.lambda_2 = nn.Parameter(torch.tensor([-6.0], device=device))

        print("Using ResNetPINN")

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        
        out = self.activation(self.input_layer(inputs))
        
        for block in self.res_blocks:
            out = block(out)
            
        out = self.output_layer(out)
        return out

    def physics_loss(self, x, t):
        x.requires_grad = True
        t.requires_grad = True
        
        u = self.forward(x, t)
        
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        
        f = u_t + self.lambda_1 * u * u_x - torch.exp(self.lambda_2) * u_xx
        return f
    