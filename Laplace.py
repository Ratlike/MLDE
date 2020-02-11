# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 22:52:23 2020

This script uses a neural network to solve the partial differential equation

Nabla^2 f(x, y) = 0

f(0, y) = f(pi, y) = sin(y),
f(x, 0) = f(x, pi) = sin(x),

0 <= y <= pi, 0 <= x <= pi.

The following calculation currently does not converge. Or maybe with would
with enough time, but I'm not patient enough. The calculation has to be moved
onto the GPU, or I have to choose a problem with fewer poles, as I think they
make convergence difficult.'

@author: jurik
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad
import matplotlib.pyplot as plt
import numpy as np

# Create training data points in interval 0 < x < pi
train = torch.rand(100, 2)
train = np.pi*train
# Split the training data into batches and shuffle them. Shuffling does
# nothing in this instance (since data is random) but might be useful later.
trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)

bc = np.pi*np.linspace(0,1,25)
bnd1 = np.column_stack((bc, np.zeros_like(bc)))
bnd2 = np.column_stack((bc, np.pi*np.ones_like(bc)))
bnd = np.concatenate((bnd1, bnd2, np.flip(bnd1, 1), np.flip(bnd2, 1)))
bnd = torch.from_numpy(bnd).float()
boundaryset = torch.utils.data.DataLoader(bnd, batch_size=10, shuffle=True)

#     _   __     __                      __  
#    / | / /__  / /__      ______  _____/ /__
#   /  |/ / _ \/ __/ | /| / / __ \/ ___/ //_/
#  / /|  /  __/ /_ | |/ |/ / /_/ / /  / ,<   
# /_/ |_/\___/\__/ |__/|__/\____/_/  /_/|_|  
#                                            

# Define the neural network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Size of the network
        self.hidden_layers = 2
        self.width = 32
        
        # Input layer
        self.input = nn.Linear(2, self.width)
        
        # Hidden layers
        self.hidden = nn.ModuleList()      
        for i in range(self.hidden_layers):
            self.hidden.append(nn.Linear(self.width, self.width))
        
        # Output layer
        self.output = nn.Linear(self.width, 1)
        
    def forward(self, x):
        x = F.tanh(self.input(x))
        
        for layer in self.hidden:
            x = F.tanh(layer(x))
        
        x = self.output(x)
        
        return x

#     ______                 __  _                 
#    / ____/_  ______  _____/ /_(_)___  ____  _____
#   / /_  / / / / __ \/ ___/ __/ / __ \/ __ \/ ___/
#  / __/ / /_/ / / / / /__/ /_/ / /_/ / / / (__  ) 
# /_/    \__,_/_/ /_/\___/\__/_/\____/_/ /_/____/  
#                                                  

def LHS(f, X):
    # Calculate the left-hand side of the differential equation
    # Nabla^2 f(X=[x, y]) = 0.

    dX = grad(f, X, create_graph=True, grad_outputs=torch.ones_like(f))[0]
    dx = dX[:, 0]
    dy = dX[:, 1]
    
    dxdx = grad(dx, X, create_graph=True, grad_outputs=torch.ones_like(dx))[0]
    dydy = grad(dy, X, create_graph=True, grad_outputs=torch.ones_like(dy))[0]
    
    return dxdx[:, 0] + dydy[:, 1]
    
def Loss(data, bnd):
    # Calculate loss
    
    # Internal points where loss is calculated from the diff. equation
    X = data.clone().detach().requires_grad_(True)
    f = net(X)
    lhs = LHS(f, X)

    loss_fn = nn.MSELoss()
    loss = loss_fn(lhs, torch.zeros_like(lhs))

    # Boundary conditions
    X = bnd.clone().detach().requires_grad_(True)
    bc = net(X) - torch.sum(torch.sin(X), 1)
    bc_loss = loss_fn(bc, torch.zeros_like(bc))

    # Combine loss functions from internal and boundary points
    return loss + bc_loss


#   ______           _       _            
#  /_  __/________ _(_)___  (_)___  ____ _
#   / / / ___/ __ `/ / __ \/ / __ \/ __ `/
#  / / / /  / /_/ / / / / / / / / / /_/ / 
# /_/ /_/   \__,_/_/_/ /_/_/_/ /_/\__, /  
#                                /____/   

net = Net()

optimizer = optim.Adam(net.parameters(), lr=1e-3)

n_epoch = 5000 # Number of training epochs

# Train network
for epoch in range(1, n_epoch + 1):
    for data, bnd in zip(trainset, boundaryset):
        
        net.zero_grad()
        
        loss = Loss(data, bnd)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
    print("Epoch:", epoch, "/", n_epoch, "Loss:", loss.detach().numpy())

#     ____                  ____      
#    / __ \___  _______  __/ / /______
#   / /_/ / _ \/ ___/ / / / / __/ ___/
#  / _, _/  __(__  ) /_/ / / /_(__  ) 
# /_/ |_|\___/____/\__,_/_/\__/____/  
#                                    

# Generate a mesh of plotting points
x_vals = np.linspace(0, np.pi, 100)
y_vals = np.linspace(0, np.pi, 100)
X, Y = np.meshgrid(x_vals, y_vals)

# Plot figures
Z = np.sinh(np.pi - X)*np.sin(Y) + np.sinh(X)*np.sin(Y) + \
    np.sin(X)*np.sinh(np.pi - Y) + np.sin(X)*np.sinh(Y)
Z = Z/np.sinh(np.pi)

fig, axes = plt.subplots(nrows=1, ncols=2,dpi=1200)
lvl = np.linspace(min(Z.flatten()), max(Z.flatten()), 10)
cp = axes[0].contourf(X, Y, Z, levels=lvl)
axes[0].set_aspect('equal')
fig.colorbar(cp, ax=axes[0], shrink=0.4)
axes[0].set_title('Solution')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')

C = np.column_stack((np.ndarray.flatten(X), np.ndarray.flatten(Y)))
C = torch.from_numpy(C).float()
Z = net(C.view(-1, 2))
Z = Z.detach().numpy()
Z = np.reshape(Z, (100,100))

lvl = np.linspace(min(Z.flatten()), max(Z.flatten()), 10)
cp = axes[1].contourf(X, Y, Z, levels=lvl)
axes[1].set_aspect('equal')
fig.colorbar(cp, ax=axes[1], shrink=0.4)
axes[1].set_title('NN')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')

fig.tight_layout()