# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 17:58:00 2020

This script uses a neural network to solve the differential equation

f''(x) + f(x) = 0,

with boundary conditions

f(0) = 0,
f(pi/2) = 1,

in the interval 0 < x < pi. The analytical solution to this problem is 
f(x) = sin(x).

@author: Juri Kuorelahti
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad
import matplotlib.pyplot as plt
import numpy as np

# Create training data points in interval 0 < x < pi
train = torch.rand(1000, 1)
train = np.pi*train
# Split the training data into batches and shuffle them. Shuffling does
# nothing in this instance (since data is random) but might be useful later.
trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)

# Define the neural network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Size of the network
        self.hidden_layers = 2
        self.width = 32
        
        # Input layer
        self.input = nn.Linear(1, self.width)
        
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
        
net = Net()
print(net)

optimizer = optim.Adam(net.parameters(), lr=1e-4)

n_epoch = 30 # Number of training epochs

# Train network
for epoch in range(1, n_epoch + 1):
    for data in trainset:
        
        x = data.clone().detach().requires_grad_(True)
        
        net.zero_grad()
        f = net(x.view(-1, 1))
        
        # Calculate the differential equation
        # f''(x) + f(x) = 0:
        dx = grad(f, x, create_graph=True, grad_outputs=torch.ones_like(x))[0]
        dxdx = grad(dx, x, create_graph=True, grad_outputs=torch.ones_like(x))[0]
        diffeq = dxdx + f
        
        # Calculate loss
        loss_fn = nn.MSELoss()
        #loss_fn = nn.MSELoss(reduction='none')
        loss = loss_fn(diffeq, torch.zeros_like(diffeq))
        
        # Boundary conditions
        # f(0) = 0:
        x = torch.Tensor([0])
        bc1 = net(x)
        bc1_loss = loss_fn(bc1, torch.Tensor([0]))
        # f(pi/2) = 1:
        x = torch.Tensor([np.pi/2])
        bc2 = net(x)
        bc2_loss = loss_fn(bc2, torch.Tensor([1]))
        
        # Combine loss functions
        loss = loss + (bc1_loss + bc2_loss)/2
        #loss = torch.cat((loss, bc1_loss.view(-1, 1), bc2_loss.view(-1, 1)), 0)
        #loss = torch.mean(loss)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
    print("Epoch:", epoch, "/", n_epoch, "Loss:", loss.detach().numpy())

# Plot the numerical solution together with the analytical solution
x = torch.linspace(0, np.pi, 100)
y = net(x.view(-1, 1))
y = y.detach().numpy()
x = x.detach().numpy()
plt.figure(dpi=500)
plt.plot(x, np.sin(x), 'c', x, y, 'b--')

# Calculate maximum deviation from the analytical solution
print("Maximum deviation:", max(np.abs(y.flatten() - np.sin(x))))