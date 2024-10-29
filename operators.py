import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import qmc
import numpy as np
import time
import timeit
import itertools


optimizers = itertools.cycle([optim.Adam, optim.RMSprop, optim.SGD])

lrs = itertools.cycle([.001, .01, .005])
iterations = itertools.cycle([700, 1000, 1300])

# Define the neural network with 2000 hidden neurons, 10 input dims, and 1 output dim
class ANeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation):
        super(ANeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu2':
            self.activation = ReLU2()
        self.fc2 = nn.Linear(hidden_size, output_size, bias=False)
        
        # Initialize all weights and biases to zero
        nn.init.zeros_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


# Custom ReLU^2 activation function
class ReLU2(nn.Module):
    def forward(self, x):
        return torch.relu(x) ** 2


# Four criteria of fit

def diff_l2_loss0(candidate, data):
    candidate_L = data['Lnn'](candidate, data['i_batch'], data['activation'])
    candidate_B = data['Bnn'](candidate, data['b_batch'], data['N_facet_b'], data['activation'])
    h = torch.cat([candidate_L, candidate_B])
    
    return torch.square(torch.linalg.norm(data['residual_b'] - h))/2


def diff_l2_loss1(candidate, data):
    candidate_L = data['Lnn'](candidate, data['i_batch'], data['activation'])
    candidate_B = data['Bnn'](candidate, data['b_batch'], data['N_facet_b'], data['activation'])
    h = torch.cat([candidate_L, candidate_B])
    scaling = 1 / torch.linalg.norm(h)
    
    return torch.square(torch.linalg.norm(data['residual_b'] - scaling * h))/2


def diff_l2_loss2(candidate, data):
    candidate_L = data['Lnn'](candidate, data['i_batch'], data['activation'])
    candidate_B = data['Bnn'](candidate, data['b_batch'], data['N_facet_b'], data['activation'])
    h = torch.cat([candidate_L, candidate_B])
    scaling = torch.dot(data['residual_b'], h)/torch.dot(h, h)
    
    return torch.square(torch.linalg.norm(data['residual_b'] - scaling * h))/2


def quot_l2_loss(candidate, data):
    candidate_L = data['Lnn'](candidate, data['i_batch'], data['activation'])
    candidate_B = data['Bnn'](candidate, data['b_batch'], data['N_facet_b'], data['activation'])
    h = torch.cat([candidate_L, candidate_B])

    numerator = -torch.abs(torch.dot(data['residual_b'], h))
    denominator = torch.linalg.norm(h)

    return numerator / denominator





def my_laplace_relu2(model, x):
    w = torch.sum(model.fc1.weight**2, axis=1)
    sd = 2 * (model.fc1(x) > 0)
    a = sd * w
    laplacian = model.fc2(a)
    return laplacian.squeeze()


def my_laplace_tanh(model, x):
    # Faster than using autograd.
    w = torch.sum(model.fc1.weight**2, axis=1)
    y = model.activation(model.fc1(x))
    sd = -2 * y * (1 - y**2)
    a = sd * w
    laplacian = model.fc2(a)
    return laplacian.squeeze()

def gradient(model, x, activation):
    var1 = model.fc1(x)
    if activation == 'relu2':
        y = 2 * var1 * (var1 > 0)
    elif activation == 'tanh':
        y = 1 - torch.square(torch.tanh(var1))

    c = model.fc2.weight#.squeeze().unsqueeze(0)
    y = y * c
    y = y @ model.fc1.weight

    return y


def Lerror(u_L, model_L, N_interior):
    dif_L = torch.sum((u_L - model_L)**2) / N_interior
    dif_norm = torch.sqrt(dif_L)
    c = torch.sum(u_L**2) / N_interior
    u_norm = torch.sqrt(c)
    if u_norm == 0:
        u_norm = 1
    return dif_norm / u_norm


def Berror(u_B, model_B, N_facet):
    dif_B = torch.sum((u_B - model_B)**2) / N_facet
    dif_norm = torch.sqrt(dif_B)
    c = torch.sum(u_B**2) / N_facet
    u_norm = torch.sqrt(c)
    return dif_norm / u_norm


def energy_error(u_L, u_B, model_L, model_B, N_interior, N_facet, u_norm=None):
    dif_L = torch.sum((u_L - model_L)**2) / N_interior
    dif_B = torch.sum((u_B - model_B)**2) / N_facet
    dif_norm = torch.sqrt(dif_L + dif_B)
    if u_norm == None:
        L_part = torch.sum(u_L**2) / N_interior
        B_part = torch.sum(u_B**2) / N_facet
        u_norm = torch.sqrt(L_part + B_part)
    return dif_norm / u_norm


def energy_inner_product(v_L, v_B, w_L, w_B, N_interior, N_facet):
    LvLw = torch.sum(v_L * w_L) / N_interior

    BvBw = torch.sum(v_B * w_B) / N_facet

    return LvLw + BvBw


def enorm(v_L, v_B, N_interior, N_facet):
    enorm_squared = energy_inner_product(v_L, v_B, v_L, v_B, N_interior, N_facet)
    return torch.sqrt(enorm_squared)


def L2norm(v_vals, N):
    output = torch.sqrt(torch.sum(torch.square(v_vals)) / N)
    return output

def H1norm(v_vals, v_grads, N):
    output1 = torch.sum(torch.square(v_vals)) / N
    output2 = torch.sum(torch.square(v_grads)) / N
    return torch.sqrt(output1 + output2)


def ridge_lstsq(A, b, lambda_reg):
    m, n = A.shape
    # Add regularization term to normal equations
    ATA = A.T @ A
    I = torch.eye(n, device=A.device)
    ATb = A.T @ b
    # Solve (A^T A + λI)x = A^T b
    x = torch.linalg.solve(ATA + lambda_reg * I, ATb)
    return x



