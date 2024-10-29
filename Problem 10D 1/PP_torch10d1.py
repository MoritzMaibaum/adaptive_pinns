import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import qmc
import numpy as np
import sys
import copy


""" Problem HD 1 """

seed = 1
torch.manual_seed(seed)
torch.set_default_dtype(torch.float64)

sys.path.append('/Users/moritz/Library/Mobile Documents/com~apple~Preview/Documents/MA_Literatur/Neural Networks/Experiments/Project')

from quadrature import *
from operators import *
from plot import *


def train():

    def Lproblemhd1(model, x, activation):
        if activation == 'relu2':
            laplacian = my_laplace_relu2(model, x).squeeze()
        elif activation == 'tanh':
            laplacian = my_laplace_tanh(model, x).squeeze()


        return -laplacian


    def Bproblemhd1(model, x, increment, activation):
        vals = model(x).squeeze()
        return vals


    def Bu(x):
        output = torch.sum(x[:, 0::2] * x[:, 1::2], axis=1)
        return output



    dim = 10
    iterations = 300
    width = 1
    activation = 'relu2'

    if activation == 'relu2':
        gain = torch.nn.init.calculate_gain('relu')
    elif activation == 'tanh':
        gain = torch.nn.init.calculate_gain('tanh')

    train_errors = []
    test_errors = []
    residuals = []
    Lerrors = []
    Berrors = []
    test_L2errors = []

    total_batches = 300
    batch_size = 400
    interior, boundary, interior_b, boundary_b, N_interior_b, N_facet_b = generate_batch_quadrature(dim, total_batches, batch_size, .5)
    N_interior, N_facet = interior.shape[0], boundary.shape[0]//(2*dim)

    u_L = interior[:, 0] * 0
    u_B = Bu(boundary)

    u_enorm = enorm(u_L, u_B, N_interior, N_facet)


    test_N_interior = 10 * N_interior
    test_N_facet = test_N_interior//(2*dim)
    test_interior, test_boundary = test_generate_quadrature(test_N_interior, test_N_facet, dim)

    test_u_L = test_interior[:, 0] * 0
    test_u_B = Bu(test_boundary)
    test_u_L2 = torch.sum(test_interior[:, 0::2] * test_interior[:, 1::2], axis=1)

    test_u_enorm = enorm(test_u_L, test_u_B, test_N_interior, test_N_facet)
    test_u_L2norm = L2norm(test_u_L2, test_N_interior)


    train_A = torch.zeros((N_interior + 2*dim*N_facet, iterations))
    train_b = torch.cat([u_L, u_B], dim=0)
    b_norm = torch.linalg.norm(train_b)
    residual = train_b
    residual_2norm = torch.linalg.norm(train_b)

    test_A = torch.zeros((test_N_interior + 2*dim*test_N_facet, iterations))
    test_V = torch.zeros((test_N_interior, iterations))

    data = {'Lnn' : Lproblemhd1, 'Bnn' : Bproblemhd1, 'activation' : activation, 'N_facet_b' : N_facet_b}

    approximants = []

    best_test = np.inf
    best_model = None

    for i in range(iterations):

        best_loss = np.inf
        best_candidate = None

        multistart = 10
        k = 0

        i_batch = next(interior_b)
        b_batch = next(boundary_b)

        data['i_batch'] = i_batch
        data['b_batch'] = b_batch

        mother_L_b = torch.zeros([N_interior_b])
        mother_B_b = torch.zeros([N_facet_b * 2 * dim])
        with torch.no_grad():
            for j in range(i):
                mother_L_b += c[j] * Lproblemhd1(approximants[j], i_batch, activation)
                mother_B_b += c[j] * Bproblemhd1(approximants[j], b_batch, None, activation)
        mother_b = torch.cat([mother_L_b, mother_B_b])

        u_L_b = i_batch[:, 0] * 0
        u_B_b = Bu(b_batch)
        u_b = torch.cat([u_L_b, u_B_b])

        residual_b = u_b - mother_b
        data['residual_b'] = residual_b


        while k < multistart or best_candidate == None:
        
            candidate_model = ANeuralNet(dim, width, 1, activation)
            with torch.no_grad():
                if k % 3 == 0:
                    nn.init.xavier_uniform_(candidate_model.fc1.weight, gain=gain)
                    nn.init.xavier_uniform_(candidate_model.fc2.weight, gain=gain)
                elif k % 3 == 1:
                    nn.init.xavier_normal_(candidate_model.fc1.weight, gain=gain)
                    nn.init.xavier_normal_(candidate_model.fc2.weight, gain=gain)
                else:
                    nn.init.uniform_(candidate_model.fc1.weight, a=-1, b=1)
                    nn.init.uniform_(candidate_model.fc2.weight, a=-1, b=1)
                nn.init.uniform_(candidate_model.fc1.bias, a=-1, b=1)
                

            if False:
                optimizer = next(optimizers)(candidate_model.parameters())

                for _ in range(1400):
                    optimizer.zero_grad()
                    loss = diff_l2_loss0(candidate_model, data)
                    loss.backward()
                    optimizer.step()
            
            # Optimization subproblem
            if True:
                optimizer = optim.LBFGS(candidate_model.parameters(),
                            lr=1,
                            max_iter=80,
                            max_eval=80*1.25,
                            tolerance_grad=1e-9, # default: 1e-7
                            tolerance_change=1e-11, # default: 1e-9
                            history_size=100,
                            line_search_fn="strong_wolfe")

                def closure():
                    optimizer.zero_grad()
                    loss = diff_l2_loss0(candidate_model, data)
                    loss.backward()
                    return loss

                loss = optimizer.step(closure)


            with torch.no_grad():
                l = diff_l2_loss0(candidate_model, data) 
            if l < best_loss:
                best_loss = l
                best_candidate = candidate_model
            k += 1


        print(f"{i+1} Iterations")
        with torch.no_grad():
            best_candidate.fc2.weight /= torch.linalg.norm(best_candidate.fc2.weight.squeeze())

        approximants.append(best_candidate)

        with torch.no_grad():
            candidate_L = Lproblemhd1(best_candidate, interior, activation)
            candidate_B = Bproblemhd1(best_candidate, boundary, None, activation)

            test_candidate_L = Lproblemhd1(best_candidate, test_interior, activation)
            test_candidate_B = Bproblemhd1(best_candidate, test_boundary, None, activation)

            test_V[:, i] = best_candidate(test_interior).squeeze()

        train_A[:N_interior, i] = candidate_L
        train_A[N_interior:, i] = candidate_B

        test_A[:test_N_interior, i] = test_candidate_L
        test_A[test_N_interior:, i] = test_candidate_B


        
        # OGA (Projection)
        dict_ = torch.linalg.lstsq(train_A[:, :i+1], train_b, rcond=1e-15, driver='gelsd')
        c = dict_.solution
        #c = ridge_lstsq(train_A[:, :i+1], train_b, 0)
        train_mother = train_A[:, :i+1] @ c
        test_mother = test_A[:, :i+1] @ c
        test_mother_L2 = test_V[:, :i+1] @ c
        

        """
        # PGA
        if i == 0:
            c = torch.zeros([iterations])

        c[i] = torch.dot(residual, train_A[:, i])/torch.dot(train_A[:, i], train_A[:, i])
        train_mother = train_A[:, :i+1] @ c[:i+1]
        test_mother = test_A[:, :i+1] @ c[:i+1]
        test_mother_L2 = test_V[:, :i+1] @ c[:i+1]
        """
        
        
        train_mother_L = train_mother[:N_interior]
        train_mother_B = train_mother[N_interior:]

        residual = torch.cat([u_L - train_mother_L, u_B - train_mother_B])
        residual_2norm = torch.linalg.norm(residual)
        residuals.append(residual_2norm / b_norm)

        test_mother_L = test_mother[:test_N_interior]
        test_mother_B = test_mother[test_N_interior:]

        Lerrors.append(torch.sqrt(torch.sum(torch.square(u_L-train_mother_L))/N_interior))
        Berrors.append(torch.sqrt(torch.sum(torch.square(u_B-train_mother_B))/N_facet))
        print(f"L error: {Lerrors[-1].round(decimals=5)}, B error: {Berrors[-1].round(decimals=5)}")

        train_error = energy_error(u_L, u_B, train_mother_L, train_mother_B, N_interior, N_facet, u_enorm)
        train_errors.append(train_error)
        print(f"Train ||u-v||_a: {train_error}")
        #print(f"Train ||u-v||_a / ||u||_a: {train_error / u_enorm}")

        test_error = energy_error(test_u_L, test_u_B, test_mother_L, test_mother_B, test_N_interior, test_N_facet, test_u_enorm)
        test_errors.append(test_error)
        print(f"Test ||u-v||_a: {test_error}")
        #print(f"Test ||u-v||_a / ||u||_a: {test_error / test_u_enorm}")

        L2error = L2norm(test_u_L2 - test_mother_L2, test_N_interior)
        test_L2errors.append(L2error)
        print(f"Test ||u-v||_L2: {L2error}")
        #print(f"Test ||u-v||_L2 / ||u||_L2: {L2error / test_u_L2norm}")

        #print(f"Condition number: {torch.linalg.cond(train_A[:,:i+1])}")
        print("\n")

return test_errors, test_L2errors




