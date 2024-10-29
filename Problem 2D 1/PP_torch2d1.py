import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import qmc
import numpy as np
import sys
import copy
from scipy.optimize import minimize

""" Problem 2D 1 """

#seed = 2
#torch.manual_seed(seed)
torch.set_default_dtype(torch.float64)

sys.path.append('/Users/moritz/Library/Mobile Documents/com~apple~Preview/Documents/MA_Literatur/Neural Networks/Experiments/Project')

from quadrature import *
from operators import *
from plot import *


def train():
    # Interior differential operator
    def Lproblem2d1(model, x, activation):
        if activation == 'relu2':
            laplacian = my_laplace_relu2(model, x).squeeze()
        elif activation == 'tanh':
            laplacian = my_laplace_tanh(model, x).squeeze()

        reaction = model(x).squeeze()

        return laplacian + reaction

    # Boundary differential operator
    def Bproblem2d1(model, x, increment, activation):
        l = x.shape[0]
        output = torch.zeros([l])
        grads = gradient(model, x, activation=activation)
        vals = model(x).squeeze()
        counter = 0
        while counter < l:
            output[counter:counter+increment] = vals[counter:counter+increment]
            counter += increment

            output[counter:counter+increment] = grads[counter:counter+increment, 0]
            counter += increment

            output[counter:counter+increment] = vals[counter:counter+increment]
            counter += increment

            output[counter:counter+increment] = grads[counter:counter+increment, 1]
            counter += increment


        return output

    # Right hand side of the boundary condition.
    def Bu(x, increment):
        l = x.shape[0]
        output = torch.zeros([l])
        counter = 0
        while counter < l:
            # x = 0
            output[counter:counter+increment] = 0
            counter += increment
            # x = 1
            output[counter:counter+increment] = 2*torch.exp(1 - x[counter:counter+increment, 1])
            counter += increment
            # y = 0
            output[counter:counter+increment] = x[counter:counter+increment, 0]*torch.exp(x[counter:counter+increment, 0])
            counter += increment
            # y = 1
            output[counter:counter+increment] = -x[counter:counter+increment, 0]*torch.exp(x[counter:counter+increment, 0] - 1)
            counter += increment

        return output



    dim = 2
    iterations = 300
    # Width of single layer network dictionary
    width = 1
    # Activation function
    activation = 'tanh'

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

    total_batches = 1
    batch_size = 10000
    interior, boundary, interior_b, boundary_b, N_interior_b, N_facet_b = generate_batch_quadrature(dim, total_batches, batch_size, .5)
    N_interior, N_facet = interior.shape[0], boundary.shape[0]//(2*dim)

    f = (2 + 3*interior[:,0])*torch.exp(interior[:,0] - interior[:, 1])
    g = Bu(boundary, N_facet_b)

    u_enorm = enorm(f, g, N_interior, N_facet)


    test_N_interior = 10*N_interior
    test_N_facet = test_N_interior//(2*dim)
    test_interior, test_boundary = test_generate_quadrature(test_N_interior, test_N_facet, dim)

    test_f = (2 + 3*test_interior[:,0])*torch.exp(test_interior[:,0] - test_interior[:, 1])
    test_g = Bu(test_boundary, test_N_facet)
    test_u_L2 = test_interior[:, 0]*torch.exp(test_interior[:, 0] - test_interior[:, 1])

    test_u_enorm = enorm(test_f,test_g,test_N_interior,test_N_facet)
    test_u_L2norm = L2norm(test_u_L2, test_N_interior)


    # Initalize matrix to store model plugged into differential operators and evaluated on quadrature points for training
    train_A = torch.zeros((N_interior + 2*dim*N_facet, iterations))
    train_b = torch.cat([f, g], dim=0)
    b_norm = torch.linalg.norm(train_b)
    # Initialize residual
    residual = train_b
    residual_2norm = torch.linalg.norm(train_b)
    # Same as above but for testing
    test_A = torch.zeros((test_N_interior + 2*dim*test_N_facet, iterations))
    # Initialize matrix to store model function values for computing test L2 error.
    test_V = torch.zeros((test_N_interior, iterations))


    data = {'Lnn' : Lproblem2d1, 'Bnn' : Bproblem2d1, 'activation' : activation, 'N_facet_b' : N_facet_b}

    approximants = []

    best_test = np.inf
    best_model = None

    for i in range(iterations):

        best_loss = np.inf
        best_candidate = None

        multistart = 8
        k = 0

        i_batch = next(interior_b)
        b_batch = next(boundary_b)

        data['i_batch'] = i_batch
        data['b_batch'] = b_batch

        mother_L_b = torch.zeros([N_interior_b])
        mother_B_b = torch.zeros([N_facet_b * 2 * dim])
        with torch.no_grad():
            for j in range(i):
                mother_L_b += c[j] * Lproblem2d1(approximants[j], i_batch, activation)
                mother_B_b += c[j] * Bproblem2d1(approximants[j], b_batch, N_facet_b, activation)
        mother_b = torch.cat([mother_L_b, mother_B_b])

        f_b = (2 + 3*i_batch[:,0])*torch.exp(i_batch[:,0] - i_batch[:, 1])
        g_b = Bu(b_batch, N_facet_b)
        u_b = torch.cat([f_b, g_b])

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
                    nn.init.uniform_(candidate_model.fc1.weight, a=-2, b=2)
                    nn.init.uniform_(candidate_model.fc2.weight, a=-2, b=2)
                nn.init.uniform_(candidate_model.fc1.bias, a=-2, b=2)
            

            if False:
                optimizer = next(optimizers)(candidate_model.parameters())

                for _ in range(700):
                    optimizer.zero_grad()
                    loss = quot_l2_loss(candidate_model, data)
                    loss.backward()
                    optimizer.step()


            if True:
                optimizer = optim.LBFGS(candidate_model.parameters(),
                            lr=1,
                            history_size=1000, 
                            max_iter=40,
                            line_search_fn="strong_wolfe")

                def closure():
                    optimizer.zero_grad()
                    loss = diff_l2_loss0(candidate_model, data)
                    loss.backward()
                    return loss

                loss = optimizer.step(closure)


            if False:
                theta_start = 4*(np.random.random((N_params,))-1)
                res = minimize(scipy_quot_l2_loss, theta_start, method='SLSQP')
                vector_to_model_params(candidate_model, torch.tensor(res.x))

            
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
            candidate_L = Lproblem2d1(best_candidate, interior, activation)
            candidate_B = Bproblem2d1(best_candidate, boundary, N_facet_b, activation)

            test_candidate_L = Lproblem2d1(best_candidate, test_interior, activation)
            test_candidate_B = Bproblem2d1(best_candidate, test_boundary, test_N_facet, activation)

            test_V[:, i] = best_candidate(test_interior).squeeze()

        train_A[:N_interior, i] = candidate_L
        train_A[N_interior:, i] = candidate_B

        test_A[:test_N_interior, i] = test_candidate_L
        test_A[test_N_interior:, i] = test_candidate_B

        
        # OGA
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

        residual = torch.cat([f - train_mother_L, g - train_mother_B])
        residual_2norm = torch.linalg.norm(residual)
        residuals.append(residual_2norm / b_norm)

        test_mother_L = test_mother[:test_N_interior]
        test_mother_B = test_mother[test_N_interior:]

        Lerrors.append(torch.sqrt(torch.sum(torch.square(f-train_mother_L))/N_interior))
        Berrors.append(torch.sqrt(torch.sum(torch.square(g-train_mother_B))/N_facet))
        print(f"L error: {Lerrors[-1].round(decimals=5)}, B error: {Berrors[-1].round(decimals=5)}")

        train_error = energy_error(f, g, train_mother_L, train_mother_B, N_interior, N_facet, u_enorm)
        train_errors.append(train_error)
        print(f"Train ||u-v||_a / ||u||_a: {train_error}")

        test_error = energy_error(test_f, test_g, test_mother_L, test_mother_B, test_N_interior, test_N_facet, test_u_enorm)
        test_errors.append(test_error)
        print(f"Test ||u-v||_a / ||u||_a: {test_errors[-1]}")

        L2error = L2norm(test_u_L2 - test_mother_L2, test_N_interior)
        test_L2errors.append(L2error)
        print(f"Test ||u-v||_L2 / ||u||_L2: {test_L2errors[-1]}")

        #print(f"Condition number: {torch.linalg.cond(train_A[:,:i+1])}")
        print("\n")

    return test_errors, test_L2errors

