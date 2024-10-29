import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.stats import linregress
import numpy as np
import sys
import torch
import pickle
from plot import *

sys.path.append('/Users/moritz/Library/Mobile Documents/com~apple~Preview/Documents/MA_Literatur/Neural Networks/Experiments/Project/Problem 2D 1')
sys.path.append('/Users/moritz/Library/Mobile Documents/com~apple~Preview/Documents/MA_Literatur/Neural Networks/Experiments/Project/Problem 2D 2')
sys.path.append('/Users/moritz/Library/Mobile Documents/com~apple~Preview/Documents/MA_Literatur/Neural Networks/Experiments/Project/Problem HD 1')


def collect(problem, runs=30):
    """ Collect test energy and L2 errors for 30 runs """
    if problem == 1:
        from PP_torch2d1 import train
    elif problem == 2:
        from PP_torch2d2 import train
    elif problem == 3:
        from PP_torch10d1 import train

    seeds = list(range(runs))
    eerrors_seed = []
    l2errors_seed = []

    for j in range(runs):

        torch.manual_seed(seeds[j])
        print(f"Run number {j+1}")
        test_errors, L2errors = train()
        eerrors_seed.append(test_errors)
        l2errors_seed.append(L2errors)

    return eerrors_seed, l2errors_seed




def load_data():
    # Specify the correct relative path to the file
    path = '/Users/moritz/Library/Mobile Documents/com~apple~Preview/Documents/MA_Literatur/Neural Networks/Experiments/Project/Problem HD 1/errors_lists.pkl'

    # Loading the lists of arrays from the file
    with open(path, 'rb') as f:
        data = pickle.load(f)

    # Access the lists
    l2errors_lst = data['l2errors_lst']
    eerrors_lst = data['eerrors_lst']

    return l2errors_lst, eerrors_lst

"""
l2errors_lst, eerrors_lst = load_data()
band_plot(l2errors_lst, eerrors_lst)
"""
