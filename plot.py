import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.stats import linregress
import numpy as np

# Use LaTeX fonts in plots
rc('text', usetex=True)
rc('font', family='serif')


# simple_error_plot(neurons, train_errors, test_errors, residuals)
"""
neurons = len(train_errors)
simple_error_plot(neurons, train_errors, test_errors, test_L2errors)
"""

def simple_error_plot(neurons, train_errors, test_errors, l2errors=None,
    algo='projection pursuit', residuals=None):
    X = range(1, neurons+1)
    plt.figure(figsize=(6, 4))
    plt.plot(X, train_errors, label='Train error', color='blue')
    plt.plot(X, test_errors, label='Test error', color='orange')
    if residuals != None:
        plt.plot(X, residuals, label='Residual', color='green')
    if l2errors != None:
        plt.plot(X, l2errors, label='$L^2$ error', color='green')
    plt.xlabel('Neurons')
    plt.ylabel('Error')
    plt.title(f'Relative errors of {algo}')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()
    return

# decay_order(train_errors)
# decay_order(test_errors)

def decay_order(l):
    # Generate the indices (n values) starting from 1
    n = np.arange(1, len(l) + 1)

    # Perform the log-log transformation
    log_n = np.log(n)
    log_l = np.log(l)

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(log_n, log_l)

    # The slope is -alpha
    alpha = -slope

    print(f"Estimated alpha: {alpha}")

    return


def band_plot(lst1, lst2):

    n = len(lst1[0])
    X = range(1, n+1)

    lst1_median = np.median(lst1, axis=0)
    lst1_5th = np.percentile(lst1, 5, axis=0)
    lst1_95th = np.percentile(lst1, 95, axis=0)

    lst2_median = np.median(lst2, axis=0)
    lst2_5th = np.percentile(lst2, 5, axis=0)
    lst2_95th = np.percentile(lst2, 95, axis=0)

    plt.figure(figsize=(6, 4))
    plt.plot(X, lst1_median, label='$L^2$ Median', color='blue')
    plt.fill_between(X, lst1_5th, lst1_95th, color='blue', alpha=0.2)
    plt.plot(X, lst2_median, label='Energy Median', color='orange')
    plt.fill_between(X, lst2_5th, lst2_95th, color='orange', alpha=0.2)

    plt.xlabel('Terms')
    plt.ylabel('Error')
    plt.title('Absolute errors of projection pursuit')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()


# plot_error_decomp(train_errors, Lerrors, Berrors, neurons)

def plot_error_decomp(train_errors, Lerrors, Berrors, neurons):
    X = range(1, neurons+1)
    plt.figure(figsize=(6, 4))
    plt.plot(X, Lerrors, label='$L$-error', color='blue')
    plt.plot(X, Berrors, label='$B$-error', color='orange')
    plt.plot(X, train_errors, label='Train error', color='green')
    plt.xlabel('Neurons')
    plt.ylabel('Error')
    plt.title('Projection pursuit errors')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()
    return




