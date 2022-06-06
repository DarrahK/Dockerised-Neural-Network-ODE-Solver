
import numpy as np
import matplotlib.pyplot as plt

def create_linspace(internal):
    return np.linspace(internal[0], internal[1], 100)[:, None]

def create_fig(interval_linspace, network_trajectory, actual_trajectory):
    fig, ax = plt.subplots(dpi=100)
    ax.plot(interval_linspace, actual_trajectory, label='True')
    ax.plot(interval_linspace, network_trajectory, '--', label='Neural network approximation')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$Psi(x)$')
    plt.legend(loc='best')
    return fig