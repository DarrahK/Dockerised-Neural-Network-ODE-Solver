
import numpy as np
import torch

def create_linspace(internal):
    return np.linspace(internal[0], internal[1], 100)[:, None]
