import torch


import numpy as np
import streamlit as st

from model import NeuralNetwork
from train import train_model 
from utils import create_linspace, create_fig

# Steamlit
st.title('Solving ODE with neural network')
st.latex(r"""
\frac{d}{dx} \Psi(x) + \frac{1}{5}\Psi(x) = e^{-\frac{x}{5}} \cos(x)
""")

initial_value = st.sidebar.number_input('Initial value')

start_point = st.sidebar.number_input('Start point',)
end_point = st.sidebar.number_input('End point', value=2.0)
interval = (start_point, end_point)

# Solved ODE using Neural Network
model = NeuralNetwork()
network_psi = lambda x: initial_value + x * model(x)

# Forcing function that we want of the ODE
forcing_function = lambda x, Psi: torch.exp(-x / 5.0) * torch.cos(x) - Psi / 5.0

# Training Network
train_model(model, interval, network_psi, forcing_function)

# Working out the trajectories
interval_linspace = create_linspace(interval)

with torch.no_grad():
    network_trajectory = network_psi(torch.Tensor(interval_linspace)).numpy()

actual_psi = lambda x: np.exp(-x / 5.0) * (np.sin(x) + initial_value)
actual_trajectory = actual_psi(interval_linspace)

# Steamlit plotting
fig = create_fig(interval_linspace, network_trajectory, actual_trajectory)
st.pyplot(fig)
