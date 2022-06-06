import torch


import numpy as np
import streamlit as st

from model import NeuralNetwork
from utils import create_linspace, create_fig

# Steamlit
st.title('Solving ODE with neural network')
initial_value = st.sidebar.number_input('Initial value')

start_point = st.sidebar.number_input('Start point',)
end_point = st.sidebar.number_input('End point', value=2.0)
internal = (start_point, end_point)

N = NeuralNetwork()

network_psi = lambda x: initial_value + x * N(x)
forcing_function = lambda x, Psi: torch.exp(-x / 5.0) * torch.cos(x) - Psi / 5.0
actual_psi = lambda x: np.exp(-x / 5.0) * (np.sin(x) + initial_value)

def train():

    ## check if GPU is available and use it; otherwise use CPU
    torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def loss_function(x):

        x.requires_grad = True
        outputs = network_psi(x)
        network_psi_x = torch.autograd.grad(outputs, x, grad_outputs=torch.ones_like(outputs),
                            create_graph=True)[0]

        return  torch.mean( ( network_psi_x - forcing_function(x, outputs) )  ** 2)

    optimizer = torch.optim.LBFGS(N.parameters())

    x = torch.Tensor(create_linspace(internal))

    def closure():

        optimizer.zero_grad()
        l = loss_function(x)
        l.backward()
        
        return l

    for _ in range(100):
        optimizer.step(closure)

train()
internal_linspace = create_linspace(internal)

with torch.no_grad():
    network_trajectory = network_psi(torch.Tensor(internal_linspace)).numpy()
actual_trajectory = actual_psi(internal_linspace)


fig = create_fig(internal_linspace, network_trajectory, actual_trajectory)
st.pyplot(fig)
