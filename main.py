import torch

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from model import NeuralNetwork



# Steamlit
st.title('Solving ODE with neural network')
initial_value = st.sidebar.number_input('Initial value')
start_point = st.sidebar.number_input('Start point',)
end_point = st.sidebar.number_input('End point', value=2.0)

internal = (start_point, end_point)

N = NeuralNetwork()

Psi_t = lambda x: initial_value + x * N(x)
forcing_function = lambda x, Psi: torch.exp(-x / 5.0) * torch.cos(x) - Psi / 5.0
actual_Psi = lambda x: np.exp(-x / 5.0) * (np.sin(x) + initial_value)

def train():

    ## check if GPU is available and use it; otherwise use CPU
    torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def loss(x):

        x.requires_grad = True
        outputs = Psi_t(x)
        Psi_t_x = torch.autograd.grad(outputs, x, grad_outputs=torch.ones_like(outputs),
                            create_graph=True)[0]

        return  torch.mean( ( Psi_t_x - forcing_function(x, outputs) )  ** 2)

    optimizer = torch.optim.LBFGS(N.parameters())

    x = torch.Tensor(np.linspace(start_point, end_point, 100)[:, None])

    def closure():

        optimizer.zero_grad()
        l = loss(x)
        l.backward()
        
        return l

    for _ in range(100):
        optimizer.step(closure)

train()
xx = np.linspace(start_point, end_point, 100)[:, None]

with torch.no_grad():
    yy = Psi_t(torch.Tensor(xx)).numpy()
yt = actual_Psi(xx)

fig, ax = plt.subplots(dpi=100)
ax.plot(xx, yt, label='True')
ax.plot(xx, yy, '--', label='Neural network approximation')
ax.set_xlabel('$x$')
ax.set_ylabel('$Psi(x)$')
plt.legend(loc='best')

st.pyplot(fig)
