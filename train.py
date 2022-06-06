import torch
from utils import create_linspace

## check if GPU is available and use it; otherwise use CPU
torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(model, interval, network_psi, forcing_function):

    def loss_function(x):

        x.requires_grad = True
        outputs = network_psi(x)
        network_psi_x = torch.autograd.grad(outputs, x, grad_outputs=torch.ones_like(outputs),
                            create_graph=True)[0]

        return  torch.mean( ( network_psi_x - forcing_function(x, outputs) )  ** 2)

    optimizer = torch.optim.LBFGS(model.parameters())

    x = torch.Tensor(create_linspace(interval))

    def closure():

        optimizer.zero_grad()
        l = loss_function(x)
        l.backward()
        
        return l

    for _ in range(100):
        optimizer.step(closure)
