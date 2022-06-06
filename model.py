import torch.nn as nn
import torch

class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 50), 
            nn.Sigmoid(), 
            nn.Linear(50,1, bias=False)
        )

    def forward(self, x):
        return self.network(x)
