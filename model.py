import torch
import torch.nn as nn
from collections import OrderedDict

class SimpleMLP(nn.Module):
    def __init__(self, inp, hidden, out):
        super().__init__()
        self.net = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(inp, hidden)),
            ("relu1", nn.ReLU()),
            ("head", nn.Linear(hidden, out))
        ]))
    def forward(self, x):
        return self.net(x)

def get_model(input_dim, hidden, num_classes, device='cpu'):
    m = SimpleMLP(input_dim, hidden, num_classes)
    return m.to(device)
