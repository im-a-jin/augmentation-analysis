import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class NonlinearModel(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=500, out_dim=1, init_var=0.0000625):
        super(NonlinearModel, self).__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.init_var = init_var

        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, out_dim, bias=False)

        # initialize weights
        init_std = np.sqrt(self.init_var)
        nn.init.normal_(self.linear1.weight, mean=0, std=init_std)
        nn.init.normal_(self.linear1.bias, mean=0, std=init_std)
        split = hidden_dim / 2
        weights = torch.zeros((1, hidden_dim))
        weights[:, :int(np.floor(split))] = 1 / np.sqrt(hidden_dim)
        weights[:, int(np.ceil(split)):] = -1 / np.sqrt(hidden_dim)
        self.linear2.weight = nn.Parameter(weights, requires_grad=False)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))
