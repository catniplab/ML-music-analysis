"""
This module is defines custom pytorch modules that do thing we like.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# A linear dynamical system whose input is a linear transformation of the data.
class LINEAR(nn.Module):

    def __init__(self, *args, **kwargs):

        super(LINEAR, self).__init__()

        # a dictionary containing all initialization parameters
        self.initializer = kwargs['initializer']

        self.input_size = kwargs['input_size']
        self.hidden_size = kwargs['hidden_size']

        self.weight_ih_l0 = nn.Linear(self.input_size, self.hidden_size)
        self.weight_hh_l0 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.mt19937 = np.random.MT19937()
        self.hh_seed = np.random.get_state()

        if self.initializer['init'] == identity:
            self.weight_hh_l0.weight.data = torch.zeros(self.weight_hh_l0.data.shape)
            for i in range(self.weight.data.shape[0]):
                self.weight.hh_l0.weight.data[i, i] = 1.0

        elif self.initializer['init'] != 'default':
            raise ValueError("Initialization {} is not recognized".format(self.initializer['init']))

    def forward(self, x):

        #dev = x.get_device()

        N = x.shape[0]
        T = x.shape[1]

        hiddens = torch.zeros((N, T, self.hidden_size), dtype=torch.float)
        initial = torch.randn((N, self.hidden_size), dtype=torch.float)

        #hiddens = torch.zeros((N, T, self.hidden_size), dtype=torch.float, device=dev)
        #initial = torch.randn((N, self.hidden_size), dtype=torch.float, device=dev)
        #hiddens[:, 0] = initial + self.weight_ih_l0(x[:, 0])

        hiddens[:, 0] = initial + self.weight_ih_l0(x[:, 0])

        for t in range(1, T):
            hidden = self.weight_hh_l0(hiddens[:, t - 1]) + self.weight_ih_l0(x[:, t - 1])
            hiddens[:, t] = hidden

        return hiddens, hiddens

