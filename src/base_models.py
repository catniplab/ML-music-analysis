"""
This module is defines custom pytorch modules that do thing we like.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# A linear dynamical system whose input is a linear transformation of the data.
class LINEAR(nn.Module):

    def __init__(self, model_dict):

        super(LINEAR, self).__init__()

        self.input_size = model_dict['input_size']
        self.hidden_size = model_dict['hidden_size']
        self.output_size = model_dict['output_size']

        self.weight_ih = nn.Parameter(torch.randn(self.input_size, self.hidden_size,))
        self.bias = nn.Parameter(torch.randn(self.hidden_size))
        self.weight_hh = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size))
        #self.weight_ih = nn.Linear(self.input_size, self.hidden_size, bias=False)
        #self.weight_hh = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        #self.weight_ho = nn.Linear(self.hidden_size, self.output_size, bias=False)

        self.mt19937 = np.random.MT19937()
        self.hh_seed = np.random.get_state()

    def forward(self, x):

        dev = 'cpu'
        #if torch.cuda.is_available():
        #    dev = torch.cuda.current_device()

        N = x.shape[0]
        T = x.shape[1]

        hiddens = torch.tensor((N, T, self.hidden_size), dtype=torch.float, device=dev)
        initial = torch.randn((N, self.hidden_size), dtype=torch.float, device=dev)
        hiddens[:, 0] = initial + torch.mm(self.weight_ih, x[:, 0])

        for t in range(1, T):
            hiddens[:, t] = torch.mm(self.weight_hh, hiddens[:, t - 1]) + torch.mm(self.weight_ih, x[:, t - 1]) + self.bias

        return hiddens, hiddens

