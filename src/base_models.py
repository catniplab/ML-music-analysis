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

        self.mt19937 = np.random.MT19937()
        self.hh_seed = np.random.get_state()

        self.input_size = model_dict['input_size']
        self.hidden_size = model_dict['hidden_size']
        self.output_size = model_dict['output_size']

        #self.weight_ih = nn.Parameter(torch.randn(self.input_size, self.hidden_size,))
        #self.bias = nn.Parameter(torch.randn(self.hidden_size))
        #self.weight_hh = nn.Parameter(torch.randn(self.hidden_size, self.hidden_size))
        self.weight_ih_l0 = nn.Linear(self.input_size, self.hidden_size)
        self.weight_hh_l0 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        #self.weight_ho = nn.Linear(self.hidden_size, self.output_size, bias=False)

    def forward(self, x):

        dev = 'cpu'
        if torch.cuda.is_available():
            dev = torch.cuda.current_device()

        N = x.shape[0]
        T = x.shape[1]

        hiddens = []
        initial = torch.randn((N, self.hidden_size), dtype=torch.float, device=dev)
        hidden = initial + self.weight_ih_l0(x[:, 0])
        hiddens.append(hidden)

        for t in range(1, T):
            hidden = self.weight_hh_l0(hidden) + self.weight_ih_l0(x[:, t - 1])
            hiddens.append(hidden)

        return hiddens, hiddens

