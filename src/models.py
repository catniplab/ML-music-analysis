"""
This module provides functions which return just-in-time (jit) compiled models from a dictionary specifying the architecture, initialization, and other parameters.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.jit as jit
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

    def reset_parameters(self) -> None:
        """
        Initializes weights (in place)
        """

        kwargs = self.init_kwargs
        init = self.kwargs['initializer']

        raise NotImplementedError

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


# From the name of the architecture, return its constructor.
arch_to_constructor = {"LINEAR": LINEAR}


# A pytorch model together with a linear read-out
class ReadOutModel(nn.Module):

    def __init__(self, *args, **kwargs):

        super(ReadOutModel, self).__init__()

        print(kwargs)

        self.hidden_size = kwargs['hidden_size']
        self.output_size = kwargs['output_size']
        self.output_weights = nn.Linear(self.hidden_size, self.output_size)

        constructor = arch_to_constructor[kwargs['architecture']]
        self.rnn = constructor(**kwargs)

    def forward(self, x):

        hiddens, hn = self.rnn(x)
        output = self.output_weights(hiddens)
        return output, hiddens


# Get a just-in-time-compiled model from the argument dictionary.
def get_model(**kwargs):

    model = ReadOutModel(**kwargs)

    fast_model = jit.script(model)

    return fast_model

