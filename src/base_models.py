"""
This module is defines custom pytorch modules that do thing we like.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# A linear dynamical system whose input is a linear transformation of the data.
# Doesn't use any contexts and can therefore be just-in-time compiled.
class LINEAR_JIT(nn.Module):

    def __init__(self, model_dict):

        super(LINEAR_JIT, self).__init__()

        self.mt19937 = np.random.MT19937()
        self.hh_seed = np.random.get_state()

        self.input_size = model_dict['input_size']
        self.hidden_size = model_dict['hidden_size']
        self.output_size = model_dict['output_size']

        self.weight_ih_l0 = nn.Linear(self.input_size, self.hidden_size)
        self.weight_hh_l0 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(self, x):

        N = x.shape[0]
        T = x.shape[1]

        hiddens = []
        initial = torch.randn((N, self.hidden_size), dtype=torch.float)
        hidden = initial + self.weight_ih_l0(x[:, 0])
        hiddens.append(hidden)

        for t in range(1, T):
            hidden = self.weight_hh_l0(hidden) + self.weight_ih_l0(x[:, t - 1])
            hiddens.append(hidden)

        return hiddens, hiddens


# A linear dynamical system whose input is a linear transformation of the data.
# Uses contexts to force all tensors to stay on the gpu.
class LINEAR_CUDA(nn.Module):

    def __init__(self, model_dict):

        super(LINEAR_CUDA, self).__init__()

        self.mt19937 = np.random.MT19937()
        self.hh_seed = np.random.get_state()

        self.input_size = model_dict['input_size']
        self.hidden_size = model_dict['hidden_size']
        self.output_size = model_dict['output_size']

        self.weight_ih_l0 = nn.Linear(self.input_size, self.hidden_size)
        self.weight_hh_l0 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(self, x):

        device = self.weight_hh_l0.weight.data.get_device()
        x = x.to(device)

        N = x.shape[0]
        T = x.shape[1]

        hiddens = []
        initial = torch.randn((N, self.hidden_size), dtype=torch.float, device=device)
        hidden = initial + self.weight_ih_l0(x[:, 0])
        hiddens.append(hidden)

        for t in range(1, T):
            hidden = self.weight_hh_l0(hidden) + self.weight_ih_l0(x[:, t - 1])
            hiddens.append(hidden)

        return hiddens, hiddens


# Simple affine transformation of the last time step, doesn't take the past into account
class REGRESSION_CUDA(nn.Module):

    def __init__(self, model_dict):

        super(REGRESSION_CUDA, self).__init__()

        self.mt19937 = np.random.MT19937()
        self.hh_seed = np.random.get_state()

        self.weights = nn.Linear(88, 88)

    def forward(self, x):

        device = self.weights.weight.data.get_device()
        x = x.to(device)

        N = x.shape[0]
        T = x.shape[1]

        outputs = []
        for t in range(T):
            outputs.append(self.weights(x[:, t]))

        return outputs, outputs


# Simple affine transformation of the last time step, doesn't take the past into account
class REGRESSION_JIT(nn.Module):

    def __init__(self, model_dict):

        super(REGRESSION_JIT, self).__init__()

        self.mt19937 = np.random.MT19937()
        self.hh_seed = np.random.get_state()

        self.weights = nn.Linear(88, 88)

    def forward(self, x):

        N = x.shape[0]
        T = x.shape[1]

        outputs = []
        for t in range(T):
            outputs.append(self.weights(x[:, t]))

        return outputs, outputs
