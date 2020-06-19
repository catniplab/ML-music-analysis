"""
This module is defines custom pytorch modules that do thing we like.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Not a model, but a pytorch module which computes the accuracy of binary data
class Accuracy(nn.Module):

    def __init__(self):

        super(Accuracy, self).__init__()

    def forward(self, output, target):

        size = 0
        for d in output.shape:
            size += d

        prediction = 1.0*(torch.softmax(output) > 0.5)

        agreement = torch.sum(prediction*target + (1 - prediction)*(1 - target))

        return agreement/size

# A linear dynamical system whose input is a linear transformation of the data.
class LINEAR(nn.Module):

    def __init__(self, dict):

        super(LINEAR, self).__init__()

        self.input_size = dict['input_size']
        self.hidden_size = dict['hidden_size']

        self.weight_ih_l0 = nn.Linear(self.input_size, self.hidden_size)
        self.weight_hh_l0 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.mt19937 = np.random.MT19937()
        self.hh_seed = np.random.get_state()

    def forward(self, x):

        #dev = x.get_device()

        N = x.shape[0]
        T = x.shape[1]

        hiddens = torch.zeros((N, T, self.hidden_size), dtype=torch.float)
        initial = torch.randn((N, self.hidden_size), dtype=torch.float)

        #hiddens = torch.zeros((N, T, self.hidden_size), dtype=torch.float, device=dev)
        #initial = torch.randn((N, self.hidden_size), dtype=torch.float, device=dev)
        #hiddens[:, 0] = initial + self.weight_ih_l0(x[:, 0])

        with torch.no_grad():
            hiddens[:, 0] = initial + self.weight_ih_l0(x[:, 0])

        for t in range(1, T):
            hidden = self.weight_hh_l0(hiddens[:, t - 1]) + self.weight_ih_l0(x[:, t - 1])
            with torch.no_grad():
                hiddens[:, t] = hidden

        return hiddens, hiddens

