"""
This module provides functions which return just-in-time (jit) compiled models from a dictionary specifying the architecture, initialization, and other parameters.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.jit as jit
import torch.nn.functional as F

from custom_models import LINEAR

# From the name of the architecture, return its constructor.
arch_to_constructor = {"LINEAR": LINEAR, "RNN_TANH": nn.RNN, "GRU": nn.GRU, "LSTM": nn.LSTM}


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

    jit_model = jit.script(model)

    return jit_model

