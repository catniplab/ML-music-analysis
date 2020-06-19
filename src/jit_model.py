"""
This module provides functions which return just-in-time (jit) compiled models from a dictionary specifying the architecture, initialization, and other parameters.

We also initialize the hidden weights with this module.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.jit as jit
import torch.nn.functional as F

from custom_models import LINEAR

# TODO
# make sure we are using the same conventions as pytorch RNNs in terms of storing hidden states


# From the name of the architecture, return its constructor.
arch_to_constructor = {"LINEAR": LINEAR, "RNN_TANH": nn.RNN, "GRU": nn.GRU, "LSTM": nn.LSTM}

def make_identity(shape):
    """
    :param shape: shape of the desired tensor
    :return: tensor which is zeroes everywhere and ones on diagonal
    """

    result = torch.zeros(shape)

    for i in range(shape[0]):
        result[i, i] = 1.0

    return result

def _initialize(model, **initializer):
    """
    :param model: ReadOutModel
    :param initializer: a dictionary specifying all information about the desired initialization
    Initialize the model in-place.
    """

    if initializer['init'] == 'identity':
        shape = model.rnn.weight_hh_l0.weight.data.shape
        model.rnn.weight_hh_l0.weight.data = make_identity(shape)

    elif initializer['init'] != 'default':
        raise ValueError("Initialization {} not recognized.".format(initializer['init']))

    # scale field of the dictionary applies to all intializations
    model.rnn.weight_hh_l0.weight.data *= intializer['scale']


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
    _initialize(model, **kwargs['initializer'])

    jit_model = jit.script(model)

    return jit_model

