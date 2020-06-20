"""
This module handles things we would like to put on top of our models such as

Linear Readout
Special Initialization
Just-in-Time (jit) compilation
Gradient Clipping
"""

import numpy as np

import torch
import torch.nn as nn
import torch.jit as jit
import torch.nn.functional as F

from src.base_models import LINEAR

# TODO
# make sure we are using the same conventions as pytorch RNNs in terms of storing hidden states
# get jit working

def make_identity(shape: torch.Size):
    """
    :param shape: shape of the desired matrix
    :return: matrix which is zeroes everywhere and ones on diagonal
    """

    result = torch.zeros(shape)

    for i in range(np.minimum(shape[0], shape[1])):
        result[i, i] = 1.0

    return result


# A pytorch model together with a linear read-out
class ReadOutModel(nn.Module):

    def __init__(self, model_dict):

        super(ReadOutModel, self).__init__()

        self.hidden_size = model_dict['hidden_size']
        self.output_size = model_dict['output_size']
        self.output_weights = nn.Linear(self.hidden_size, self.output_size)

        # From the name of the architecture, return its constructor.
        arch_to_constructor = {"LINEAR": LINEAR, "TANH_RNN": nn.RNN, "GRU": nn.GRU, "LSTM": nn.LSTM}

        constructor = arch_to_constructor[model_dict['architecture']]
        self.rnn = constructor(model_dict)

        clip = model_dict['gradient_clipping']
        if clip != None:
            for p in self.parameters():
                p.register_hook(lambda grad: torch.clamp(grad, -clip, clip))

    def forward(self, x):

        hiddens, hn = self.rnn(x)
        output = self.output_weights(hiddens)
        return output, hiddens


def _initialize(model: ReadOutModel, initializer: dict) -> ReadOutModel:
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
    model.rnn.weight_hh_l0.weight.data *= initializer['scale']


# Get a just-in-time-compiled model from the argument dictionary.
def get_model(model_dict: dict, initializer: dict) -> ReadOutModel:

    model = ReadOutModel(model_dict)
    _initialize(model, initializer)

    if model_dict['jit']:
        model = jit.script(model)

    return model

