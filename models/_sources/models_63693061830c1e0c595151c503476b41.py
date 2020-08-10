"""
This module handles the readout we put on top of models.
"""

import numpy as np
import numpy.linalg as la

import torch
import torch.nn as nn
import torch.jit as jit
import torch.nn.functional as F
from torch.distributions import Bernoulli
from torch.distributions.distribution import Distribution

from math import sin, cos

from src.neural_nets.base_models import LDS, REGRESSION, REGRESSION_WIDE
from src.neural_nets.initialization import _initialize


def get_constructor(architecture: str):
    """
    :param architecture: name of the architecture we want to use
    :param cuda: whether or not the model needs to be run on the gpu
    """

    if architecture == 'LDS':
        return LDS

    if architecture == 'GRU':
        return nn.GRU

    if architecture == 'LSTM':
        return nn.LSTM

    if architecture == 'TANH':
        return nn.RNN

    else:
        raise ValueError("Architecture {} not recognized.".format(architecture))


# A pytorch model together with a linear read-out
class LinReadOutModel(nn.Module):

    def __init__(self, model_dict):

        super(LinReadOutModel, self).__init__()

        # construct linear readout
        self.hidden_size = model_dict['hidden_size']
        self.output_size = model_dict['output_size']
        self.output_weights = nn.Linear(self.hidden_size, self.output_size)

        # get constructor for input and hidden layers
        architecture = model_dict['architecture']
        self.rnn = None
        constructor = get_constructor(model_dict['architecture'])

        # construct the model based on whether it is implemented in pytorch or base_models.py
        if architecture in ["LDS"]:
            self.rnn = constructor(model_dict)
        else:
            ins = model_dict['input_size']
            hids = model_dict['hidden_size']
            lays = model_dict['num_layers']
            self.rnn = constructor(input_size=ins, hidden_size=hids, num_layers=lays, batch_first=True)

        # gradient clipping if we want it
        clip = model_dict['gradient_clipping']
        if clip != None:
            for p in self.parameters():
                p.register_hook(lambda grad: torch.clamp(grad, -clip, clip))

    def forward(self, x):

        hiddens, hn = self.rnn(x)
        output = self.output_weights(hiddens)
        return output, hn


def get_model(model_dict: dict, initializer: dict, cuda: bool):
    """
    :param model_dict: dictionary specifying everything about the architecture
    :param initializer: dictionary specifying how to initialize the weights of the model
    :param cuda: whether or not the model needs to be cuda-compatible
    """

    readout = model_dict['readout']
    architecture = model_dict['architecture']

    # construct the model based on the type of readout
    # originally I thought I would incorporate non-linear readout
    #involving Boltzmann machines or something but it looks like that isn't happenning
    model = None
    if readout == 'linear':
        model = LinReadOutModel(model_dict)
    else:
        raise ValueError("readout {} not recognized.".format(readout))

    # initialize the model if necessary
    if initializer['init'] != 'default':

        # because of the way the parameters are structured,
        # initialization must take into account the architecture of the model
        _initialize(model, initializer, architecture=architecture)

    return model

