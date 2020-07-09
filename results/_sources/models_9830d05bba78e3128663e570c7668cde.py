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
from torch.distributions import Bernoulli
from torch.distributions.distribution import Distribution

from math import sin, cos

from src.base_models import LINEAR_CUDA, LINEAR_JIT, REGRESSION, REGRESSION_WIDE

# TODO
# make sure we are using the same conventions as pytorch RNNs in terms of storing hidden states
# get jit working

def make_identity(shape: torch.Size) -> torch.Tensor:
    """
    :param shape: shape of the desired matrix
    :return: matrix which is zeroes everywhere and ones on diagonal
    """

    result = torch.zeros(shape)

    for i in range(np.minimum(shape[0], shape[1])):
        result[i, i] = 1.0

    return result


def make_block_ortho(shape: torch.Size, t_distrib: Distribution, parity=None) -> torch.Tensor:
    """
    :param shape: shape of the block orthogonal matrix that we desire
    :param t_distrib: pytorch distribution from which we will sample the angles
    :param parity: None will return a mixed parity block orthogonal matrix, 'reflect' will return a decoupled system of 2D reflections, and anything else will return decoupled system of rotations.
    """

    if shape[0] != shape[1] or shape[0]%2 == 1:
        raise ValueError("Tried to get block orthogonal matrix of shape {}".format(str(shape)))

    n = shape[0]//2
    result = torch.zeros(shape)

    bern = Bernoulli(torch.tensor([0.5]))

    for i in range(n):

        t = t_distrib.sample()
        mat = torch.tensor([[cos(t), sin(t)], [-sin(t), cos(t)]])

        if parity == None:
            mat[0] *= 2*bern.sample() - 1
        elif parity == 'reflect':
            mat[0] *= -1

        result[2*i : 2*(i + 1), 2*i : 2*(i + 1)] = mat

    return result


def get_constructor(architecture: str, cuda: bool):
    """
    :param architecture: name of the architecture we want to use
    :param cuda: whether or not the model needs to be run on the gpu
    """

    if architecture == 'LINEAR':
        if cuda:
            return LINEAR_CUDA
        else:
            return LINEAR_JIT

    else:
        raise ValueError("Architecture {} not recognized.".format(architecture))


# A pytorch model together with a linear read-out
class ReadOutModel(nn.Module):

    def __init__(self, model_dict, cuda):

        super(ReadOutModel, self).__init__()

        # construct linear readout
        self.hidden_size = model_dict['hidden_size']
        self.output_size = model_dict['output_size']
        self.output_weights = nn.Linear(self.hidden_size, self.output_size)

        # construct input and hidden layer
        constructor = get_constructor(model_dict['architecture'], cuda)
        self.rnn = constructor(model_dict)

        # gradient clipping if we want it
        clip = model_dict['gradient_clipping']
        if clip != None:
            for p in self.parameters():
                p.register_hook(lambda grad: torch.clamp(grad, -clip, clip))

    def forward(self, x):

        hiddens, hn = self.rnn(x)
        output = self.output_weights(hiddens)
        return output, hn


def _initialize(model: ReadOutModel, initializer: dict) -> ReadOutModel:
    """
    :param model: ReadOutModel
    :param initializer: a dictionary specifying all information about the desired initialization
    Initialize the model in-place.
    """

    # 1s along diagonal and 0 elsewhere
    if initializer['init'] == 'identity':
        shape = model.rnn.weight_hh_l0.weight.data.shape
        model.rnn.weight_hh_l0.weight.data = initializer['scale']*make_identity(shape)

    # diagonal is 2x2 blocks of rotation, reflection, or random parity matrices with a scale factor
    elif initializer['init'] == 'blockortho':
        shape = model.rnn.weight_hh_l0.weight.data.shape
        t_distrib = initializer['t_distrib']
        parity = initializer['parity']
        scale = initializer['scale']
        model.rnn.weight_hh_l0.weight.data = scale*make_block_ortho(shape, t_distrib, parity)

    # simply orthonormalize the hidden weights in place
    elif initializer['init'] == 'ortho':
        nn.init.orthogonal_(model.rnn.weight_hh_l0.weight.data)

    # mean 0 variance 1 normal distribution for each hidden weight
    elif initializer['init'] == 'stdnormal':
        shape = model.rnn.weight_hh_l0.weight.data.shape
        model.rnn.weight_hh_l0.weight.data = initializer['scale']*torch.randn(shape)

    # construct the initial hidden weights based on the weights of a regression model
    elif initializer['init'] == 'regression':

        in_shape = model.rnn.weight_ih_l0.weight.data.shape
        model.rnn.weight_ih_l0.weight.data = make_identity(in_shape)

        hid_shape = model.rnn.weight_hh_l0.weight.data.shape
        identity = make_identity(hid_shape)
        reg_weights = torch.load(initializer['path'])['weights.weight']
        for i in range(reg_weights.shape[0]):
            for j in range(reg_weights.shape[1]):
                identity[i, j] = reg_weights[i, j]
        model.rnn.weight_hh_l0.weight.data = initializer['scale']*identity

        out_shape = model.output_weights.weight.data.shape
        model.output_weights.weight.data = make_identity(out_shape)

    else:
        raise ValueError("Initialization {} not recognized.".format(initializer['init']))


def get_model(model_dict: dict, initializer: dict, cuda: bool):
    """
    :param model_dict: dictionary specifying everything about the architecture
    :param initializer: dictionary specifying how to initialize the weights of the model
    :param cuda: whether or not the model needs to be cuda-compatible
    """

    # construct the model
    model = None
    if not model_dict['architecture'] in ["REGRESSION", "REGRESSION_WIDE"]:
        model = ReadOutModel(model_dict, cuda)
    else:
        if model_dict['architecture'] == "REGRESSION":
            model = REGRESSION(model_dict['lag'])
        elif model_dict['architecture'] == "REGRESSION_WIDE":
            model = REGRESSION_WIDE(model_dict['window'])

    # initialize the model
    if initializer['init'] != 'default':
        _initialize(model, initializer)

    # if running on the cpu we may want to use just-in-time compilation
    if not cuda and model_dict['jit']:
        model = jit.script(model)

    return model

