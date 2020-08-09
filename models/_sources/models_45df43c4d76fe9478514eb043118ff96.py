"""
This module handles things we would like to put on top of our models such as

Linear Readout
Special Initialization
Just-in-Time (jit) compilation
Gradient Clipping
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

from src.base_models import LDS, REGRESSION, REGRESSION_WIDE


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
class ReadOutModel(nn.Module):

    def __init__(self, model_dict):

        super(ReadOutModel, self).__init__()

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


def make_identity(shape: torch.Size) -> torch.Tensor:
    """
    :param shape: shape of the desired matrix
    :return: matrix which is zeroes everywhere and ones on diagonal
    """

    result = torch.zeros(shape)

    for i in range(np.minimum(shape[0], shape[1])):
        result[i, i] = 1.0

    return result


def make_block_ortho(shape: torch.Size,
                     t_distrib: Distribution,
                     architecture="",
                     parity=None) -> torch.Tensor:
    """
    :param shape: shape of the block orthogonal matrix that we desire
    :param t_distrib: pytorch distribution from which we will sample the angles
    :param parity: None will return a mixed parity block orthogonal matrix, 'reflect' will return a decoupled system of 2D reflections, and anything else will return decoupled system of rotations.
    """

    if architecture in ["LINEAR", ""]:

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

    elif architecture == "GRU":

        hid_size = shape[1]
        tot_size = shape[0]
        result = torch.zeros(shape)

        sq = torch.Size([hid_size, hid_size])
        result[hid_size : 2*hid_size, 0 : hid_size] = make_identity(sq)
        result[0 : hid_size, 0 : hid_size] = make_identity(sq)
        result[2*hid_size : 3*hid_size, 0 : hid_size] = make_block_ortho(sq, t_distrib, parity=parity)

        return result

    else:
        raise TypeError("Unaccounted case in make_block_ortho")


def _initialize(model: ReadOutModel, architecture: str, initializer: dict) -> ReadOutModel:
    """
    :param model: ReadOutModel
    :param initializer: a dictionary specifying all information about the desired initialization
    Initialize the model in-place.
    """

    # 1s along diagonal and 0 elsewhere
    if initializer['init'] == 'identity':
        shape = model.rnn.weight_hh_l0.data.shape
        model.rnn.weight_hh_l0.data = initializer['scale']*make_identity(shape)

    # diagonal is 2x2 blocks of rotation, reflection, or random parity matrices with a scale factor
    elif initializer['init'] == 'blockortho':
        shape = model.rnn.weight_hh_l0.data.shape
        t_distrib = initializer['t_distrib']
        parity = initializer['parity']
        scale = initializer['scale']
        bortho = make_block_ortho(shape, t_distrib, architecture=architecture, parity=parity)
        model.rnn.weight_hh_l0.data = scale*bortho

    # simply orthonormalize the hidden weights in place
    elif initializer['init'] == 'ortho':
        nn.init.orthogonal_(model.rnn.weight_hh_l0.weight.data)

    # mean 0 variance 1 normal distribution for each hidden weight
    elif initializer['init'] == 'stdnormal':
        shape = model.rnn.weight_hh_l0.data.shape
        model.rnn.weight_hh_l0.data = initializer['scale']*torch.randn(shape)

    # construct the initial hidden weights based on the weights of a regression model
    elif initializer['init'] == 'regression':

        sq = torch.Size([88, 88])
        model.rnn.weight_ih_l0.weight.data[0 : 88, 0 : 88] = make_identity(sq)

        hid_size = model.rnn.weight_hh_l0.weight.data.shape
        model.rnn.weight_hh_l0.weight.data = torch.zeros(hid_size)
        slen = hid_size[0] - 88
        sub_size = torch.Size([slen, slen])
        scale = initializer['scale']
        t_distrib = initializer['t_distrib']
        parity = initializer['parity']
        model.rnn.weight_hh_l0.weight.data[88:, 88:] = scale*make_block_ortho(sub_size, t_distrib, parity=parity)


        model.output_weights.weight.data = torch.zeros(model.output_weights.weight.data.shape)
        model.output_weights.weight.data[0 : 88, 0 : 88] = torch.load(initializer['path'])['weights.weight']

    # construct the initial hidden weights based on the weights of a gru
    elif initializer['init'] == 'gru':

        #in_shape = model.rnn.weight_ih_l0.weight.data.shape
        #model.rnn.weight_ih_l0.data = make_identity(in_shape)

        hid_shape = model.rnn.weight_hh_l0.weight.data.shape
        identity = make_identity(hid_shape)
        gru_weights = torch.load(initializer['path'])['rnn.weight_hh_l0']
        gru_shape = gru_weights.shape

        for i in range(gru_shape[0] - hid_shape[0], gru_shape[0]):
            for j in range(hid_shape[1]):
                identity[i - 2*hid_shape[0], j] = gru_weights[i, j]
        model.rnn.weight_hh_l0.weight.data = initializer['scale']*identity

        #out_shape = model.output_weights.weight.data.shape
        #model.output_weights.data = make_identity(out_shape)

    # initialize an RNN based on the weights of an LDS
    if initializer['init'] == 'lds':

        lds_sd = torch.load(initializer['path'])

        model.rnn.weight_ih_l0.data = lds_sd['rnn.weight_ih_l0.weight']
        hidden_weights = lds_sd['rnn.weight_hh_l0.weight'].detach().numpy()
        absdet = abs(la.det(hidden_weights))
        model.rnn.weight_hh_l0.data = lds_sd['rnn.weight_hh_l0.weight']/absdet
        model.output_weights.weight.data = lds_sd['output_weights.weight']

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
    if model_dict['readout'] == 'linear':
        model = ReadOutModel(model_dict)
    else:
        if model_dict['architecture'] == "REGRESSION":
            model = REGRESSION(model_dict['lag'])
        elif model_dict['architecture'] == "REGRESSION_WIDE":
            model = REGRESSION_WIDE(model_dict['window'])
        else:
            raise ValueError("Architecture {} not recognized.".format(model_dict['architecture']))

    # initialize the model
    if initializer['init'] != 'default':
        _initialize(model, model_dict['architecture'], initializer)

    # if running on the cpu we may want to use just-in-time compilation
    if not cuda and model_dict['jit']:
        model = jit.script(model)

    return model

