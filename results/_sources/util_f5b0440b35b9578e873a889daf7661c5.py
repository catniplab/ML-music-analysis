"""
Random utilities which don't seem to belong anywhere.
"""

from src.custom_models import Accuracy

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

from torch.utils.data import DataLoader
from sacred import Experiment
from copy import deepcopy
from tqdm import tqdm


class NullContext(object):

    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        pass


def compute_loss(loss_fcn: nn.Module, model: nn.Module, loader: DataLoader) -> float:
    """
    :param loss_fcn: pytorch module whose forward function computes a loss
    :param model: model which we are testing
    :param loader: DataLoader for either testing or validation data
    :return: average loss for every batch in the loader
    """

    all_loss = []

    for input_tensor, target_tensor in loader:

        output, hiddens = model(input_tensor)
        prediction = output[:, -1]

        loss = loss_fcn(prediction, target_tensor)
        all_loss.append(loss.cpu().detach().item())

    return np.mean(all_loss)


def compute_accuracy(model: nn.Module, loader: DataLoader) -> float:
    """
    :param model: model which we are testing
    :param loader: DataLoader for either testing or validation data
    :return: average accuracy for every batch in the loader
    """

    all_acc = []

    acc_fcn = Accuracy()

    for input_tensor, target_tensor in loader:

        output, hiddens = model(input_tensor)
        prediction = output[:, -1]

        acc = acc_fcn(prediction, target_tensor)
        all_acc.append(acc.cpu().detach().item())

    return np.mean(all_acc)


