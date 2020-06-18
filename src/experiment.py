"""
This script creates an instance of a sacred experiment and defines default configurations.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import random

from sacred import Experiment

# give all random number generators the same seed
def _seed_all(_seed) -> None:
    torch.manual_seed(_seed)
    np.random.seed(_seed)
    random.seed(_seed)


# create a new sacred experiment whose name is an integer
ex = Experiment(name=str(random.randint(0, 1000000)))


# default configurations
@ex.config
def cfg():

    # system arguments
    cuda = torch.cuda.is_available()
    save_dir = os.getcwd()

    # training arguments
    name = "Nottingham"
    num_epochs = 150
    batch_size = 128
    lr = 1e-3
    max_epochs = 5
    optimizer = "SGD"

    # do hyper-parameter search
    hpsearch = False

    # detect backprop anamolies
    detect_anomaly = False

    #RNN arguments
    architecture = 'LINEAR'
    gradient_clipping = None
    input_size = 88
    hidden_size = 300
    num_layers = 1
    output_size = 6

    # initializer dictionary contains all initialization information
    initializer = {'init': 'default', 'scale': 1.0, 'min_angle': 0.0, 'max_angle': 2.0}
