"""
For figuring out whether or not neural networks possess attractors, and where they are.
"""

import math
import torch
import numpy as np

from src.neural_nets.models import get_model


my_dict = {'architecture': "TANH",
           'readout': "linear",
           'input_size': 88,
           'hidden_size': 120,
           'num_layers': 1,
           'output_size': 88,
           'gradient_clipping': 1
           }


def generate_trajectories(model_dict: dict, sd_path: str, filename: str):

    model = get_model(model_dict, {'init': "default"}, False)
    sd = torch.load(sd_path)
    model.load_state_dict(sd)

    initial_conditions = torch.randn((8192, 1, 88))
    last_tensor = initial_conditions

    result_tensor = torch.zeros((8192, 257, 88))
    result_tensor[:, 0, :] = initial_conditions[:, 0, :]

    for t in range(256):

        output, hiddens = model(last_tensor)
        last_tensor = 2*output
        result_tensor[:, t + 1, :] = output[:, 0, :]

    result = result_tensor.detach().numpy()

    np.save(filename, result)


def get_variances(array):

    result = []

    for current in array:

        mu = np.mean(current)
        shifted = current - mu

        sum = 0

        for v in shifted:
            sum += np.linalg.norm(v)**2

        variance = math.sqrt(sum)/(len(current) - 1)

        result.append(variance)

    return result


def get_variance_over_window(filename: str, window: int, mindex: int, maxdex: int):

    array = np.load(filename)

    variances = []

    for i in range(0, 257, window):

        current = array[mindex : maxdex, i : i + window]

        variances.append(get_variances(current))

    return np.array(variances)