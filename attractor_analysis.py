"""
For figuring out whether or not neural networks possess attractors, and where they are.
"""

import torch
import numpy as np

from src.neural_nets.models import get_model

def generate_trajectories(model_dict: dict, sd_path: str, filename: str):

    model = get_model(model_dict, {'init': "default"}, False)
    sd = torch.load(sd_path)
    model.load_state_dict(sd)

    initial_conditions = torch.randn((8192, 1, 88))
    last_tensor = initial_conditions

    result_tensor = torch.zeros((8192, 256, 88))
    result_tensor[:, 0, :] = initial_conditions

    for t in range(256):

        output, hiddens = model(last_tensor)
