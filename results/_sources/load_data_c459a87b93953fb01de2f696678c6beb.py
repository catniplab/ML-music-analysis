"""
This module provides a function which returns a pytorch DataLoader for a desired music database.
"""

import numpy as np

import torch
import torch.tensor
import torch.nn.functional as F

from scipy.io import loadmat

from typing import Tuple


def get_songs(dataset: str) -> Tuple:
    """
    :param dataset: The name of the dataset we want to use. Either JSB_Chorales, MuseData, Nottingham, or Piano_midi.
    :param set: either test or valid.
    :return: Array of arrays for training, testing, and validation.
    """

    path = "data/" + dataset + ".mat"

    train_data = None
    test_val_data = None

    if dataset in ["JSB_Chorales", "MuseData", "Nottingham", "Piano_midi"]:

        # get the data from the matlab file
        mat_data = loadmat(path)

        # construct the datasets
        train_data = mat_data['traindata'][0]
        test_data = mat_data['testdata'][0]
        val_data = mat_data['validdata'][0]

        # convert each array into a tensor and store it in a list
        train_tensors = []
        test_tensors = []
        val_tensors = []

        for arr in train_data:
            T = arr.shape[0]
            tensor = torch.tensor(arr, dtype=torch.float).reshape(1, T, 88)
            train_tensors.append(tensor)

        for arr in test_data:
            T = arr.shape[0]
            tensor = torch.tensor(arr, dtype=torch.float).reshape(1, T, 88)
            test_tensors.append(tensor)

        for arr in val_data:
            T = arr.shape[0]
            tensor = torch.tensor(arr, dtype=torch.float).reshape(1, T, 88)
            val_tensors.append(tensor)

        return train_tensors, test_tensors, val_tensors

    else:
        raise ValueError("Dataset {} not found.".format(dataset))