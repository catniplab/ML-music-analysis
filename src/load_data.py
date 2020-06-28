"""
This module provides a function which returns a pytorch DataLoader for a desired music database.
"""

import numpy as np

import torch
import torch.tensor
import torch.nn.functional as F

from torch.utils.data import TensorDataset, Dataset, DataLoader

from scipy.io import loadmat

from typing import Tuple

# TODO
# Document DatasetFromArrayOfArrays

class DatasetFromArrayOfArrays(Dataset):

    def __init__(self, ArrayOfArrays):

        max_len = 0
        for array in ArrayOfArrays:
            if len(array) > max_len:
                max_len = len(array)

        data_tensor = torch.zeros((len(ArrayOfArrays), max_len, 88), dtype=torch.float)

        for i, array in enumerate(ArrayOfArrays):
            la = len(array)
            data_tensor[i, 0 : la, :] = torch.tensor(array, dtype=torch.float)

        self.data = data_tensor

    def __getitem__(self, index):

        # get a sample from the data and match our desired datatype
        index_tensor = self.data[index]

        return index_tensor, index_tensor

    def __len__(self):
        return len(self.data)


def get_data_loader(dataset: str, batch_size: int) -> Tuple:
    """
    :param dataset: The name of the dataset we want to use. Either JSB_Chorales, MuseData, Nottingham, or Piano_midi.
    :param set: either test or valid.
    :return: DataLoaders for training, testing, and validation.
    """

    path = "data/" + dataset + ".mat"

    train_data = None
    test_val_data = None

    if dataset in ["JSB_Chorales", "MuseData", "Nottingham", "Piano_midi"]:

        # get the data from the matlab file
        mat_data = loadmat(path)

        # figure out length of time sequences
        seq_len = mat_data['traindata'][0, 1].shape[0]

        # construct the datasets
        train_data = DatasetFromArrayOfArrays(mat_data['traindata'][0])
        test_data = DatasetFromArrayOfArrays(mat_data['testdata'][0])
        val_data = DatasetFromArrayOfArrays(mat_data['validdata'][0])

        # construct the data loaders
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader, val_loader

    else:
        raise ValueError("Dataset {} not found.".format(dataset))