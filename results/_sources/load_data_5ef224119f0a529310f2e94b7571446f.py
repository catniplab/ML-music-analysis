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

    def __init__(self, ArrayOfArrays, seq_len):
        data = ArrayOfArrays if ArrayOfArrays.ndim == 1 else ArrayOfArrays.flatten()
        data = [torch.from_numpy(d) for d in data]
        data = [torch.stack(d.split(seq_len)[0:len(d)//seq_len]) for d in data if (len(d)//seq_len)>=1]

        self.data = torch.cat(data, dim=0)

    def __getitem__(self, index):

        # get a sample from the data and match our desired datatype
        index_tensor = self.data[index].type(torch.get_default_dtype())

        # everything except the last time step for the input
        input_tensor = torch.zeros(index_tensor.shape)
        input_tensor[0 : -1, :] = index_tensor[0 : -1, :]

        # last time step for target
        target_tensor = index_tensor[-1, :]

        return input_tensor, target_tensor

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
        train_data = DatasetFromArrayOfArrays(mat_data['traindata'][0], seq_len)
        test_data = DatasetFromArrayOfArrays(mat_data['testdata'][0], seq_len)
        val_data = DatasetFromArrayOfArrays(mat_data['validdata'][0], seq_len)

        # construct the data loaders
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader, val_loader

    else:
        raise ValueError("Dataset {} not found.".format(dataset))