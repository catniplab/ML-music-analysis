"""
This module provides a function which returns a pytorch DataLoader for a desired music database.
"""

import numpy as np

import torch
import torch.tensor
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataSet, DataLoader

from scipy.io import loadmat

# TODO
# Investigate MATLAB files to ensure data is being loaded properly, and figure out what the hell is wrong with JSB_Chorales
# Document DatasetFromArrayOfArrays

class DatasetFromArrayOfArrays(Dataset):

    def __init__(self, ArrayOfArrays, seq_len=200):
        data = ArrayOfArrays if ArrayOfArrays.ndim == 1 else ArrayOfArrays.flatten()
        data = [torch.from_numpy(d) for d in data]
        data = [torch.stack(d.split(seq_len)[0:len(d)//seq_len]) for d in data if (len(d)//seq_len)>=1]

        self.data = torch.cat(data, dim=0)

    def __getitem__(self, index):
        input = self.data[index].type(torch.get_default_dtype())
        input, target = input[:-1, :], input[ 1:, :]
        return input, target.permute([1,0])  # batch size x num classes x ...

    def __len__(self):
        return len(self.data)


def get_data_loader(name: str, set: str, batch_size: int):
    """
    :param name: The name of the data set we want to use. Either JSB_Chorales, MuseData, Nottingham, or Piano_midi.
    :param set: either test or valid.
    :return: DataLoaders for training, testing, and validation.
    """

    path = "../data/" + name + ".mat"

    train_data = None
    test_val_data = None

    if name in ["JSB_Chorales", "MuseData", "Nottingham", "Piano_midi"]:

        # get the data from the matlab file
        mat_data = loadmat(path)

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
        raise ValueError("Data set {} not found.".format(name))