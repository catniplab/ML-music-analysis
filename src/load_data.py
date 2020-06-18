"""
This script is called to create a DataLoader for the specific dataset we want to use.
"""

import numpy as np

import torch
import torch.tensor
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset

from scipy.io import loadmat

# TODO: investigate MATLAB files to ensure data is being loaded properly, and figure out what the hell is wrong with JSB_Chorales

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


def get_data_loader(dataSetName: str, set: str, batch_size: int):
    """
    :param dataSetName: The name of the data set we want to use. Either JSB_Chorales, MuseData, Nottingham, or Piano_midi.
    :param set: either test or valid.
    :return: DataLoader for our desired data set.
    """

    path = "../data/" + dataSetName + ".mat"

    train_data = None
    test_val_data = None

    if dataSetName == "MuseData":

        mat_data = loadmat(path)
        train_data = DatasetFromArrayOfArrays(mat_data['traindata'][0])
        if set == 'test':
            test_val_data = DatasetFromArrayOfArrays(mat_data['testdata'][0])
        elif set == 'valid':
            test_val_data = DatasetFromArrayOfArrays(mat_data['validdata'][0])
        else:
            raise RuntimeError("set {} not supported".format(set))

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_val_loader = DataLoader(test_val_data, batch_size=batch_size, shuffle=False)

        return train_loader, test_val_loader

    elif dataSetName == "Nottingham":

        mat_data = loadmat(path)
        train_data = DatasetFromArrayOfArrays(mat_data['traindata'][0])
        if set == 'test':
            test_val_data = DatasetFromArrayOfArrays(mat_data['testdata'][0])
        elif set == 'valid':
            test_val_data = DatasetFromArrayOfArrays(mat_data['validdata'][0])
        else:
            raise RuntimeError("set {} not supported".format(set))

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_val_loader = DataLoader(test_val_data, batch_size=batch_size, shuffle=False)

        return train_loader, test_val_loader

    elif dataSetName == "Piano_midi":

        mat_data = loadmat(path)
        train_data = DatasetFromArrayOfArrays(mat_data['traindata'][0])
        if set == 'test':
            test_val_data = DatasetFromArrayOfArrays(mat_data['testdata'][0])
        elif set == 'valid':
            test_val_data = DatasetFromArrayOfArrays(mat_data['validdata'][0])
        else:
            raise RuntimeError("set {} not supported".format(set))

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_val_loader = DataLoader(test_val_data, batch_size=batch_size, shuffle=False)

        return train_loader, test_val_loader

    elif dataSetName == "JSB_Chorales":

        mat_data = loadmat(path)
        train_data = DatasetFromArrayOfArrays(mat_data['traindata'][0])
        if set == 'test':
            test_val_data = DatasetFromArrayOfArrays(mat_data['testdata'][0])
        elif set == 'valid':
            test_val_data = DatasetFromArrayOfArrays(mat_data['validdata'][0])
        else:
            raise RuntimeError("set {} not supported".format(set))

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_val_loader = DataLoader(test_val_data, batch_size=batch_size, shuffle=False)

        return train_loader, test_val_loader

    else:
        raise ValueError("Data set {} not found.".format(dataSetName))