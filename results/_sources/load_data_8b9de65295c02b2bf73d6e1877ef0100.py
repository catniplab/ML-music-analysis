"""
This module provides a function which returns a pytorch DataLoader for a desired music database.
"""

import numpy as np

import torch
import torch.tensor
import torch.nn.functional as F

from torch.utils.data import TensorDataset, Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from scipy.io import loadmat

from typing import Tuple


class DatasetFromArrayOfArrays(Dataset):

    def __init__(self, ArrayOfArrays):

        # turn data into a list of arrays sorted by length
        data = ArrayOfArrays if ArrayOfArrays.ndim == 1 else ArrayOfArrays.flatten()
        data = [torch.from_numpy(d) for d in data]
        data.sort(key=len)

        self.data = data

    def __getitem__(self, index):

        ix_tensor = self.data[index].type(torch.get_default_dtype())

        # all except last time step for input
        # all except first time step for target
        in_tensor, targ_tensor = ix_tensor[0 : -1], ix_tensor[1 : ]

        return in_tensor, targ_tensor

    def __len__(self):
        return len(self.data)


def length_to_mask(length, max_len=None, dtype=None):
    """
    :param length: list of sequence lengths
    :max_len: maximum length in the list
    :dtype: desired data type for the mask
    :return: 2D tensor indexed by [sequence, time] indicating whether or not there is actual data at that index
    """

    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'

    max_len = max_len or length.max().item()

    mask = torch.arange(max_len, device=length.device, dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)

    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)

    return mask


def collate_fun(batch):

    # separate inputs and targets and record lengths of inputs
    (inputs, output) = zip(*batch)
    lengths = [len(i) for i in inputs]

    # zero-pad all tensors (all songs are of different length!)
    inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    outputs = pad_sequence(output, batch_first=True, padding_value=0)

    # keep a mask which tells us which parts of the tensor are actual data
    mask = length_to_mask(torch.tensor(lengths, requires_grad=False), dtype=torch.float)

    return inputs, outputs, mask


def get_loader(dataset: str, batch_size: int) -> Tuple:
    """
    :param dataset: The name of the dataset we want to use. Either JSB_Chorales, MuseData, Nottingham, or Piano_midi.
    :param batch_size: how many sequences to train on at once.
    :return: DataLoaders for training, testing, and validation.
    """

    path = "data/" + dataset + ".mat"

    train_data = None
    test_val_data = None

    if dataset in ["JSB_Chorales", "MuseData", "Nottingham", "Piano_midi"]:

        # read the matlab file
        data = loadmat(path)

        # construct the data sets
        train_data = DatasetFromArrayOfArrays(data['traindata'][0])
        test_data = DatasetFromArrayOfArrays(data['testdata'][0])
        val_data = DatasetFromArrayOfArrays(data['validdata'][0])

        # construct the loaders
        train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fun)
        test_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=collate_fun)
        val_loader = DataLoader(val_data, batch_size=batch_size, collate_fn=collate_fun)

        return train_loader, test_loader, val_loader

    else:
        raise ValueError("Dataset {} not found.".format(dataset))