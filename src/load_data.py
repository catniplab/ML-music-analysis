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
        in_tensor, targ_tensor = input[:-1, :], input[1:, :]

        return in_tensor, targ_tensor

    def __len__(self):
        return len(self.data)


def collate_fun(batch):

    # separate inputs and targets and record lengths of inputs
    (inputs, output) = zip(*batch)
    lengths = [len(i) for i in inputs]

    # zero-pad all tensors (all songs are of different length!)
    inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    outputs = pad_sequence(output, batch_first=True, padding_value=0)

    # keep a mask which tells us which parts of the tensor are actual data
    mask = length_to_mask(torch.tensor(lengths, dtype=torch.int, requires_grad=False),  dtype=None)

    return inputs, outputs, mask


def get_data_loader(dataset: str, batch_size: int) -> Tuple:
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

        # construct the samplers
        sampler_train = PseudoBucketSampler(train_data, batch_size)
        sampler_test = PseudoBucketSampler(test_data, batch_size)
        sampler_val = PseudoBucketSampler(val_data, batch_size)

        # construct the loaders
        train_loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler_train, collate_fn=collate_fun)
        test_loader = DataLoader(test_data, batch_size=batch_size, sampler=sampler_test, collate_fn=collate_fun)
        val_loader = DataLoader(val_data, batch_size=batch_size, sampler=sampler_val, collate_fn=collate_fun)

        return train_loader, test_loader, val_loader

    else:
        raise ValueError("Dataset {} not found.".format(dataset))