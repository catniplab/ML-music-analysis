"""
Here we define pytorch modules for efficiently computing accuracy and loss.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

# average binary cross entropy per time step, with a mask to indicate where the data actually is
# returns sum over all sequences, be sure to take an average after calling
class MaskedBCE(nn.Module):

    def __init__(self, regularization: float, low_off_notes=0, high_off_notes=88, p='fro'):

        super(MaskedBCE, self).__init__()

        # coefficient for regularization
        self.regular = regularization

        # which norm to use for regularization
        self.p = p

        self.low = low_off_notes
        self.high = high_off_notes

    def forward(self, output, target, mask, model):

        # binary cross entropy
        bce = nn.BCEWithLogitsLoss(reduction='sum')

        # compute for each sequence
        loss_each_seq = []

        # average over time is different for each sequence
        for i in range(len(output)):

            # actual duration of the sequence
            T = torch.sum(mask[i]).detach().item()
            Ti = int(T)

            # get the particular sequence
            this_out = output[i, 0 : Ti, self.low : self.high]
            this_targ = target[i, 0 : Ti, self.low : self.high]

            # average BCE over time
            loss = bce(this_out, this_targ)/T
            loss = loss.reshape((1)) # pytorch shapes are annoying
            #print(loss)
            loss_each_seq.append(loss)

        result = torch.sum(torch.cat(loss_each_seq))

        # regularization
        if self.regular > 0:
            regular_term = []
            for p in model.parameters():
                regular_term.append(torch.norm(p.data, p=self.p).reshape((1)))
            return result + self.regular*torch.sum(torch.cat(regular_term))

        return result


# see Bay et al 2009 for the definition of frame-level accuracy
# this module also returns the sum over all sequences
class Accuracy(nn.Module):

    def __init__(self, low_off_notes=0, high_off_notes=88):

        super(Accuracy, self).__init__()

        self.low = low_off_notes
        self.high = high_off_notes

    def forward(self, output, target, mask):

        N = output.shape[0]

        prediction = (torch.sigmoid(output) > 0.5).type(torch.get_default_dtype())[:, :, self.low : self.high]

        target = target[:, :, self.low : self.high]

        #print(prediction)
        #print(target)

        # sum over notes
        tru_pos = torch.sum(prediction*target, dim=2)
        # Bay et al sum over time but this yields way higher results than Boulanger-Lewandowski
        #tru_pos = torch.sum(tru_pos, dim=1)

        # compute accuracy for all sequences at each time point
        T = output.shape[1]
        acc_over_time = []

        # actual lengths of each sequence
        lens = torch.sum(mask, dim=1)

        for t in range(T):

            # get false positives and negatives for each sequence
            false_pos = torch.sum(prediction[:, t]*(1 - target[:, t]), dim=1)
            false_neg = torch.sum((1 - prediction[:, t])*target[:, t], dim=1)

            # quick trick to try to avoid NaNs
            false_pos += F.relu(1 - tru_pos[:, t])

            # true negatives are unremarkable for sparse binary sequences
            this_acc = mask[:, t]*tru_pos[:, t]/(tru_pos[:, t] + false_pos + false_neg)

            acc_over_time.append(this_acc)

        # first take the average for each sequence, then sum over sequences
        result = torch.cat(acc_over_time).reshape(T, N)
        result = torch.sum(result, dim=0)/lens
        result = torch.sum(result)

        return result


def compute_loss(loss_fcn: MaskedBCE, model: nn.Module, loader: DataLoader):
    """
    :param loss_fcn: instance of MasdeBCE, whose average value over the dataset we will compute
    :param model: the model whose loss we are computing
    :param loader: data loader containing all appropriate inputs, targets, and masks
    """

    # record loss for all sequences and count the total number of sequences
    all_loss = []
    num_seqs = 0

    for input_tensor, target_tensor, mask in loader:

        num_seqs += input_tensor.shape[0]

        output, hiddens = model(input_tensor)

        loss = loss_fcn(output, target_tensor, mask, model)
        all_loss.append(loss.cpu().detach().item())

    # return the average loss across every sequence
    # np.mean will NOT work here because the sum over multiple sequences is included in each entry
    avg = np.sum(all_loss)/num_seqs
    return avg


def compute_acc(model: nn.Module, loader: DataLoader, low=0, high=88):
    """
    :param model: model which we are testing
    :param loader: loader containing all appropriate inputs, targets, and masks
    :return: average accuracy for every song in the loader
    """

    acc_fcn = Accuracy(low_off_notes=low, high_off_notes=high)

    # record accuracy for all sequences and count the total number of sequences
    all_acc = []
    num_seqs = 0

    for input_tensor, target_tensor, mask in loader:

        num_seqs += input_tensor.shape[0]

        output, hiddens = model(input_tensor)

        acc = acc_fcn(output, target_tensor, mask)
        all_acc.append(acc)

    # log the average accuracy across every sequence
    # np.mean will NOT work here because the sum over multiple sequences is included in each entry
    avg = np.sum(all_acc)/num_seqs
    return avg