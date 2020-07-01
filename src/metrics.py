"""
Here we define pytorch modules for efficiently computing accuracy and loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# average binary cross entropy per time step, with a mask to indicate where the data actually is
# returns sum over all sequences, be sure to take an average after calling
class MaskedBCE(nn.Module):

    def __init__(self):
        super().__init__(MaskedBCE, self)

    def forward(self, output, target, mask):

        # binary cross entropy
        bce = nn.BCEWithLogitsLoss(reduction='sum')

        # compute for each sequence
        loss_each_seq = []

        for i in range(len(output)):

            # actual duration of the sequence
            T = torch.sum(mask[i])

            # get the particular sequence
            this_out = output[i]
            this_targ = target[i]

            # average BCE over time
            loss = bce(this_out, this_targ)/T
            loss_each_seq.append(loss)

        return torch.sum(torch.cat(loss_each_seq))


# see Bay et al 2009 for the definition of frame-level accuracy
# this module also returns the sum over all sequences
class Accuracy(nn.Module):

    def __init__(self):
        super().__init__(Accuracy, self)

    def forward(self, output, target, mask):

        prediction = (torch.sigmoid(output) > 0.5).type(torch.get_default_type())

        # sum over notes and time
        tru_pos = torch.sum(prediction*target, dim=2)
        tru_pos = torch.sum(tru_pos, dim=1)

        # lengths of each sequence
        lens = torch.sum(mask, dim=1)

        # compute for all sequences at each time point
        T = output.shape[1]
        acc_over_time = []

        for t in range(T):

            # get false positives and negatives for each sequence
            false_pos = torch.sum(prediction[:, t]*(1 - target[:, t]), dim=1)
            false_neg = torch.sum((1 - prediction[:, t])*target[:, t], dim=1)

            # this trick ensures we don't get a divide-by-zero error or NaNs
            false_pos += 1 - mask[:, t]

            # true negatives are unremarkable for sparse binary sequences
            this_acc = tru_pos/(tru_pos + false_pos + false_neg)

            acc_over_time.append(this_acc)

        # first sum over time, take the average for each sequence, then sum over sequences
        result = torch.sum(torch.cat(acc_over_time), dim=0)
        result /= lens
        result = torch.sum(result)
        return result
