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
        super(MaskedBCE, self).__init__()

    def forward(self, output, target, mask):

        # binary cross entropy
        bce = nn.BCEWithLogitsLoss(reduction='sum')

        # compute for each sequence
        loss_each_seq = []

        # average over time is different for each sequence
        for i in range(len(output)):

            # actual duration of the sequence
            T = torch.sum(mask[i]).detach().item()

            # get the particular sequence
            this_out = output[i, 0 : T]
            this_targ = target[i, 0 : T]

            # average BCE over time
            loss = bce(this_out, this_targ)/float(T)
            loss = loss.reshape((1)) # pytorch shapes are annoying
            #print(loss)
            loss_each_seq.append(loss)

        return torch.sum(torch.cat(loss_each_seq))


# see Bay et al 2009 for the definition of frame-level accuracy
# this module also returns the sum over all sequences
class Accuracy(nn.Module):

    def __init__(self):
        super(Accuracy, self).__init__()

    def forward(self, output, target, mask):

        N = output.shape[0]

        prediction = (torch.sigmoid(output) > 0.5).type(torch.get_default_dtype())

        # sum over notes and time
        tru_pos = torch.sum(prediction*target, dim=2)
        tru_pos = torch.sum(tru_pos, dim=1)

        # get false positives and negatives for each sequence
        false_pos = torch.sum(prediction*(1 - target), dim=2)
        false_neg = torch.sum((1 - prediction)*target, dim=2)

        tot_acc = torch.zeros(N)

        # compute accuracy for all sequences at each time point
        for i in range(N):

            T = torch.sum(mask[i]).detach().item()

            acc = 0

            for t in range(T):
                acc += tru_pos[i]/(tru_pos[i] + false_pos[i, t] + false_neg[i, t])

            acc /= T
            tot_acc[i] = acc

        return torch.sum(tot_acc)