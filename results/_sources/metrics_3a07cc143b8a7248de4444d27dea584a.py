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

        # sum over notes
        tru_pos = torch.sum(prediction*target, dim=2)
        # Bay et al sum over time but this yields way higher results than Boulanger-Lewandowski
        #tru_pos = torch.sum(tru_pos, dim=1)

        # compute accuracy for all sequences at each time point
        T = output.shape[1]
        acc_over_time = []

        for t in range(T):

            # get false positives and negatives for each sequence
            false_pos = torch.sum(prediction[:, t]*(1 - target[:, t]), dim=2)
            false_neg = torch.sum((1 - prediction[:, t])*target[:, t], dim=2)

            # true negatives are unremarkable for sparse binary sequences
            this_acc = mask[:, t]*tru_pos[t]/(tru_pos[t] + false_pos + false_neg)

            acc_over_time.append(this_acc)

        # first take the average for each sequence, then sum over sequences
        result = torch.cat(acc_over_time).reshape(T, N)
        result = torch.sum(result, dim=0)/lens
        #print(result)
        result = torch.sum(result)
        return result
