import sys
import math
import numpy as np
import sklearn.linear_model as lm

from scipy.io import loadmat

from tqdm import tqdm

# For JSB_Chorales, notes 27 through 75 are the ones which are actually played
# For Nottingham, it is 10 through 72, with some missing in between

def get_dataset(dataname: str,
                key: str,
                low_off_notes: int,
                high_off_notes: int,
                lag=1,
                window=1,
                format='flattened'):
    """
    :param dataname: which dataset is to be used
    :param key: 'traindata', 'testdata', 'validdata'
    :param lag: how many steps into the future are we predicting
    :param window: how many steps are we predicting
    """

    num_notes = high_off_notes - low_off_notes

    data_dict = loadmat('data/' + dataname)
    arrays = data_dict[key][0]

    # this much will have to be chopped off at the beginning and end of each sequence
    offset = lag + window - 1

    # store sequences separately here
    xlist = []
    ylist = []

    # record each array, reformatted appropriately
    for array in arrays:

        T = len(array)

        newx = np.zeros((T - offset, num_notes*window))
        for t in range(T - offset):
            for i in range(window):
                newx[t, num_notes*i : num_notes*(i + 1)] = array[t + i, low_off_notes : high_off_notes]
        xlist.append(newx)

        ylist.append(array[offset:, low_off_notes : high_off_notes])

    # this format is needed for computing average loss and accuracy over time and sequences
    if format == 'listofarrays':

        return xlist, ylist

    # this format is needed for training
    elif format == 'flattened':

        # count how big the whole array needs to be
        size = 0
        for xseq in xlist:
            size += len(xseq)

        # initialize the flattened inputs and targets
        x = np.zeros((size, num_notes*window))
        y = np.zeros((size, num_notes))

        # keep track of where we are
        ix = 0

        # put every sequence together into one array
        for xseq, yseq in zip(xlist, ylist):

            T = len(xseq)

            x[ix : ix + T] = xseq
            y[ix : ix + T] = yseq

            ix += T

        return x, y

    else:
        raise ValueError("Format {} not recognized".format(format))


# some of the notes might be off the entire time, find them!
def find_off_notes(x):

    off_notes = []

    num_notes = x.shape[1]

    for note in range(num_notes):

        if not 1 in x[:, note]:
            off_notes.append(note)

    return off_notes


def train_models(dataname: str,
                 num_epochs: int,
                 low_off_notes: int,
                 high_off_notes: int,
                 _seed,
                 lag=1,
                 window=1):
    """
    :param dataname: which dataset to use for training
    :param lag: how many steps into the future are we predicting
    :param window: how many steps are we predicting
    """

    num_notes = high_off_notes - low_off_notes

    # load the data
    x, y = get_dataset(dataname,
                      'traindata',
                      low_off_notes,
                      high_off_notes,
                      lag=lag,
                      window=window)

    off_notes = find_off_notes(x)

    # model is needed for every channel (note)
    model_list = []

    # train every model
    for channel in tqdm(range(num_notes)):

        # append a placeholder to the model list if this note is not played
        if channel in off_notes:
            model_list.append(None)

        # otherwise train the model on this particular note and append it
        else:
            model = lm.LogisticRegression(solver='saga', penalty='elasticnet', l1_ratio=0.9,     random_state=_seed, max_iter=num_epochs)

            model.fit(x, y[:, channel])

            model_list.append(model)

    return model_list


def compute_accuracy(model_list,
                     dataname: str,
                     key: str,
                     low_off_notes: int,
                     high_off_notes: int,
                     lag=1,
                     window=1):
    """
    :param model_list: the trained regression model for every note
    :param dataname: dataname of the dataset to be used
    :param key: 'traindata', 'testdata', 'validdata'
    :param lag: how many steps into the future are we predicting
    :param window: how many steps are we predicting
    """

    # how many notes we are predicting
    num_notes = len(model_list)

    # load the data
    x, y = get_dataset(dataname,
                       key,
                       low_off_notes,
                       high_off_notes,
                       lag=lag,
                       window=window,
                       format='listofarrays')

    # accumulate accuracy over all sequences
    tot_over_seqs = 0

    for xarr, yarr in tqdm(zip(x, y)):

        # accumulate accuracy over time
        tot_over_time = 0

        for xt, yt in zip(xarr, yarr):

            # true positives, false positives, false negatives
            tp = 0
            fp = 0
            fn = 0

            # compute for each note
            for channel in range(num_notes):

                # get the appropriate model and prediction
                model = model_list[channel]

                if model != None:
                    pred = model.predict(xt.reshape(1, -1))[0]

                    tp += yt[channel]*pred
                    fp += (1 - yt[channel])*pred
                    fn += yt[channel]*(1 - pred)

            # avoid nans
            if tp == 0 and fp == 0 and fn == 0:
                tot_over_time += 0
            else:
                tot_over_time += tp/(tp + fp + fn)

        tot_over_seqs += tot_over_time/len(xarr)

    return tot_over_seqs/len(x)


def compute_loss(model_list,
                dataname: str,
                key: str,
                low_off_notes: int,
                high_off_notes: int,
                lag=1,
                window=1):
    """
    :param model_list: the trained regression model for every note
    :param dataname: dataname of the dataset to be used
    :param key: 'traindata', 'testdata', 'validdata'
    :param lag: how many steps into the future are we predicting
    :param window: how many steps are we predicting
    """

    # how many notes we are predicting
    num_notes = len(model_list)

    # load the data
    x, y = get_dataset(dataname,
                       key,
                       low_off_notes,
                       high_off_notes,
                       lag=lag,
                       window=window,
                       format='listofarrays')

    # accumulate loss over all sequences
    tot_over_seqs = 0

    for xarr, yarr in tqdm(zip(x, y)):

        # accumulate loss over time
        tot_over_time = 0

        for xt, yt in zip(xarr, yarr):

            # accumulate over each note
            tot = 0

            for channel in range(num_notes):

                model = model_list[channel]

                if model != None:

                    # sigmoid of the trained affine transformation
                    pred = 1.0/(1.0 + np.exp(-(model.coef_ @ xt + model.intercept_)))

                    # binary cross entropy with logits
                    tot -= yt[channel]*math.log(pred) + (1 - yt[channel])*math.log(1 - pred)

            tot_over_time += tot

        tot_over_seqs += tot_over_time/len(xarr)

    return tot_over_seqs/len(x)

