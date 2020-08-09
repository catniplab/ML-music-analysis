import sys
import math
import numpy as np
import sklearn.linear_model as lm

from scipy.io import loadmat

from tqdm import tqdm

# For JSB_Chorales, notes 27 through 75 are the ones which are actually played


def get_dataset(dataname: str, key: str, lag=1, window=1, format='flattened'):
    """
    :param dataname: which dataset is to be used
    :param key: 'traindata', 'testdata', 'validdata'
    :param lag: how many steps into the future are we predicting
    :param window: how many steps are we predicting
    """

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

        newx = np.zeros((T - offset, 48*window))
        for t in range(T - offset):
            for i in range(window):
                newx[t, 48*i : 48*(i + 1)] = array[t + i, 27 : 75]
        xlist.append(newx)

        ylist.append(array[offset:, 27 : 75])

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
        x = np.zeros((size, 48*window))
        y = np.zeros((size, 48))

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


def train_models(dataname: str, _seed, lag=1, window=1):
    """
    :param dataname: which dataset to use for training
    :param lag: how many steps into the future are we predicting
    :param window: how many steps are we predicting
    """

    # load the data
    x, y = get_dataset(dataname, 'traindata', lag=lag, window=window)

    # model is needed for every channel (note)
    model_list = []

    # train every model
    for channel in tqdm(range(48)):

        model = lm.LogisticRegression(solver='saga', penalty='elasticnet', l1_ratio=0.9, random_state=_seed)

        model.fit(x, y[:, channel])

        model_list.append(model)

    return model_list


def compute_accuracy(model_list, dataname: str, key: str, lag=1, window=1):
    """
    :param model_list: the trained regression model for every note
    :param dataname: dataname of the dataset to be used
    :param key: 'traindata', 'testdata', 'validdata'
    :param lag: how many steps into the future are we predicting
    :param window: how many steps are we predicting
    """

    # load the data
    x, y = get_dataset(dataname, key, lag=lag, window=window, format='listofarrays')

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
            for channel in range(48):

                # get the appropriate model and prediction
                model = model_list[channel]
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


def compute_loss(model_list, dataname: str, key: str, lag=1, window=1):
    """
    :param model_list: the trained regression model for every note
    :param dataname: dataname of the dataset to be used
    :param key: 'traindata', 'testdata', 'validdata'
    :param lag: how many steps into the future are we predicting
    :param window: how many steps are we predicting
    """

    # load the data
    x, y = get_dataset(dataname, key, lag=lag, window=window, format='listofarrays')

    # accumulate loss over all sequences
    tot_over_seqs = 0

    for xarr, yarr in tqdm(zip(x, y)):

        # accumulate loss over time
        tot_over_time = 0

        for xt, yt in zip(xarr, yarr):

            # accumulate over each note
            tot = 0

            for channel in range(48):

                model = model_list[channel]

                # sigmoid of the trained affine transformation
                pred = 1.0/(1.0 + np.exp(-(model.coef_ @ xt + model.intercept_)))

                # binary cross entropy with logits
                tot -= yt[channel]*math.log(pred) + (1 - yt[channel])*math.log(1 - pred)

            tot_over_time += tot

        tot_over_seqs += tot_over_time/len(xarr)

    return tot_over_seqs/len(x)

