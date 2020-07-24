import sys
import math
import numpy as np
import sklearn.linear_model as lm

from scipy.io import loadmat

from sklearn.linear_model.logistic import _logistic_loss

# For JSB_Chorales, notes 27 through 75 are the ones which are actually played


def get_dataset(dataname: str, key: str, lag=1, window=1, format='flattened'):
    """
    :param dataname: which dataset is to be used
    :param key: 'traindata', 'testdata', 'validdata'
    :param lag: how many steps into the future are we predicting
    :param window: how many steps are we predicting
    """

    data_dict = loadmat(dataname)
    arrays = data_dict[key][0]

    # this much will have to be chopped off at the beginning and end of each sequence
    offset = lag + window

    # store sequences separately here
    xlist = []
    ylist = []

    # record each array, reformatted appropriately
    for array in arrays:

        T = len(array)

        newx = np.zeros((T - offset, 49*window))
        for t in range(T - offset):
            for i in range(window):
            newx[t, 49*i : 49*(i + 1)] = array[t + i]
        xlist.append(newx)

        ylist.append(array[offset:])

    # this format is needed for computing average loss and accuracy over time and sequences
    if format == 'listofarrays':

        return xlist, ylist

    # this format is needed for training
    elif format == 'flattened':

        # count how big the whole array needs to be
        size = 0
        for xseq in xlist:
            size += len(x)

        # initialize the flattened inputs and targets
        x = np.zeros((size, 49*window))
        y = np.zeros((size, 49))

        # keep track of where we are
        ix = 0

        # put every sequence together into one array
        for xseq, yseq in zip(xlist, ylist):

            T = len(xlist)

            x[ix : ix + T] = xseq
            y[ix : ix + T] = yseq

            ix += T

        return x, y

    else:
        raise ValueError("Format {} not recognized".format(format))


def train_models(dataname: str, lag=1, window=1):
    """
    :param dataname: which dataset to use for training
    :param lag: how many steps into the future are we predicting
    :param window: how many steps are we predicting
    """

    # load the data
    x, y = get_dataset(dataset, 'traindata', lag=lag, window=window)

    # model is needed for every channel (note)
    model_list = []

    # train every model
    for channel in tqdm(range(49)):

        model = lm.LogisticRegression(solver='saga', penalty='elasticnet', l1_ratio=0.91, random_state=24)

        model.fit(x, y[:, channel])

        model_list.append(model)

    return model_list


def compute_accuracy(model_list, dataname: str, set: str, lag=1, window=1):
    """
    :param model_list: the trained regression model for every note
    :param dataname: dataname of the dataset to be used
    :param set: 'traindata', 'testdata', 'validdata'
    :param lag: how many steps into the future are we predicting
    :param window: how many steps are we predicting
    """

    # load the data
    x, y = get_dataset(dataname, set, lag=lag, window=window, format='listofarrays')

    # accumulate accuracy over all sequences
    tot_over_seqs = 0

    for xarr, yarr in zip(x, y):

        # accumulate accuracy over time
        tot_over_time = 0

        for t in range(len(xarr) - 1):

            # input and target for current time step
            xt = xarr[t]
            yt = yarr[t + 1]

            # true positives, false positives, false negatives
            tp = 0
            fp = 0
            fn = 0

            # compute for each note
            for channel in range(49):

                # get the appropriate model and prediction
                model = model_list[channel]
                pred = model.predict(xt.reshape(1, -1))[0]

                tp += y[channel]*pred
                fp += (1 - y[channel])*pred
                fn += y[channel]*(1 - pred)

            # avoid nans
            if tp == 0 and fp == 0 and fn == 0:
                tot_over_time += 0
            else:
                tot_over_time += tp/(tp + fp + fn)

        tot_over_seqs += tot_over_time/len(array)

    return tot_over_seqs/len(arrays)


def compute_loss(model_list, dataname: str, set: str, lag=1, window=1):
    """
    :param model_list: the trained regression model for every note
    :param dataname: dataname of the dataset to be used
    :param set: 'traindata', 'testdata', 'validdata'
    :param lag: how many steps into the future are we predicting
    :param window: how many steps are we predicting
    """

    # load the data
    x, y = get_dataset(dataname, set, lag=lag, window=window, format='listofarrays')

    # accumulate loss over all sequences
    tot_over_seqs = 0

    for xarr, yarr in zip(x, y):

        # accumulate loss over time
        tot_over_time = 0

        for t in range(len(array) - 1):

            # input and target for current time step
            xt = xarr[t]
            yt = yarr[t + 1]

            # accumulate over each note
            tot = 0

            for channel in range(49):

                model = model_list[channel]

                # sigmoid of the trained affine transformation
                pred = 1.0/(1.0 + np.exp(-(model.coef_ @ x + model.intercept_)))

                # binary cross entropy with logits
                tot -= y[channel]*math.log(pred) + (1 - y[channel])*math.log(1 - pred)

            tot_over_time += tot

        tot_over_seqs += tot_over_time/len(array)

    return tot_over_seqs/len(arrays)

