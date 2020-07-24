import sys
import math
import numpy as np
import sklearn.linear_model as lm

from scipy.io import loadmat

from sklearn.linear_model.logistic import _logistic_loss

# notes 27 through 75 are the ones which are actually played

def get_dataset(name: str, set: str, lag: int, window: int, format='listofarrays'):

    data_dict = loadmat(name)
    arrays = data_dict[set][0]

    offset = lag + window

    xlist = []
    ylist = []

    for array in arrays:

        T = len(array)

        xlist.append(array[0 : T - offset])
        ylist.append(array[offset:])

    if format == 'listofarrays':

        return xlist, ylist

    elif format == 'flattened':

        size = 0
        for xseq in xlist:
            size += len(x)

        x = np.zeros((size, 49))
        y = np.zeros((size, 49))

        ix = 0

        for xseq, yseq in zip(xlist, ylist):

            T = len(xlist)

            x[ix : ix + T] = xseq
            y[ix : ix + T] = yseq

            ix += T

        return x, y


    else:
        raise ValueError("Format {} not recognized".format(format))

    T = arrays.shape[1]

    x = arrays[:, T - offset, 27 : 76]
    y = arrays[:, offset:, 27 : 76]

    return x, y


def train_models(dataset: str, lag: int, window: int):

    x, y = get_dataset(dataset, 'traindata', lag, window)

    model_list = []

    for channel in range(49):

        model = lm.LogisticRegression(solver='saga', penalty='elasticnet', l1_ratio=0.91, random_state=24)

        model.fit(x, y[:, channel])

        model_list.append(model)

    return model_list


def compute_accuracy(set: str, model_list, lag: int, window: int):

    arrays = data_dict[set][0]

    tot_over_seqs = 0

    for array in arrays:

        tot_over_time = 0

        for t in range(len(array) - 1):

            xt = array[t]
            yt = array[t + 1]

            tp = 0
            fp = 0
            fn = 0

            for channel in range(49):

                model = model_list[channel]
                pred = model.predict(x.reshape(1, -1))[0]

                tp += y[channel]*pred
                fp += (1 - y[channel])*pred
                fn += y[channel]*(1 - pred)

            if tp == 0 and fp == 0 and fn == 0:
                tot_over_time += 0
            else:
                tot_over_time += tp/(tp + fp + fn)

        tot_over_seqs += tot_over_time/len(array)

    return tot_over_seqs/len(arrays)


def compute_loss(set: str, model_list):

    arrays = data_dict[set][0]

    tot_over_seqs = 0

    for array in arrays:

        tot_over_time = 0

        for t in range(len(array) - 1):

            x = array[t, 27 : 76]
            y = array[t + 1, 27 : 76]

            tot = 0

            for channel in range(49):

                model = model_list[channel]
                pred = 1.0/(1.0 + np.exp(-(model.coef_ @ x + model.intercept_)))

                tot -= y[channel]*math.log(pred) + (1 - y[channel])*math.log(1 - pred)

            tot_over_time += tot

        tot_over_seqs += tot_over_time/len(array)

    return tot_over_seqs/len(arrays)

