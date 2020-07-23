import sys
import math
import numpy as np
import sklearn.linear_model as lm
from scipy.io import loadmat

# notes 27 through 75 are the ones which are actually played

data_dict = loadmat('../data/JSB_Chorales.mat')
train_arrays = data_dict['traindata'][0]


def train_models():

    size = 0
    for array in train_arrays:
        size += len(array)

    x = np.zeros((size, 49))
    count = 0
    for array in train_arrays:
        t = len(array)
        x[count : count + t - 1] = array[0 : -1, 27 : 76]
        count += t

    y = np.zeros((size, 49))
    count = 0
    for array in train_arrays:
        t = len(array)
        y[count : count + t - 1] = array[1:, 27 : 76]
        count += t

    model_list = []

    for channel in range(49):

        model = lm.LogisticRegression(penalty='none', solver='newton-cg')

        model.fit(x, y[:, channel])

        model_list.append(model)

    return model_list


def compute_accuracy(set: str, model_list):

    arrays = data_dict[set][0]

    tot_over_seqs = 0

    for array in arrays:

        tot_over_time = 0

        for t in range(len(array) - 1):

            x = array[t, 27 : 76]
            y = array[t + 1, 27 : 76]

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
                pred = model.predict(x.reshape(1, -1))[0]
                if pred == 0.0:
                    pred += 0.00001
                if pred == 1.0:
                    pred -= 0.00001

                tot += y[channel]*math.log(pred) + (1 - y[channel])*math.log(1 - pred)

            tot_over_time += tot

        tot_over_seqs += tot_over_time/len(array)

    return tot_over_seqs/len(arrays)

