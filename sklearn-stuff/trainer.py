import sys
import numpy as np
import sklearn.linear_model as lm
from scipy.io import loadmat

data_dict = loadmat('../data/JSB_Chorales.mat')
train_arrays = data_dict['traindata'][0]

# notes 27 through 75 are the ones which are actually played

def train_models():

    model_list = []

    for channel in range(49):

        model = lm.LogisticRegression()

        for array in train_arrays:

            x = array[0 : -1, 27 : 76]
            y = array[1:, 27 + channel]

            if 1.0 in y:
                model.fit(x, y)

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
    raise NotImplementedError