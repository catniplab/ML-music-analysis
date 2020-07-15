import sys
import numpy as np
import sklearn.linear_model as lm
from scipy.io import loadmat

data_dict = loadmat('../data/JSB_Chorales.mat')
train_arrays = data_dict['traindata'][0]

# notes 27 through 75 are the ones which are actually played


size = 0
for array in train_arrays:
    size += len(array) - 1

X = np.zeros((size, 49))
ix = 0
for array in train_arrays:
    for x in array[0 : -1]:
        X[ix] = x[27 : 76]
        ix += 1

Y = np.zeros((size, 49))
ix = 0
for array in train_arrays:
    for y in array[1:]:
        Y[ix] = y[27 : 76]
        ix += 1

#for i in range(49):
#    if not 1.0 in X[:, i]:
#        print(i)

model_list = []

for i in range(49):
    this_model = lm.LogisticRegression()
    this_model.fit(X, Y[:, i])
    model_list.append(this_model)

def compute_accuracy(set: str):

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
                tot_over_time += 1
            else:
                tot_over_time += tp/(tp + fp + fn)

        tot_over_seqs += tot_over_time/len(array)

    return tot_over_seqs/len(arrays)


def compute_loss():
    raise NotImplementedError