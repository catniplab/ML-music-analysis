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

def compute_accuracy():
    raise NotImplementedError

def compute_loss():
    raise NotImplementedError