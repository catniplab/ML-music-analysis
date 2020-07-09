import numpy as np
import numpy.linalg as la
import scipy.io as io
import torch
import json
import query_results as qr
import subprocess

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Custom colormap
cdict = {'red': ((0.0, 0.0, 0.0),
                 (0.5, 0.0, 0.0),
                 (1.0, 1.0, 1.0)),
         'green': ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),
         'blue': ((0.0, 1.0, 1.0),
                  (0.5, 0.0, 0.0),
                  (1.0, 0.0, 0.0))}
mymap = LinearSegmentedColormap('MyMap', cdict)
plt.register_cmap(cmap=mymap)

def plot_scalar(dir: str, name: str):
    """
    :param dir: directory of the file storage system whose results we are looking at
    :param name: name of the scalar metric we want to visualize
    """

    path = 'results/' + dir + '/'
    handle = open(path + 'metrics.json')
    content = handle.read()
    handle.close()

    json_dict = json.loads(content)
    values = json_dict[name]['values']

    plt.plot(values)
    plt.title(name + ' ' + dir)
    plt.show()


def plot_hidden_weights(dir: str, dict_name: str, param_name:str, vmin: float, vmax: float):
    """
    :param dir: directory of the file storage system whose results we are looking at
    :param dict_name: name of the .pt file whose .weight_hh_l0.weight we will visualize
    :param param_name: name of the key in the dictionary we are interested in.
    :param vmin: expected minimum weight
    :param vmax: expected maximum weight
    """

    path = 'results/' + dir + '/'

    sd = torch.load(path + dict_name, map_location='cpu')
    hidden_weights = sd[param_name].detach().numpy()

    #plt.title(name + ' weights ' + dir)
    fig, ax = plt.subplots()
    ax.pcolor(hidden_weights, vmin=vmin, vmax=vmax, cmap='MyMap')
    ax.set_aspect('equal')
    fig.show()
    plt.gca().invert_yaxis()


def plot_eigs(dir: str, name: str, lim: float):
    """
    :param dir: directory of the file storage system whose results we are looking at
    :param name: name of the .pt file whose .weight_hh_l0.weight eigenvalues we will visualize
    :param lim: how large is the square defining the plot
    """

    path = 'results/' + dir + '/'

    sd = torch.load(path + name, map_location='cpu')
    hidden_weights = sd['rnn.weight_hh_l0.weight'].detach().numpy()

    vals, vecs = la.eig(hidden_weights)

    fig, ax = plt.subplots()
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.scatter(np.real(vals), np.imag(vals))
    ax.set_aspect('equal')
    fig.show()


def get_metrics(dirs: str, metric: str):
    """
    :param dirs: list of directories for which we will look for the final metric
    :param metric: name of the metric we are going to plot
    :return: list of metrics after training
    """

    result = []

    for name in dirs:

        handle = open('results/' + name + '/metrics.json')
        my_dict = json.loads(handle.read())
        handle.close()

        result.append(my_dict[metric]['values'][-1])

    return result


def get_all_metrics(list_of_configs):

    train_loss = []
    test_loss = []
    valid_loss = []

    train_acc = []
    test_acc = []
    valid_acc = []

    for config_dict in list_of_configs:

        dirs = qr.find_results(config_dict)

        trainLoss = np.mean(get_metrics(dirs, "trainLoss"))
        testLoss = np.mean(get_metrics(dirs, "testLoss"))
        validLoss = np.mean(get_metrics(dirs, "validLoss"))

        train_loss.append(trainLoss)
        test_loss.append(testLoss)
        valid_loss.append(validLoss)

        trainAcc = np.mean(get_metrics(dirs, "trainAccuracy"))
        testAcc = np.mean(get_metrics(dirs, "testAccuracy"))
        validAcc = np.mean(get_metrics(dirs, "validAccuracy"))

        train_acc.append(trainAcc)
        test_acc.append(testAcc)
        valid_acc.append(validAcc)

    return train_loss, test_loss, valid_loss, train_acc, test_acc, valid_acc


def make_bar(labels, title, train, test, validate):

    x = 2.0*np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, train, width, label='Train')
    rects2 = ax.bar(x, test, width, label='Test')
    rects3 = ax.bar(x + width, validate, width, label='Validation')

    ax.tick_params(axis='x', which='major', labelsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.title(title)

    plt.show()


def duration_histogram(name: str, set: str):

    songs = io.loadmat('data/' + name)[set][0]

    record = np.zeros(32)

    for song in songs:

        for note in range(88):

            on = False
            count = 0

            for t in range(song.shape[0]):

                if song[t, note] == 1:
                    count += 1
                    on = True

                elif song[t, note] == 0:
                    if on:
                        record[count - 1] += 1
                        count = 0
                    on = False

                else:
                    raise ValueError("Piano roll should be binary.")

    durations = []
    labels = []
    for i, r in enumerate(record):
        if r > 0:
            durations.append(r)
            labels.append(str(i + 1))

    durations = durations[4:]
    labels = labels[4:]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x, durations, width)

    ax.tick_params(axis='x', which='major', labelsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    #ax.legend()

    plt.title("Note duration distribution: " + name + " " + set)

    plt.show()
