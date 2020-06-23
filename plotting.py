import numpy as np
import numpy.linalg as la
import scipy.io as io
import torch
import json

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


def plot_hidden_weights(dir: str, name: str, vmin: float, vmax: float):
    """
    :param dir: directory of the file storage system whose results we are looking at
    :param name: name of the .pt file whose .weight_hh_l0.weight we will visualize
    :param vmin: expected minimum weight
    :param vmax: expected maximum weight
    """

    path = 'results/' + dir + '/'

    sd = torch.load(path + name, map_location='cpu')
    hidden_weights = sd['rnn.weight_hh_l0.weight'].detach().numpy()

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