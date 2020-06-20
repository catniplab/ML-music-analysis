import numpy as np
import numpy.linalg as la
import scipy.io as io
import torch
import rnn

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

def plot_training(filename: str , steps_per_test: int):
    """
    :param filename: essential name of the files containing the relevant data
    :param steps_per_test: how often was the model tested in terms of training steps
    Plots the average training error for each epoch with standard deviation.
    """

    train_dir = '../output_data/training/'

    #gen = np.load(train_dir + filename + '.npy')
    #gen_std = np.load(train_dir + filename + '_std.npy')

    ortho = np.load(train_dir + 'ortho_' + filename + '.npy')
    ortho_std = np.load(train_dir + 'ortho_' + filename + '_std.npy')

    bortho = np.load(train_dir + 'bortho_' + filename + '.npy')
    bortho_std = np.load(train_dir + 'bortho_' + filename + '_std.npy')

    lstm = np.load(train_dir + 'lstm_' + filename + '.npy')
    lstm_std = np.load(train_dir + 'lstm_' + filename + '_std.npy')

    brefl = np.load(train_dir + 'brefl_' + filename + '.npy')
    brefl_std = np.load(train_dir + 'brefl_' + filename + '_std.npy')

    train_steps = len(bortho)
    test_steps = train_steps//steps_per_test

    #plt.plot(range(test_steps), [np.mean(gen[i : i + steps_per_test]) for i in range(0, train_steps, steps_per_test)], label='generic', color='red')
    #plt.fill_between(range(test_steps), [np.mean(gen[i : i + steps_per_test] + gen_std[i : i + steps_per_test]) for i in range(0, train_steps, steps_per_test)], [np.mean(gen[i : i + steps_per_test] - gen_std[i : i + steps_per_test]) for i in range(0, train_steps, steps_per_test)], color='red', alpha=0.4)

    plt.plot(range(test_steps), [np.mean(ortho[i : i + steps_per_test]) for i in range(0, train_steps, steps_per_test)], label='orthogonal', color='green')
    plt.fill_between(range(test_steps), [np.mean(ortho[i : i + steps_per_test] + ortho_std[i : i + steps_per_test]) for i in range(0, train_steps, steps_per_test)], [np.mean(ortho[i : i + steps_per_test] - ortho_std[i : i + steps_per_test]) for i in range(0, train_steps, steps_per_test)], color='green', alpha=0.4)

    plt.plot(range(test_steps), [np.mean(bortho[i : i + steps_per_test]) for i in range(0, train_steps, steps_per_test)], label='block rotation', color='blue')
    plt.fill_between(range(test_steps), [np.mean(bortho[i : i + steps_per_test] + bortho_std[i : i + steps_per_test]) for i in range(0, train_steps, steps_per_test)], [np.mean(bortho[i : i + steps_per_test] - bortho_std[i : i + steps_per_test]) for i in range(0, train_steps, steps_per_test)], color='blue', alpha=0.4)

    plt.plot(range(test_steps), [np.mean(brefl[i : i + steps_per_test]) for i in range(0, train_steps, steps_per_test)], label='block reflection', color='purple')
    plt.fill_between(range(test_steps), [np.mean(brefl[i : i + steps_per_test] + brefl_std[i : i + steps_per_test]) for i in range(0, train_steps, steps_per_test)], [np.mean(brefl[i : i + steps_per_test] - brefl_std[i : i + steps_per_test]) for i in range(0, train_steps, steps_per_test)], color='purple', alpha=0.4)

    plt.plot(range(test_steps), [np.mean(lstm[i : i + steps_per_test]) for i in range(0, train_steps, steps_per_test)], label='LSTM', color='black')
    plt.fill_between(range(test_steps), [np.mean(lstm[i : i + steps_per_test] + lstm_std[i : i + steps_per_test]) for i in range(0, train_steps, steps_per_test)], [np.mean(lstm[i : i + steps_per_test] - lstm_std[i : i + steps_per_test]) for i in range(0, train_steps, steps_per_test)], color='black', alpha=0.4)

    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title(filename)

    plt.show()

def plot_testing(filename: str , steps_per_test: int):
    """
    :param filename: essential name of the files containing the relevant data
    :param steps_per_test: how often was the model tested in terms of training steps
    Plots testing error for each epoch with standard deviation.
    """

    test_dir = '../output_data/testing/'

    #gen_test = np.load(test_dir + filename + '_test.npy')
    #gen_test_std = np.load(test_dir + filename + '_test_std.npy')

    ortho_test = np.load(test_dir + 'ortho_' + filename + '_test.npy')
    ortho_test_std = np.load(test_dir + 'ortho_' + filename + '_test_std.npy')

    bortho_test = np.load(test_dir + 'bortho_' + filename + '_test.npy')
    bortho_test_std = np.load(test_dir + 'bortho_' + filename + '_test_std.npy')

    lstm_test = np.load(test_dir + 'lstm_' + filename + '_test.npy')
    lstm_test_std = np.load(test_dir + 'lstm_' + filename + '_test_std.npy')

    brefl_test = np.load(test_dir + 'brefl_' + filename + '_test.npy')
    brefl_test_std = np.load(test_dir + 'brefl_' + filename + '_test_std.npy')

    test_steps = len(bortho_test)

    #plt.plot(range(test_steps), gen_test, '--', color='red')
    #plt.fill_between(range(test_steps), gen_test + gen_test_std, gen_test - gen_test_std, color='red', alpha=0.4)

    plt.plot(range(test_steps), ortho_test, '--', color='green')
    plt.fill_between(range(test_steps), ortho_test + ortho_test_std, ortho_test - ortho_test_std, color='green', alpha=0.4)

    plt.plot(range(test_steps), bortho_test, '--', color='blue')
    plt.fill_between(range(test_steps), bortho_test + bortho_test_std, bortho_test - bortho_test_std, color='blue', alpha=0.4)

    plt.plot(range(test_steps), brefl_test, '--', color='purple')
    plt.fill_between(range(test_steps), brefl_test + brefl_test_std, brefl_test - brefl_test_std, color='purple', alpha=0.4)

    plt.plot(range(test_steps), lstm_test, '--', color='black')
    plt.fill_between(range(test_steps), lstm_test + lstm_test_std, lstm_test - lstm_test_std, color='black', alpha=0.4)

    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title(filename + ' test')

    plt.show()

def plot_hidden_weights(filename: str):

    state_dict = torch.load('networks/' + filename + '.pt', map_location='cpu')
    hidden_weights = state_dict['hidden_weights.weight']
    hidden_dim = hidden_weights.shape[0]

    fig, ax = plt.subplots()
    ax.pcolor(hidden_weights, vmin=-0.5, vmax=0.5, cmap='MyMap')
    ax.set_aspect('equal')
    fig.show()
    plt.gca().invert_yaxis()

def plot_all_eigs(filename: str, num=''):

    state_dict = torch.load('networks/' + filename + num + '.pt', map_location='cpu')
    hidden_weights = state_dict['hidden_weights.weight']
    hidden_dim = hidden_weights.shape[0]

    vals, vecs = la.eig(hidden_weights)

    fig, ax = plt.subplots()
    ax.set_xlim([-2.0, 2.0])
    ax.set_ylim([-2.0, 2.0])

    circle = plt.Circle((0, 0), 1, color='lightgrey', fill=False)
    ax.add_artist(circle)
    ax.scatter(np.real(vals), np.imag(vals))

    ax.set_aspect('equal')
    fig.show()

def plot_block_eigs(filename: str, num=''):

    state_dict = torch.load('networks/' + filename + num + '.pt', map_location='cpu')
    hidden_weights = state_dict['hidden_weights.weight']
    hidden_dim = hidden_weights.shape[0]

    fig, ax = plt.subplots()
    ax.set_xlim([-2.0, 2.0])
    ax.set_ylim([-2.0, 2.0])

    circle = plt.Circle((0, 0), 1, color='lightgrey', fill=False)
    ax.add_artist(circle)

    for i in range(hidden_dim//2):
        block = hidden_weights[2*i : 2*(i + 1), 2*i : 2*(i + 1)]
        vals, vecs = la.eig(block)
        ax.scatter(np.real(vals), np.imag(vals))

    ax.set_aspect('equal')
    fig.show()

def plot_hidden_states(network_file: str, data_file: str, d_in: int, d_hid: int, d_out: int):

    state_dict = torch.load('networks/' + network_file + '.pt', map_location='cpu')
    model = None
    if network_file.startswith('lstm'):
        model = rnn.MyLSTM(d_in, d_hid, d_out)
    else:
        model = rnn.BlockOrthoRNN(d_in, d_hid, d_out)
    model.load_state_dict(state_dict)

    seqs = np.load('input_data/' + data_file + '.npy')
    tensor = torch.tensor(seqs, dtype=torch.float)

    output, hiddens = model(tensor)
    first_hidden = np.array([hidden[0].detach().numpy() for hidden in hiddens])

    for i in range(d_hid):
        plt.plot(first_hidden[:, i] + 3*i, linewidth=0.5)
    plt.show()

def plot_outputs(network_file: str, data_file: str, d_in: int, d_hid: int, d_out: int):

    state_dict = torch.load('networks/' + network_file + '.pt', map_location='cpu')
    model = None
    if network_file.startswith('lstm'):
        model = rnn.MyLSTM(d_in, d_hid, d_out)
    else:
        model = rnn.BlockOrthoRNN(d_in, d_hid, d_out)
    model.load_state_dict(state_dict)

    seqs = np.load('input_data/' + data_file + '.npy')
    tensor = torch.tensor(seqs, dtype=torch.float)

    output, hiddens = model(tensor)
    first_output = output[0].detach().numpy()

    for i in range(10):
        plt.plot(first_output[:, i] + 10*i, linewidth=0.5)
    plt.show()

    return seqs[0]