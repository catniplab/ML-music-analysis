"""
This script contains functions which can assist in querying the directory containing the trained models. There is a lot of overlap with the utilities here and those in plotting.py
"""

import os
import json

import numpy as np
import torch

from src.neural_nets.metrics import MaskedBCE, compute_loss
from src.neural_nets.load_data import get_loader
from src.neural_nets.models import get_model

# a list of configs whose results we want to plot
useful_configs = [
                   #{'architecture': "TANH", 'init': "zero", 'do_hpsearch': False},
                   {'architecture': "LDS", 'init': "zero", 'do_hpsearch': False},
                   #{'architecture': "TANH", 'init': "identity", 'scale': 0.01, 'do_hpsearch': False},
                   #{'architecture': "LDS", 'init': "identity", 'scale': 0.01, 'do_hpsearch': False},
                   #{'architecture': "TANH", 'init': "identity", 'scale': 1, 'do_hpsearch': False},
                   #{'architecture': "LDS", 'init': "identity", 'scale': 1, 'do_hpsearch': False},
                   #{'architecture': "TANH", 'init': "blockortho", 'scale': 0.01, 'do_hpsearch': False},
                   {'architecture': "LDS", 'init': "blockortho", 'scale': 0.01, 'do_hpsearch': False},
                   #{'architecture': "TANH", 'init': "blockortho", 'scale': 1, 'do_hpsearch': False},
                   #{'architecture': "LDS", 'init': "blockortho", 'scale': 1, 'do_hpsearch': False},
                   #{'architecture': "TANH", 'init': "normal", 'scale': 0.01, 'do_hpsearch': False},
                   #{'architecture': "LDS", 'init': "normal", 'scale': 0.01, 'do_hpsearch': False},
                   #{'architecture': "LDS", 'init': 'critical', 'scale': 1, 'do_hpsearch': False},
                   #{'architecture': "TANH", 'init': 'critical', 'scale': 1, 'do_hpsearch': False},
                   #{'architecture': "LDS", 'init': 'critical', 'scale': 0.1, 'do_hpsearch': False},
                   #{'architecture': "TANH", 'init': 'critical', 'scale': 0.1, 'do_hpsearch': False},
                   {'architecture': "REGRESSION", 'init': "regression", 'do_hpsearch': False, 'lag': 1},
                   #{'architecture': "REGRESSION", 'init': "default", 'do_hpsearch': False, 'lag': 7},
                   #{'architecture': "REGRESSION_WIDE", 'init': "default", 'window': 7,'do_hpsearch': False}
                 ]

reg_configs = [
               {'architecture': "REGRESSION", 'lag': 1, 'window': 1, 'do_hpsearch': False},
               {'architecture': "REGRESSION", 'lag': 1, 'window': 2, 'do_hpsearch': False},
               {'architecture': "REGRESSION", 'lag': 1, 'window': 3, 'do_hpsearch': False},
               {'architecture': "REGRESSION", 'lag': 1, 'window': 4, 'do_hpsearch': False},
               {'architecture': "REGRESSION", 'lag': 1, 'window': 5, 'do_hpsearch': False},
               {'architecture': "REGRESSION", 'lag': 1, 'window': 6, 'do_hpsearch': False},
               {'architecture': "REGRESSION", 'lag': 1, 'window': 7, 'do_hpsearch': False}
              ]

reg_labels = [
              'Width 1',
              'Width 2',
              'Width 3',
              'Width 4',
              'Width 5',
              'Width 6',
              'Width 7'
             ]

# labels corresponding
labels = [
          #'RNN:\nzeros',
          'LDS:\nzeros',
          #'RNN:\ndiag',
          #'LDS:\ndiag',
          #'RNN:\nidentity',
          #'LDS:\nidentity',
          #'RNN:\n sbrot',
          'LDS:\n sbrot',
          #'RNN:\nbrot',
          #'LDS:\nbrot',
          #'RNN:\nnormal',
          #'LDS:\nnormal',
          #'LDS:\ncritical',
          #'LDS:\nscritical',
          #'RNN:\ncritical',
          #'RNN:\nscritical',
          'Regression',
          #'Reg:\nlag 7',
          #'Reg:\nwidth 7'
          ]

# find the directories with these configurations
config_dict = {
               'architecture': "TANH",
               #'lag': 1,
               #'window': 1,
               'dataset': "JSB_Chorales",
               #'init': "default",
               #'parity': "rotate",
               #'scale': 0.01,
               #'lag': 8,
               #'init': "blockortho",
               'do_hpsearch': False
              }

# success argument checks if there are NaNs in the loss records
def find_results(configs, success=False):

    good_dirs = []

    dirs = os.listdir('models')

    for dir in dirs:

        if dir != "_sources":

            config_file = open('models/' + dir + '/config.json')
            config_contents = config_file.read()
            config_file.close()

            file_configs = json.loads(config_contents)

            agree = True

            for key, value in configs.items():

                try:
                    if file_configs[key] != value:
                        agree = False
                        break

                except:
                    agree = False
                    break

            if success:

                try:

                    metric_file = open('models/' + dir + '/metrics.json')
                    metric_contents = metric_file.read()
                    metric_file.close()

                    metrics = json.loads(metric_contents)
                    trainLoss = metrics['trainLoss']['values']
                    testLoss = metrics['testLoss']['values']
                    validLoss = metrics['validLoss']['values']

                    nan = float("NaN")

                    if nan in trainLoss or nan in testLoss or nan in validLoss:
                        agree = False

                except:
                    agree = False

            if agree:
                good_dirs.append(dir)

    return good_dirs


def find_recent_metrics(config_dicts, eval_loss=False, initial=False):
    """
    :param config_dicts: list of dictionaries whose metrics we want to find
    :param eval_loss: whether or not we re-compute the loss (regularized loss may've been computed during training)
    :param initial: whether or not to use the initial state dictionaries
    :return: a pair: a triple of losses and a triple of accuracies. Each entry is train, test, loss.
    """

    train_losses = []
    test_losses = []
    val_losses = []

    train_accs = []
    test_accs = []
    val_accs = []

    for cdict in config_dicts:

        # find the most recent experiment with the given configs
        good_dirs = find_results(cdict, success=True)
        recent_dir = 'results/' + str(np.sort([int(dir) for dir in good_dirs])[-1])
        metric_handle = open(recent_dir + '/metrics.json')
        recent_metrics = json.loads(metric_handle.read())
        metric_handle.close()

        # take note of all the most recent accuracies
        train_accs.append(recent_metrics['trainAccuracy']['values'][-1])
        test_accs.append(recent_metrics['testAccuracy']['values'][-1])
        val_accs.append(recent_metrics['validAccuracy']['values'][-1])

        # compute the unregularized loss for the model over each dataset, just in case
        if eval_loss and cdict['architecture'] != "REGRESSION":

            # we must read some information about the model to properly construct the data loaders
            config_handle = open(recent_dir + '/config.json')
            recent_configs = json.loads(config_handle.read())
            config_handle.close()
            dataset = recent_configs['dataset']
            batch_size = recent_configs['batch_size']
            architecture = recent_configs['architecture']

            # determine if there is a certain part of the sequences we need to cover up
            init_mask = 0
            if architecture == "REGRESSION":
                init_mask = recent_configs['lag']
            elif architecture == "REGRESSION_WIDE":
                init_mask = recent_configs['window']

            train_loader, test_loader, val_loader = get_loader(dataset, batch_size, init_mask)

            # loss function without regularization
            loss_fcn = MaskedBCE(0)

            # get the configuration for the model and construct it
            # initializer is not required
            model_dict = {'architecture': architecture,
                          'readout': recent_configs['readout'],
                          'gradient_clipping': recent_configs['gradient_clipping'],
                          'jit': recent_configs['jit'],
                          'lag': recent_configs['lag'],
                          'window': recent_configs['window'],
                          'input_size': recent_configs['input_size'],
                          'hidden_size': recent_configs['hidden_size'],
                          'num_layers': recent_configs['num_layers'],
                          'output_size': recent_configs['output_size']
                         }
            dict_name = '/final_state_dict.pt'
            if initial:
                dict_name = '/initial_state_dict.pt'
            state_dict = torch.load(str(recent_dir) + dict_name, map_location='cpu')
            model = get_model(model_dict, {'init': "default"}, False)
            model.load_state_dict(state_dict)

            # compute and record the losses
            train_losses.append(compute_loss(loss_fcn, model, train_loader))
            test_losses.append(compute_loss(loss_fcn, model, test_loader))
            val_losses.append(compute_loss(loss_fcn, model, val_loader))

    else:

        # take note of all the most recent accuracies
        train_losses.append(recent_metrics['trainLoss']['values'][-1])
        test_losses.append(recent_metrics['testLoss']['values'][-1])
        val_losses.append(recent_metrics['validLoss']['values'][-1])

    return ((train_losses, test_losses, val_losses), (train_accs, test_accs, val_accs))


if __name__ == "__main__":
    print(find_results(config_dict, success=True))