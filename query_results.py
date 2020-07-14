import os
import json

import numpy as np
import torch

from src.metrics import MaskedBCE, compute_loss
from src.load_data import get_loader
from src.models import get_model

# a list of configs whose results we want to plot
useful_configs = [ {'architecture': "LDS", 'init': "regression", 'do_hpsearch': False},
                   {'architecture': "REGRESSION", 'init': "default", 'do_hpsearch': False, 'lag': 0},
                   {'architecture': "REGRESSION", 'init': "default", 'do_hpsearch': False, 'lag': 7},
                   {'architecture': "REGRESSION_WIDE", 'init': "default", 'window': 7,'do_hpsearch': False}
                 ]

# labels corresponding
labels = ['LDS', 'Regression', 'Regression:\nlag 7', 'Regression:\nwidth 7', 'Baseline']

# find the directories with these configurations
config_dict = {
               'architecture': "REGRESSION_WIDE",
               #'lag': 7,
               'window': 7,
               #'init': "regression",
               #'parity': "rotate",
               #'scale': 1.0001,
               #'lag': 8,
               #'init': "blockortho",
               'do_hpsearch': False
              }

# success argument checks if there are NaNs in the loss records
def find_results(configs, success=False):

    good_dirs = []

    dirs = os.listdir('results')

    for dir in dirs:

        if dir != "_sources":

            config_file = open('results/' + dir + '/config.json')
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

                    metric_file = open('results/' + dir + '/metrics.json')
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


def find_recent_metrics(config_dicts):
    """
    :param config_dicts: list of dictionaries whose metrics we want to find
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
        state_dict = torch.load(str(recent_dir) + '/final_state_dict.pt', map_location='cpu')
        model = get_model(model_dict, {'init': "default"}, False)
        model.load_state_dict(state_dict)

        # compute and record the losses
        train_losses.append(compute_loss(loss_fcn, model, train_loader))
        test_losses.append(compute_loss(loss_fcn, model, test_loader))
        val_losses.append(compute_loss(loss_fcn, model, val_loader))

    # append the baselines
    train_losses.append(61)
    test_losses.append(61)
    val_losses.append(61)

    train_accs.append(.0442)
    test_accs.append(.0442)
    val_accs.append(.0442)

    return ((train_losses, test_losses, val_losses), (train_accs, test_accs, val_accs))


if __name__ == "__main__":
    print(find_results(config_dict, success=True))