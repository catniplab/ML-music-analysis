"""
This script creates an instance of a sacred experiment and defines default configurations.
"""

import src.util as util
from src.load_data import get_data_loader
from src.jit_model import get_model

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

from sacred import Experiment
from copy import deepcopy
from tqdm import tqdm


# give all random number generators the same seed
def _seed_all(_seed) -> None:
    torch.manual_seed(_seed)
    np.random.seed(_seed)
    random.seed(_seed)


# create a new sacred experiment whose name is an integer
ex = Experiment(name=str(random.randint(0, 1000000)))


# default configurations
@ex.config
def cfg():

    system = {
              'cuda': torch.cuda.is_available(),
              'gpu': 0,
              'save_dir': os.getcwd()
             }

    # supported datasets
    # Nottingham
    # Piano_midi
    # MuseData
    training = {
                'name': "Nottingham",
                'set': "test",
                'num_epochs': 150,
                'batch_size': 128,
                'lr': 0.001,
                'decay': 1.0,
                'optimizer': "SGD"
                }

    hpsearch = {
                'do_hpsearch': False,
                'learning_rates': 10**np.linspace(-2, -4, 5),
                'epochs': 50
                }

    # supported architectures
    # LINEAR
    # TANH_RNN
    # GRU
    # LSTM
    model = {
             'architecture': 'LINEAR',
             'gradient_clipping': None,
             'jit': False,
             'input_size': 88,
             'hidden_size': 300,
             'num_layers': 1,
             'output_size': 88
            }

    # supported initializations
    # Identity
    initializer = {
                  'init': 'default',
                  'scale': 1.0,
                  'min_angle': 0.0,
                  'max_angle': 2.0
                  }

    # detect backprop anamolies
    detect_anomaly = False


# main function
@ex.automain
def train_model(
                system,
                training,
                hpsearch,
                model,
                initializer,
                detect_anomaly,
                _seed,
                _log,
                _run):

    # give all random number generators the same seed
    _seed_all(_seed)

    # if we are debugging we may want to detect autograd anomalies
    torch.autograd.set_detect_anomaly(detect_anomaly)

    # get the data loaders
    train_loader, test_loader, val_loader = get_data_loader(training['name'], training['batch_size'])

    # standard training loop
    if not hpsearch['do_hpsearch']:

        # if we are on cuda we construct the device and run everything on it
        device = util.NullContext()
        if system['cuda']:
            device = torch.cuda.device('cuda:' + str(system['gpu']))

        with device:

            model = get_model(model, initializer)

            # always use this loss function for multi-variate binary prediction
            loss_fcn = nn.BCEWithLogitsLoss(reduction='sum')

            # construct the optimizer
            optimizer = None
            if training['optimizer'] == "SGD":
                optimizer = optim.SGD(model.parameters(), lr=training['lr'])
            elif training['optimizer'] == "Adam":
                optimizer = optim.Adam(model.parameters(), lr=training['lr'])
            elif training['optimizer'] == "RMSprop":
                optimizer = optim.RMSprop(model.parameters(), lr=training['lr'])
            else:
                raise ValueError("Optimizer {} not recognized.".format(optimizer))

            # learning rate decay
            #scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: decay**epoch)

            # save a copy of the initial model and make sacred remember it
            init_sd = deepcopy(model.state_dict())
            torch.save(init_sd, 'initial_state_dict.pt')
            _run.add_artifact('initial_state_dict.pt')

            # begin training loop
            for epoch in tqdm(range(training['num_epochs'])):

                for input_tensor, target_tensor in train_loader:

                    #_log.warning("\nInput tensor: " + str(input_tensor.shape))
                    #_log.warning("Target tensor: " + str(target_tensor.shape))

                    optimizer.zero_grad()

                    output_tensor, hidden_tensor = model(input_tensor)
                    prediction = output_tensor[:, -1]

                    loss = loss_fcn(prediction, target_tensor)
                    loss.backward()
                    optimizer.step()

                    # use sacred to log training loss and accuracy
                    _run.log_scalar("training.loss", loss.cpu().detach().item())
                    _run.log_scalar("training.accuracy", util.compute_accuracy(model, train_loader))

                # learning rate decay
                #scheduler.step()

                # use sacred to log testing and validation loss and accuracy

                test_loss = util.compute_loss(loss_fcn, model, test_loader)
                _run.log_scalar("testing.loss", test_loss)

                val_loss = util.compute_loss(loss_fcn, model, val_loader)
                _run.log_scalar("validation.loss", val_loss)

                test_acc = util.compute_accuracy(model, test_loader)
                _run.log_scalar("testing.accuracy", test_acc)

                val_acc = util.compute_accuracy(model, val_loader)
                _run.log_scalar("validation.accuracy", val_acc)

            # save a copy of the trained model and make sacred remember it
            sd = model.state_dict()
            torch.save(sd, 'final_state_dict.pt')
            _run.add_artifact('final_state_dict.pt')

    # only goal here is to find the best hyper parameters
    else:
        raise NotImplementedError