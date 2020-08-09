"""
This script creates an instance of a sacred experiment and defines default configurations for training a neural network or a regression model.
"""

from src.neural_nets.models import get_model
from src.neural_nets.load_data import get_loader
from src.neural_nets.metrics import MaskedBCE, Accuracy, compute_acc, compute_loss

import src.regression.logistic_regression as reg

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchsso.optim as soptim
import torch.nn.functional as F
import random

from torch.utils.data import DataLoader
from sacred import Experiment
from torch import Tensor, device
from copy import deepcopy
from time import sleep
from tqdm import tqdm

from typing import List
from itertools import product


# create a new sacred experiment whose name is an integer
ex = Experiment(name=str(random.randint(0, 1000000)))


# default configurations
@ex.config
def cfg():

    # system
    cuda = torch.cuda.is_available()
    gpu = 0
    base_dir = os.getcwd()

    # supported datasets
    # JSB_Chorales (short)
    # Nottingham (medium)
    # Piano_midi (long)
    # MuseData (extra long)
    dataset = "JSB_Chorales"

    # training
    num_epochs = 150
    batch_size = 128
    # mask some low notes and some high notes because they never show up
    low_off_notes = 0
    high_off_notes = 88
    lr = 0.001
    decay = 1.0
    optmzr = "SGD"
    regularization = 0.0

    # hyperparameter search
    do_hpsearch = False
    learning_rates = 10**np.linspace(-2, -4, 5)
    decays = 1 - np.linspace(0, 0.1, num=5)
    regularizations = 10**np.linspace(-2, -4, num=5)
    hps_epochs = 50

    # Supported architectures
    # REGRESSION
    # LDS
    # TANH
    architecture = 'LDS'
    readout = 'linear'
    gradient_clipping = 1
    jit = False # not fully implemented
    # for regression
    lag = 1
    window = 1
    # for neural networks
    input_size = 88
    hidden_size = 300
    num_layers = 1
    output_size = 88

    # see models.py and initialization.py for details
    init = 'default'
    scale = 1.0
    parity = None # see models.py
    t_distrib = torch.distributions.Uniform(0, 0.75)
    path = 'results/77/final_state_dict.pt'

    # when to save state dictionaries
    save_init_model = True
    save_final_model = True
    save_every_epoch = False

    # detect backprop anomalies
    detect_anomaly = False


# give all random number generators the same seed
def _seed_all(_seed) -> None:
    torch.manual_seed(_seed)
    np.random.seed(_seed)
    random.seed(_seed)


# this context is used when we are running things on the cpu
class NullContext(object):
    def __init__(self):
        pass
    def __enter__(self):
        pass
    def __exit__(self, type, value, traceback):
        pass


# this function simply trains regression models and logs the results
# see regression.trainer for details
@ex.capture
def sklearn_experiment(dataset: str,
                       save_dir: str,
                       num_epochs: int,
                       high_off_notes: int,
                       low_off_notes: int,
                       lag: int,
                       window: int,
                       _seed,
                       _log,
                       _run):
    """
    :param dataset: name of the dataset to be used
    :save_dir: temporary directory where artifacts are being stored
    :lag: how many time steps into the future the regression model is to predict
    :window: how many time steps the regression model is to take into account
    :param _seed: sacred random seed
    :param _log: sacred object used to output to the command line
    :param _run: sacred object used to monitor the runtime
    """

    num_notes = high_off_notes - low_off_notes

    models = reg.train_models(dataset,
                              num_epochs,
                              low_off_notes,
                              high_off_notes,
                              _seed,
                              lag=lag,
                              window=window)

    coefs = np.zeros((num_notes, num_notes*window))
    intercepts = np.zeros(num_notes*window)

    for i in range(num_notes):

        model = models[i]

        # if there were no notes played for this channel, a model won't be trained
        # simply save all parameters as -1 to discourage the note from being played
        if model == None:
            coefs[i] = -1
            intercepts[i] = -1

        else:
            coefs[i] = model.coef_
            intercepts[i] = model.intercept_

    np.save(save_dir + 'coefs.npy', coefs)
    np.save(save_dir + 'intercepts.npy', intercepts)

    _run.add_artifact(save_dir + 'coefs.npy')
    _run.add_artifact(save_dir + 'intercepts.npy')

    train_loss = reg.compute_loss(models,
                                  dataset,
                                  'traindata',
                                  low_off_notes,
                                  high_off_notes,
                                  lag=lag,
                                  window=window)
    test_loss = reg.compute_loss(models,
                                 dataset,
                                 'testdata',
                                 low_off_notes,
                                 high_off_notes,
                                 lag=lag,
                                 window=window)
    valid_loss = reg.compute_loss(models,
                                  dataset,
                                  'validdata',
                                  low_off_notes,
                                  high_off_notes,
                                  lag=lag,
                                  window=window)

    _run.log_scalar('trainLoss', train_loss)
    _run.log_scalar('testLoss', test_loss)
    _run.log_scalar('validLoss', valid_loss)

    train_acc = reg.compute_accuracy(models,
                                     dataset,
                                     'traindata',
                                     low_off_notes,
                                     high_off_notes,
                                     lag=lag,
                                     window=window)
    test_acc = reg.compute_accuracy(models,
                                    dataset,
                                    'testdata',
                                    low_off_notes,
                                    high_off_notes,
                                    lag=lag,
                                    window=window)
    valid_acc = reg.compute_accuracy(models,
                                     dataset,
                                     'validdata',
                                     low_off_notes,
                                     high_off_notes,
                                     lag=lag,
                                     window=window)

    _run.log_scalar('trainAccuracy', train_acc)
    _run.log_scalar('testAccuracy', test_acc)
    _run.log_scalar('validAccuracy', valid_acc)


# a single optimization step
@ex.capture
def train_iter(device: device,
               cuda_device: torch.cuda.device,
               input_tensor: Tensor,
               target: Tensor,
               mask: Tensor,
               model: nn.Module,
               loss_fcn: nn.Module,
               optimizer: optim.Optimizer,
               save_every_epoch: bool,
               save_dir: str,
               train_loader: DataLoader,
               test_loader: DataLoader,
               valid_loader: DataLoader,
               low_off_notes: int,
               high_off_notes: int,
               _log,
               _run,
               logging=True):

    input_tensor = input_tensor.to(device)

    # number of songs in this batch
    N = input_tensor.shape[0]

    output, hidden_tensors = model(input_tensor)

    loss = loss_fcn(output, target, mask, model)/N
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # use sacred to log training loss and accuracy
    if logging:
        train_acc = compute_acc(model, train_loader, low=low_off_notes, high=high_off_notes)
        _run.log_scalar("trainLoss", loss.cpu().detach().item())
        _run.log_scalar("trainAccuracy", train_acc)

    # save a copy of the model and make sacred remember it each epoch
    if save_every_epoch and logging:
        sd = deepcopy(model.state_dict())
        torch.save(init_sd, save_dir + 'state_dict_' + str(epoch) + '.pt')
        _run.add_artifact(save_dir + 'state_dict_' + str(epoch) + '.pt')


# train a neural network
# returns the final loss and accuracy on the training, testing, and validation sets
@ex.capture
def pytorch_train_loop(cuda: bool,
                       model_dict: dict,
                       initializer: dict,
                       train_loader: DataLoader,
                       test_loader: DataLoader,
                       valid_loader: DataLoader,
                       low_off_notes: int,
                       high_off_notes: int,
                       optmzr: str,
                       lr: float,
                       decay: float,
                       regularization: float,
                       num_epochs: int,
                       save_dir: str,
                       save_init_model,
                       save_every_epoch,
                       save_final_model,
                       _seed,
                       _log,
                       _run,
                       logging=True):

    # construct and initialize the model
    model = get_model(model_dict, initializer, cuda)

    # save a copy of the initial model and make sacred remember it
    if save_init_model and logging:
        init_sd = deepcopy(model.state_dict())
        torch.save(init_sd, save_dir + 'initial_state_dict.pt')
        _run.add_artifact(save_dir + 'initial_state_dict.pt')

    # if we are on cuda we construct the device and run everything on it
    cuda_device = NullContext()
    device = torch.device('cpu')
    if cuda:
        dev_name = 'cuda:' + str(gpu)
        cuda_device = torch.cuda.device(dev_name)
        device = torch.device(dev_name)
        model = model.to(device)

    with cuda_device:

        # see metrics.py
        loss_fcn = MaskedBCE(regularization, low_off_notes=low_off_notes, high_off_notes=high_off_notes)

        # compute the metrics before training and log them
        if logging:

            train_loss = compute_loss(loss_fcn, model, train_loader)
            test_loss = compute_loss(loss_fcn, model, test_loader)
            val_loss = compute_loss(loss_fcn, model, valid_loader)

            _run.log_scalar("trainLoss", train_loss)
            _run.log_scalar("testLoss", test_loss)
            _run.log_scalar("validLoss", val_loss)

            train_acc = compute_acc(model, train_loader, low=low_off_notes, high=high_off_notes)
            test_acc = compute_acc(model, test_loader, low=low_off_notes, high=high_off_notes)
            val_acc = compute_acc(model, valid_loader, low=low_off_notes, high=high_off_notes)

            _run.log_scalar("trainAccuracy", train_acc)
            _run.log_scalar("testAccuracy", test_acc)
            _run.log_scalar("validAccuracy", val_acc)

        # construct the optimizer
        optimizer = None
        if optmzr == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=lr)
        elif optmzr == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif optmzr == "RMSprop":
            optimizer = optim.RMSprop(model.parameters(), lr=lr)
        else:
            raise ValueError("Optimizer {} not recognized.".format(optmzr))

        # learning rate decay
        scheduler = None
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: decay**epoch)

        # begin training loop
        for epoch in tqdm(range(num_epochs)):

            for input_tensor, target, mask in train_loader:
                train_iter(device,
                           cuda_device,
                           input_tensor,
                           target,
                           mask,
                           model,
                           loss_fcn,
                           optimizer,
                           save_every_epoch,
                           save_dir,
                           train_loader,
                           test_loader,
                           valid_loader,
                           low_off_notes,
                           high_off_notes,
                           _log,
                           _run,
                           logging=logging)

            # learning rate decay
            scheduler.step()

            # use sacred to log testing and validation loss and accuracy
            if logging:

                test_loss = compute_loss(loss_fcn, model, test_loader)
                val_loss = compute_loss(loss_fcn, model, valid_loader)
                test_acc = compute_acc(model, test_loader, low=low_off_notes, high=high_off_notes)
                val_acc = compute_acc(model, valid_loader, low=low_off_notes, high=high_off_notes)

                _run.log_scalar("testLoss", test_loss)
                _run.log_scalar("validLoss", val_loss)
                _run.log_scalar("testAccuracy", test_acc)
                _run.log_scalar("validAccuracy", val_acc)

        # save a copy of the trained model and make sacred remember it
        if save_final_model and logging:
            fin_sd = deepcopy(model.state_dict())
            torch.save(fin_sd, save_dir + 'final_state_dict.pt')
            _run.add_artifact(save_dir + 'final_state_dict.pt')

    # recompute the metrics so that this function can return them
    train_loss = compute_loss(loss_fcn, model, train_loader)
    test_loss = compute_loss(loss_fcn, model, test_loader)
    val_loss = compute_loss(loss_fcn, model, valid_loader)

    train_acc = compute_acc(model, train_loader, low=low_off_notes, high=high_off_notes)
    test_acc = compute_acc(model, test_loader, low=low_off_notes, high=high_off_notes)
    val_acc = compute_acc(model, valid_loader, low=low_off_notes, high=high_off_notes)

    return ((train_loss, test_loss, val_loss), (train_acc, test_acc, val_acc))


# main function
@ex.automain
def train_loop(cuda,
               gpu,
               base_dir,
               dataset,
               num_epochs,
               batch_size,
               low_off_notes,
               high_off_notes,
               lr,
               decay,
               optmzr,
               regularization,
               do_hpsearch,
               learning_rates,
               decays,
               regularizations,
               hps_epochs,
               architecture,
               readout,
               gradient_clipping,
               jit,
               lag,
               window,
               input_size,
               hidden_size,
               num_layers,
               output_size,
               detect_anomaly,
               init,
               scale,
               parity,
               t_distrib,
               path,
               save_init_model,
               save_final_model,
               save_every_epoch,
               _seed,
               _log,
               _run):

    # save artifacts to a temporary directory that gets erased when the experiment is over
    save_dir = base_dir + '/tmp_' + str(_seed)
    os.system('mkdir ' + save_dir)
    save_dir += '/'

    # give all random number generators the same seed
    _seed_all(_seed)

    sklearn_program = architecture == 'REGRESSION'

    # regression models and neural networks are trained very differently
    if sklearn_program:

        sklearn_experiment(dataset,
                           save_dir,
                           num_epochs,
                           high_off_notes,
                           low_off_notes,
                           lag,
                           window,
                           _seed,
                           _log,
                           _run)

    # run a pytorch program
    else:

        model_dict = {'architecture': architecture,
                      'readout': readout,
                      'gradient_clipping': gradient_clipping,
                      'jit': jit,
                      'lag': lag,
                      'window': window,
                      'input_size': input_size,
                      'hidden_size': hidden_size,
                      'num_layers': num_layers,
                      'output_size': output_size
                     }

        initializer = {'init': init,
                       'scale': scale,
                       'parity': parity,
                       't_distrib': t_distrib,
                       'path': path,
                       'low_off_notes': low_off_notes,
                       'high_off_notes': high_off_notes
                      }

        # if we are debugging we may want to detect autograd anomalies
        torch.autograd.set_detect_anomaly(detect_anomaly)

        # construct the pytorch data loaders
        train_loader, test_loader, valid_loader = get_loader(dataset, batch_size)

        # standard training loop
        if not do_hpsearch:

            # the training loop function returns the metrics achieved at the end of training
            # they will be logged by default, no need to do anything with them here
            metrics = pytorch_train_loop(cuda,
                                         model_dict,
                                         initializer,
                                         train_loader,
                                         test_loader,
                                         valid_loader,
                                         low_off_notes,
                                         high_off_notes,
                                         optmzr,
                                         lr,
                                         decay,
                                         regularization,
                                         num_epochs,
                                         save_dir,
                                         save_init_model,
                                         save_every_epoch,
                                         save_final_model,
                                         _seed,
                                         _log,
                                         _run)

        # only goal here is to find the best hyper parameters
        else:

            min_test_loss = float('inf')
            best_lr = 0
            best_dcay = 0
            best_reg = 0

            hyperparams = product(learning_rates, decays, regularizations)

            for rate, dcay, reg in hyperparams:

                # train a model with the given hyperparameters
                # don't log anything, otherwise we will have a ridiculous amount of extraneous info
                metrics = pytorch_train_loop(cuda,
                                             model_dict,
                                             initializer,
                                             train_loader,
                                             test_loader,
                                             valid_loader,
                                             optmzr,
                                             rate,
                                             dcay,
                                             reg,
                                             hps_epochs,
                                             save_dir,
                                             save_init_model,
                                             save_every_epoch,
                                             save_final_model,
                                             _seed,
                                             _log,
                                             _run,
                                             logging=False)

                # loss is first index, test set is second index
                test_loss = metrics[0][1]

                # compare loss against other hyperparams and update if necessary
                if test_loss == test_loss and test_loss < min_test_loss:
                    min_test_loss = test_loss
                    best_lr = rate
                    best_dcay = dcay
                    best_reg = reg

            # record the best hyperparameters
            _run.log_scalar("learning_rate", best_lr)
            _run.log_scalar("decay", best_dcay)
            _run.log_scalar("regularization", best_reg)

    # wait a second then remove the temporary directory used for storing artifacts
    sleep(1)
    os.system('rm -r ' + save_dir)
