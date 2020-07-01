"""
This script creates an instance of a sacred experiment and defines default configurations.
"""

from src.load_data import get_loader
from src.models import get_model
from src.metrics import MaskedBCE, Accuracy

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
from copy import deepcopy
from time import sleep
from tqdm import tqdm

from typing import List

# TODO
# make sure test and validation accuracy are being logged correctly, might have to take unbalanced average over batches in case batches have very different sizes


# create a new sacred experiment whose name is an integer
ex = Experiment(name=str(random.randint(0, 1000000)))


# default configurations
@ex.config
def cfg():

    system = {
              'cuda': torch.cuda.is_available(),
              'gpu': 0,
              'base_dir': os.getcwd()
             }

    # supported datasets
    # JSB_Chorales (short)
    # Nottingham (medium)
    # Piano_midi (long)
    # MuseData (extra long)
    training = {
                'dataset': "JSB_Chorales",
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
    model_dict = {
                  'architecture': 'LINEAR',
                  'gradient_clipping': 1,
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

    # when to save state dictionaries
    saving = {
              'init_model': True,
              'final_model': True,
              'every_epoch': False
             }

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


@ex.capture
def log_loss(loss_fcn: nn.Module,
             model: nn.Module,
             loader: DataLoader,
             log_name: str,
             _run):
    """
    :param loss_fcn: pytorch module whose forward function computes a loss
    :param model: model which we are testing
    :param loader: data loader for which we will be testing the loss
    :return: log average loss for every song in the list
    """

    # record loss for all sequences and count the total number of sequences
    all_loss = []
    num_seqs = 0

    for input_tensor, target_tensor, mask in loader:

        num_seqs += input_tensor.shape[0]

        output, hiddens = model(input_tensor)

        loss = loss_fcn(output, target_tensor, mask)
        all_loss.append(loss.cpu().detach().item())

    # log the average loss across every sequence
    avg = np.sum(all_loss)/num_seqs
    _run.log_scalar(log_name, avg)


@ex.capture
def log_accuracy(model: nn.Module,
                 loader: DataLoader,
                 log_name: str,
                 device,
                 _log,
                 _run):
    """
    :param model: model which we are testing
    :param loader: list of loader on which we will evaluate frame-level accuracy
    :param log_name: name of the log where we store the accuracy
    :return: average accuracy for every song in the list
    """

    # see metrics.py
    acc_fcn = Accuracy()

    # record accuracy for all sequences and count the total number of sequences
    all_acc = []
    num_seqs = 0

    for input_tensor, target_tensor, mask in loader:

        num_seqs += input_tensor.shape[0]

        output, hiddens = model(input_tensor)

        acc = acc_fcn(output, target_tensor, mask)
        all_acc.append(acc)

    # log the average accuracy across every sequence
    avg = np.sum(all_acc)/num_seqs
    _run.log_scalar(log_name, np.sum(all_acc))


# main function
@ex.automain
def train_model(
                system,
                training,
                hpsearch,
                model_dict,
                initializer,
                saving,
                detect_anomaly,
                _seed,
                _log,
                _run):

    # give all random number generators the same seed
    _seed_all(_seed)

    # save artifacts to a temporary directory that gets erased when the experiment is over
    save_dir = system['base_dir'] + '/tmp_' + str(_seed)
    os.system('mkdir ' + save_dir)
    save_dir += '/'

    # if we are debugging we may want to detect autograd anomalies
    torch.autograd.set_detect_anomaly(detect_anomaly)

    # get the data loaders
    dataset = training['dataset']
    batch_size = training['batch_size']
    train_loader, test_loader, val_loader = get_loader(dataset, batch_size)

    # standard training loop
    if not hpsearch['do_hpsearch']:

        # construct and initialize the model
        cuda = system['cuda']
        model = get_model(model_dict, initializer, cuda)

        # save a copy of the initial model and make sacred remember it
        if saving['init_model']:
            init_sd = deepcopy(model.state_dict())
            torch.save(init_sd, save_dir + 'initial_state_dict.pt')
            _run.add_artifact(save_dir + 'initial_state_dict.pt')

        # if we are on cuda we construct the device and run everything on it
        cuda_device = NullContext()
        device = torch.device('cpu')
        if cuda:
            dev_name = 'cuda:' + str(system['gpu'])
            cuda_device = torch.cuda.device(dev_name)
            device = torch.device(dev_name)
            model = model.to(device)

        with cuda_device:

            # see metrics.py
            loss_fcn = MaskedBCE()

            # construct the optimizer
            optimizer = None
            if training['optimizer'] == "SGD":
                optimizer = optim.SGD(model.parameters(), lr=training['lr'])
            elif training['optimizer'] == "Adam":
                optimizer = optim.Adam(model.parameters(), lr=training['lr'])
            elif training['optimizer'] == "RMSprop":
                optimizer = optim.RMSprop(model.parameters(), lr=training['lr'])
            elif training['optimizer'] == "SecondOrder":
                # see https://github.com/cybertronai/pytorch-sso/blob/master/torchsso/optim/secondorder.py
                shapes = {"Linear": "Diag"}
                kwargs = {"damping": 1e-3, "ema_decay": 0.999}
                optimizer = soptim.SecondOrderOptimizer(model, "Cov", shapes, kwargs)
            else:
                raise ValueError("Optimizer {} not recognized.".format(training['optimizer']))

            # learning rate decay
            decay = training['decay']
            scheduler = None
            if training['optimizer'] != "SecondOrder":
                scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: decay**epoch)

            # begin training loop
            for epoch in tqdm(range(training['num_epochs'])):

                for input_tensor, target_tensor, mask in train_loader:

                    input_tensor = input_tensor.to(device)

                    # number of songs in this batch
                    N = input_tensor.shape[0]

                    output, hidden_tensors = model(input_tensor)

                    loss = loss_fcn(output, target_tensor, mask)/N
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # use sacred to log training loss and accuracy
                    _run.log_scalar("trainLoss", loss.cpu().detach().item())
                    log_accuracy(model, train_loader, "trainAccuracy", cuda_device, _log, _run)

                    # save a copy of the model and make sacred remember it each epoch
                    if saving['every_epoch']:
                        sd = deepcopy(model.state_dict())
                        torch.save(init_sd, save_dir + 'state_dict_' + str(epoch) + '.pt')
                        _run.add_artifact(save_dir + 'state_dict_' + str(epoch) + '.pt')

                # learning rate decay
                if training['optimizer'] != "SecondOrder":
                    scheduler.step()

                # use sacred to log testing and validation loss and accuracy
                log_loss(loss_fcn, model, test_loader, 'testLoss', _run)
                log_loss(loss_fcn, model, val_loader, 'validLoss', _run)
                log_accuracy(model, test_loader, 'testAccuracy', _run)
                log_accuracy(model, val_loader, 'validAccuracy', _run)

            # save a copy of the trained model and make sacred remember it
            if saving['final_model']:
                fin_sd = deepcopy(model.state_dict())
                torch.save(fin_sd, save_dir + 'final_state_dict.pt')
                _run.add_artifact(save_dir + 'final_state_dict.pt')

            sleep(1)
            os.system('rm -r ' + save_dir)

    # only goal here is to find the best hyper parameters
    else:
        raise NotImplementedError