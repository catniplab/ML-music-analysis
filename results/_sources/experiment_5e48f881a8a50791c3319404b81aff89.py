"""
This script creates an instance of a sacred experiment and defines default configurations.
"""

from src.load_data import get_data_loader
from src.models import get_model

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

from torch.utils.data import DataLoader
from sacred import Experiment
from copy import deepcopy
from time import sleep
from tqdm import tqdm


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
    :param loader: DataLoader for either testing or validation data
    :return: log average loss for every batch in the loader
    """

    all_loss = []

    for input_tensor, target_tensor in loader:

        output, hiddens = model(input_tensor)
        prediction = output[-1]

        loss = loss_fcn(prediction, target_tensor)
        all_loss.append(loss.cpu().detach().item())

    _run.log_scalar(log_name, np.mean(all_loss))


@ex.capture
def log_accuracy(model: nn.Module,
                 loader: DataLoader,
                 log_name: str,
                 _run):
    """
    :param model: model which we are testing
    :param loader: DataLoader for either testing or validation data
    :return: average accuracy for every batch in the loader
    """

    all_acc = []

    # one way of computing accuracy?
    def acc_fcn(output, target):

        prediction = (torch.sigmoid(output) > 0.5).type(torch.get_default_dtype())

        true_pos = torch.sum(prediction*target)
        false_pos = torch.sum(prediction*(1 - target))
        false_neg = torch.sum((1 - prediction)*target)

        return true_pos/(true_pos + false_pos + false_neg)

    for input_tensor, target_tensor in loader:

        output, hiddens = model(input_tensor)
        prediction = output[-1]

        acc = acc_fcn(prediction, target_tensor)
        all_acc.append(acc.cpu().detach().item())

    _run.log_scalar(log_name, np.mean(all_acc))


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
    train_loader, test_loader, val_loader = get_data_loader(dataset, batch_size)

    # standard training loop
    if not hpsearch['do_hpsearch']:

        # construct and initialize the model
        model = get_model(model_dict, initializer)

        # save a copy of the initial model and make sacred remember it
        if saving['init_model']:
            init_sd = deepcopy(model.state_dict())
            torch.save(init_sd, save_dir + 'initial_state_dict.pt')
            _run.add_artifact(save_dir + 'initial_state_dict.pt')

        # if we are on cuda we construct the device and run everything on it
        cuda = system['cuda']
        cuda_device = NullContext()
        device = torch.device('cpu')
        if cuda:
            dev_name = 'cuda:' + str(system['gpu'])
            cuda_device = torch.cuda.device(dev_name)
            device = torch.device(dev_name)
            model = model.to(device)

        with cuda_device:

            # always use this loss function for multi-variate binary prediction
            # on piano music where there are 88 possible notes
            loss_fcn = lambda out, targ: 88*nn.BCEWithLogitsLoss()(out, targ)

            # construct the optimizer
            optimizer = None
            if training['optimizer'] == "SGD":
                optimizer = optim.SGD(model.parameters(), lr=training['lr'])
            elif training['optimizer'] == "Adam":
                optimizer = optim.Adam(model.parameters(), lr=training['lr'])
            elif training['optimizer'] == "RMSprop":
                optimizer = optim.RMSprop(model.parameters(), lr=training['lr'])
            else:
                raise ValueError("Optimizer {} not recognized.".format(training['optimizer']))

            # learning rate decay
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: decay**epoch)

            # begin training loop
            for epoch in tqdm(range(training['num_epochs'])):

                for input_tensor, target_tensor in train_loader:

                    input_tensor = input_tensor.to(device)
                    target_tensor = target_tensor.to(device)

                    optimizer.zero_grad()

                    #_log.warning(str([p for p in model.parameters()]))
                    output_tensor, hidden_tensor = model(input_tensor)
                    prediction = output_tensor[-1]

                    loss = loss_fcn(prediction, target_tensor)
                    loss.backward()
                    optimizer.step()

                    # use sacred to log training loss and accuracy
                    _run.log_scalar("trainLoss", loss.cpu().detach().item())
                    log_accuracy(model, train_loader, "trainAccuracy", _run)

                    # save a copy of the model and make sacred remember it each epoch
                    if saving['every_epoch']:
                        sd = deepcopy(model.state_dict())
                        torch.save(init_sd, save_dir + 'state_dict_' + str(epoch) + '.pt')
                        _run.add_artifact(save_dir + 'state_dict_' + str(epoch) + '.pt')

                # learning rate decay
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