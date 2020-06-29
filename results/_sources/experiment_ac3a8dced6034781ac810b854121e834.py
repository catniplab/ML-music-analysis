"""
This script creates an instance of a sacred experiment and defines default configurations.
"""

from src.load_data import get_songs
from src.models import get_model

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
             songs: List,
             log_name: str,
             _run):
    """
    :param loss_fcn: pytorch module whose forward function computes a loss
    :param model: model which we are testing
    :param songs: list of song tensors which we evaluate average loss on
    :return: log average loss for every song in the list
    """

    all_loss = []

    for song in songs:

        T = song.shape[1]

        output, hiddens = model(song)
        output_tensor = torch.cat(output).reshape(T, 1, 88).permute([1,0,2])
        prediction = output_tensor[:, 0 : -1]
        target = song[:, 1 : T]

        # append average loss over time
        loss = loss_fcn(prediction, target)/T
        all_loss.append(loss.cpu().detach().item())

    # log the mean across every batch
    _run.log_scalar(log_name, np.mean(all_loss))


@ex.capture
def log_accuracy(model: nn.Module,
                 songs: List,
                 log_name: str,
                 device,
                 _log,
                 _run):
    """
    :param model: model which we are testing
    :param songs: list of songs on which we will evaluate frame-level accuracy
    :param log_name: name of the log where we store the accuracy
    :return: average accuracy for every song in the list
    """

    # store weighted accuracy for each batch
    all_acc = []

    # count total number of sequences
    num_seqs = 0

    # see Bay et al 2009 for the definition of frame-level accuracy
    def acc_fcn(output, target):

        T = target.shape[1]

        prediction = (torch.sigmoid(output) > 0.5).type(torch.get_default_dtype())

        #_log.warning(str(prediction.shape))
        #_log.warning(str(target.shape))

        # count total true positives for each sequence
        true_pos = torch.sum(prediction*target, dim=2) # sum over channels (notes)
        true_pos = torch.sum(true_pos, dim=1) # sum over time

        # where we store accuracy at each time step
        acc_over_time = []

        for t in range(T):

            # get false positives and negatives for each sequence
            false_pos = torch.sum(prediction[:, t]*(1 - target[:, t]), dim=1)
            false_neg = torch.sum((1 - prediction[:, t])*target[:, t], dim=1)

            # compute accuracy for each sequence
            acc_each_sequence = true_pos/(true_pos + false_pos + false_neg)

            # total across all sequences in the batch
            acc_over_time.append(torch.sum(acc_each_sequence))

        # return average over time
        return np.mean(acc_over_time)

    for song in songs:

        T = song.shape[1]

        output, hiddens = model(song)
        output_tensor = torch.cat(output).reshape(T, 1, 88).permute([1,0,2])
        prediction = output_tensor[:, 0 : -1]
        target = song[:, 1 : T]
        #_log.warning(prediction.shape)
        #_log.warning(target.shape)

        # append accuracy to the accumulation list
        acc = acc_fcn(prediction.cpu(), target)
        all_acc.append(acc)

    # log the average accuracy across every batch
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

    # get the song lists
    dataset = training['dataset']
    train_songs, test_songs, val_songs = get_songs(dataset)

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

                for song in train_songs:

                    song = song.to(device)

                    # length of current song
                    T = song.shape[1]
                    #_log.warning(T)

                    #_log.warning(str([p for p in model.parameters()]))
                    output, hidden_tensors = model(song)
                    #_log.warning(len(output))
                    #_log.warning(output[0].shape)
                    output_tensor = torch.cat(output).reshape(T, 1, 88).permute([1,0,2])
                    prediction = output_tensor[:, 0 : -1]
                    target = song[:, 1 : T]

                    #_log.warning(str(output_tensors[-1].shape))
                    #_log.warning(str(target_tensor.shape))
                    #loss = loss_fcn(output_tensors[-1], target_tensor[:, -1])
                    loss = loss_fcn(prediction, target)/T
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    tot_loss = loss.cpu().detach().item()

                    # use sacred to log training loss and accuracy
                    _run.log_scalar("trainLoss", tot_loss)
                    log_accuracy(model, train_songs, "trainAccuracy", cuda_device, _log, _run)

                    # save a copy of the model and make sacred remember it each epoch
                    if saving['every_epoch']:
                        sd = deepcopy(model.state_dict())
                        torch.save(init_sd, save_dir + 'state_dict_' + str(epoch) + '.pt')
                        _run.add_artifact(save_dir + 'state_dict_' + str(epoch) + '.pt')

                # learning rate decay
                if training['optimizer'] != "SecondOrder":
                    scheduler.step()

                # use sacred to log testing and validation loss and accuracy
                log_loss(loss_fcn, model, test_songs, 'testLoss', _run)
                log_loss(loss_fcn, model, val_songs, 'validLoss', _run)
                log_accuracy(model, test_songs, 'testAccuracy', _run)
                log_accuracy(model, val_songs, 'validAccuracy', _run)

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