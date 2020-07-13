"""
This script creates an instance of a sacred experiment and defines default configurations.
"""

from src.load_data import get_loader
from src.models import get_model
from src.metrics import MaskedBCE, Accuracy, compute_acc, compute_loss

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

# TODO
# make sure test and validation accuracy are being logged correctly, might have to take unbalanced average over batches in case batches have very different sizes


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
    lr = 0.001
    decay = 1.0
    optmzr = "SGD"
    # second order optimizer params
    ema_decay = 0.99
    damping = 0.001
    regularization = 0.0

    # hyperparameter search
    do_hpsearch = False
    learning_rates = 10**np.linspace(-2, -4, 5)
    decays = 0.98 - np.linspace(0, 0.1, num=5)
    dampings = 10**np.linspace(-2, -4, 5)
    ema_decays = 0.98 - np.linspace(0, 0.1, num=5)
    hps_epochs = 50

    # Supported architectures
    # LINEAR (LDS)
    # REGRESSION (regress next note based on last note)
    # REGRESSION_8_STEP (regress next note based on last 8 notes)
    architecture = 'LDS'
    readout = 'linear'
    gradient_clipping = 1
    jit = False # not fully implemented
    # for regression
    lag = 0
    window = 1
    # for neural networks
    input_size = 88
    hidden_size = 300
    num_layers = 1
    output_size = 88

    # supported initializations
    # default
    # identity
    # blockortho
    # orthogonal
    # stdnormal
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
               val_loader: DataLoader,
               _log,
               _run):

    input_tensor = input_tensor.to(device)

    # number of songs in this batch
    N = input_tensor.shape[0]

    output, hidden_tensors = model(input_tensor)

    loss = loss_fcn(output, target, mask, model)/N
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # use sacred to log training loss and accuracy
    _run.log_scalar("trainLoss", loss.cpu().detach().item())
    train_acc = compute_acc(model, train_loader)
    _run.log_scalar("trainAccuracy", train_acc)

    # save a copy of the model and make sacred remember it each epoch
    if save_every_epoch:
        sd = deepcopy(model.state_dict())
        torch.save(init_sd, save_dir + 'state_dict_' + str(epoch) + '.pt')
        _run.add_artifact(save_dir + 'state_dict_' + str(epoch) + '.pt')


# a single optimization step
# meant for hp search where we aren't saving anything
@ex.capture
def hps_train_iter(device: device,
                   cuda_device: torch.cuda.device,
                   input_tensor: Tensor,
                   target: Tensor,
                   mask: Tensor,
                   model: nn.Module,
                   loss_fcn: nn.Module,
                   optimizer: optim.Optimizer,
                   _log,
                   _run):

    input_tensor = input_tensor.to(device)

    # number of songs in this batch
    N = input_tensor.shape[0]

    output, hidden_tensors = model(input_tensor)

    loss = loss_fcn(output, target, mask, model)/N
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# main function
@ex.automain
def train_loop(cuda,
               gpu,
               base_dir,
               dataset,
               num_epochs,
               batch_size,
               lr,
               decay,
               optmzr,
               ema_decay,
               damping,
               regularization,
               do_hpsearch,
               learning_rates,
               decays,
               ema_decays,
               dampings,
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
                   'path': path
                  }

    # give all random number generators the same seed
    _seed_all(_seed)

    # if we are debugging we may want to detect autograd anomalies
    torch.autograd.set_detect_anomaly(detect_anomaly)

    # decide how much of the beginning of each sequence to ignore and construct the data loaders
    init_mask = 0
    if architecture == "REGRESSION":
        init_mask = lag
    elif architecture == "REGRESSION_WIDE":
        init_mask = window
    train_loader, test_loader, val_loader = get_loader(dataset, batch_size, init_mask)

    # save artifacts to a temporary directory that gets erased when the experiment is over
    save_dir = base_dir + '/tmp_' + str(_seed)
    os.system('mkdir ' + save_dir)
    save_dir += '/'

    # standard training loop
    if not do_hpsearch:

        # construct and initialize the model
        model = get_model(model_dict, initializer, cuda)

        # save a copy of the initial model and make sacred remember it
        if save_init_model:
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
            loss_fcn = MaskedBCE(regularization)

            # construct the optimizer
            optimizer = None
            if optmzr == "SGD":
                optimizer = optim.SGD(model.parameters(), lr=lr)
            elif optmzr == "Adam":
                optimizer = optim.Adam(model.parameters(), lr=lr)
            elif optmzr == "RMSprop":
                optimizer = optim.RMSprop(model.parameters(), lr=lr)
            elif optmzr == "SecondOrder":
                # see https://github.com/cybertronai/pytorch-sso/blob/master/torchsso/optim/secondorder.py
                shapes = {"Linear": "Diag"}
                kwargs = {"damping": damping, "ema_decay": ema_decay}
                optimizer = soptim.SecondOrderOptimizer(model, "Cov", shapes, kwargs)
            else:
                raise ValueError("Optimizer {} not recognized.".format(optmzr))

            # learning rate decay
            scheduler = None
            if optmzr != "SecondOrder":
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
                               val_loader,
                               _log,
                               _run)

                # learning rate decay
                if optmzr != "SecondOrder":
                    scheduler.step()

                # use sacred to log testing and validation loss and accuracy
                test_loss = compute_loss(loss_fcn, model, test_loader)
                _run.log_scalar("testLoss", test_loss)
                val_loss = compute_loss(loss_fcn, model, val_loader)
                _run.log_scalar("validLoss", val_loss)
                test_acc = compute_acc(model, test_loader)
                _run.log_scalar("testAccuracy", testAcc)
                val_acc = compute_acc(model, val_loader)
                _run.log_scalar("validAccuracy", val_acc)

            # save a copy of the trained model and make sacred remember it
            if save_final_model:
                fin_sd = deepcopy(model.state_dict())
                torch.save(fin_sd, save_dir + 'final_state_dict.pt')
                _run.add_artifact(save_dir + 'final_state_dict.pt')

    # only goal here is to find the best hyper parameters
    else:

        if optmzr == "SecondOrder":

            min_loss = float('inf')

            best_decay = 1.0
            best_damping = 0.1

            for i, decay in enumerate(ema_decays):

                for j, damping in enumerate(dampings):

                    # construct and initialize the model
                    model = get_model(model_dict, initializer, cuda)

                    # save a copy of the initial model and make sacred remember it
                    if save_init_model:
                        init_sd = deepcopy(model.state_dict())
                        torch.save(init_sd, save_dir + 'initial_state_dict_' + str(i) + '_' + str(j) + '.pt')
                        _run.add_artifact(save_dir + 'initial_state_dict_' + str(i) + '_' + str(j) + '.pt')

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
                        loss_fcn = MaskedBCE(regularization)

                        # construct the optimizer
                        # see https://github.com/cybertronai/pytorch-sso/blob/master/torchsso/optim/secondorder.py
                        shapes = {"Linear": "Diag"}
                        kwargs = {"damping": damping, "ema_decay": decay}
                        optimizer = soptim.SecondOrderOptimizer(model, "Cov", shapes, kwargs)

                        # begin training loop
                        for epoch in tqdm(range(hps_epochs)):

                            for input_tensor, target, mask in train_loader:
                                hps_train_iter(device,
                                               cuda_device,
                                               input_tensor,
                                               target,
                                               mask,
                                               model,
                                               loss_fcn,
                                               optimizer,
                                               _log,
                                               _run)

                        # after training, compute average test loss
                        num_seqs = 0
                        test_loss = 0

                        for input_tensor, target, mask in test_loader:

                            num_seqs += input_tensor.shape[0]

                            output, hiddens = model(input_tensor)
                            loss = loss_fcn(output, target, mask, model)
                            test_loss += loss.cpu().detach().item()

                        test_loss /= num_seqs

                        # compare against other hyperparameters
                        if test_loss < min_loss:
                            min_loss = test_loss
                            best_decay = decay
                            best_damping = damping

                    # save a copy of the initial model and make sacred remember it
                    if save_final_model:
                        init_sd = deepcopy(model.state_dict())
                        torch.save(init_sd, save_dir + 'final_state_dict_' + str(i) + '_' + str(j) + '.pt')
                        _run.add_artifact(save_dir + 'final_state_dict_' + str(i) + '_' + str(j) + '.pt')

            # use sacred to record the best hyperparameters
            _run.log_scalar("ema_decay", best_decay)
            _run.log_scalar("damping", best_damping)


        else:

            min_loss = float('inf')

            best_decay = 1.0
            best_lr = 0.1

            for i, decay in enumerate(decays):

                for j, lr in enumerate(learning_rates):

                    # construct and initialize the model
                    model = get_model(model_dict, initializer, cuda)

                    # save a copy of the initial model and make sacred remember it
                    if save_init_model:
                        init_sd = deepcopy(model.state_dict())
                        torch.save(init_sd, save_dir + 'initial_state_dict_' + str(i) + '_' + str(j) + '.pt')
                        _run.add_artifact(save_dir + 'initial_state_dict_' + str(i) + '_' + str(j) + '.pt')

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
                        loss_fcn = MaskedBCE(regularization)

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
                        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: decay**epoch)

                        # begin training loop
                        for epoch in tqdm(range(hps_epochs)):

                            for input_tensor, target, mask in train_loader:
                                hps_train_iter(device,
                                               cuda_device,
                                               input_tensor,
                                               target,
                                               mask,
                                               model,
                                               loss_fcn,
                                               optimizer,
                                               _log,
                                               _run)

                            # learning rate decay
                            scheduler.step()

                        # after training, compute average test loss
                        num_seqs = 0
                        test_loss = 0

                        for input_tensor, target, mask in test_loader:

                            num_seqs += input_tensor.shape[0]

                            output, hiddens = model(input_tensor)
                            loss = loss_fcn(output, target, mask, model)
                            test_loss += loss.cpu().detach().item()

                        test_loss /= num_seqs

                        # compare against other hyperparameters
                        if test_loss < min_loss:
                            min_loss = test_loss
                            best_decay = decay
                            best_lr = lr

                    # save a copy of the initial model and make sacred remember it
                    if save_final_model:
                        init_sd = deepcopy(model.state_dict())
                        torch.save(init_sd, save_dir + 'final_state_dict_' + str(i) + '_' + str(j) + '.pt')
                        _run.add_artifact(save_dir + 'final_state_dict_' + str(i) + '_' + str(j) + '.pt')

            # use sacred to record the best hyperparameters
            _run.log_scalar("decay", best_decay)
            _run.log_scalar("learning_rate", best_lr)

    # wait a second then remove the temporary directory used for storing artifacts
    sleep(1)
    os.system('rm -r ' + save_dir)
