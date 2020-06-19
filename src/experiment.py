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

    # system arguments
    cuda = torch.cuda.is_available()
    gpu = 0
    save_dir = os.getcwd()

    # training arguments
    name = "Nottingham"
    set = "test"
    num_epochs = 150
    batch_size = 128
    lr = 1e-3
    decay = 1.0
    optimizer = "SGD"

    # dictionary containing all the information about hyper-parameter search
    hpsearch = {'do_hpsearch': False, 'learning_rates': 10**np.linspace(-2, -4, 5), 'epochs': 50}

    # detect backprop anamolies
    detect_anomaly = False

    #RNN arguments
    architecture = 'LINEAR'
    gradient_clipping = None
    input_size = 88
    hidden_size = 300
    num_layers = 1
    output_size = 88

    # initializer dictionary contains all initialization information
    initializer = {'init': 'default', 'scale': 1.0, 'min_angle': 0.0, 'max_angle': 2.0}


# main function
@ex.automain
def train_model(
                cuda,
                gpu,
                save_dir,
                name,
                set,
                num_epochs,
                batch_size,
                lr,
                optimizer,
                hpsearch,
                detect_anomaly,
                architecture,
                gradient_clipping,
                input_size,
                hidden_size,
                num_layers,
                output_size,
                initializer,
                _seed,
                _run):

    # give all random number generators the same seed
    _seed_all(_seed)

    # if we are debugging we may want to detect autograd anomalies
    torch.autograd.set_detect_anomaly(detect_anomaly)

    # get the data loaders
    train_loader, test_loader, val_loader = get_data_loader(name, set, batch_size)

    # standard training loop
    if not hpsearch['do_hpsearch']:

        # if we are on cuda we construct the device and run everything on it
        device = util.NullContext()
        if cuda:
            device = torch.cuda.device('cuda:' + str(gpu))

        with device:

            # construct the model
            model_kwargs = {
                            "architecture": architecture,
                            "gradient_clipping": gradient_clipping,
                            "input_size": input_size,
                            "hidden_size": hidden_size,
                            "num_layers": num_layers,
                            "output_size": output_size,
                            "initializer": initializer
                            }
            model = get_model(**model_kwargs)

            # always use this loss function for multi-variate binary prediction
            loss_fcn = nn.BCEWithLogitsLoss(reduction='sum')

            # construct the optimizer
            if optimizer == "SGD":
                optimizer = optim.SGD(model.parameters(), lr=lr)
            elif optimizer == "Adam":
                optimizer = optim.Adam(model.parameters(), lr=lr)
            elif optimizer == "RMSprop":
                optimizer = optim.RMSprop(model.parameters(), lr=lr)
            else:
                raise ValueError("Optimizer {} not recognized.".format(optimizer))

            # learning rate decay
            #scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: decay**epoch)

            # save a copy of the initial model and make sacred remember it
            init_sd = deepcopy(model.state_dict())
            torch.save(init_sd, 'initial_state_dict.pt')
            _run.add_artifact('initial_state_dict.pt')

            # begin training loop
            for epoch in tqdm(range(num_epochs)):

                for input_tensor, target_tensor in train_loader:

                    optimizer.zero_grad()

                    #input_tensor = input_tensor.permute([0, 2, 1])
                    output_tensor, hidden_tensor = model(input_tensor)
                    output_tensor = output_tensor.permute([0, 2, 1])

                    loss = loss_fcn(output_tensor, target_tensor)
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
            sd = mode.state_dict()
            torch.save(sd, 'final_state_dict.pt')
            _run.add_artifact('final_state_dict.pt')

    # only goal here is to find the best hyper parameters
    else:
        raise NotImplementedError