"""
This script creates an instance of a sacred experiment and defines default configurations.
"""

import load_data
import jits_model

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random

from sacred import Experiment

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
    factor = 1.0
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
    output_size = 6

    # initializer dictionary contains all initialization information
    initializer = {'init': 'default', 'scale': 1.0, 'min_angle': 0.0, 'max_angle': 2.0}

# main function
@ex.automain
def do_training(
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
                hpsearch_epochs,
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

    # figure out which device we are on
    device = None
    if not cuda:
        device = torch.cuda.device('cpu')
    else:
        device = torch.cuda.device('cuda': + str(gpu))

    # give all random number generators the same seed
    _seed_all(_seed)

    with device:

        # get the data loaders
        train_loader, test_val_loader = load_data.get_data_loader(name, set, batch_size)

        # we always use this loss function because it is specific to the binary prediction task
        loss_fcn = nn.BCEWithLogitsLoss(reduction='sum')

        # construct the optimizer
        if optimizer == "SGD":
            optimizer = optim.SGD(lr=lr)
        elif optimizer == "Adam":
            optimizer = optim.Adam(lr=lr)
        elif optimizer == "RMSprop":
            optimizer = optim.RMSprop(lr=lr)
        else
            raise ValueError("Optimizer {} not recognized".format(optimizer))

        # learning rate decay
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: decay**epoch)

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
        model = jits_model.get_model(**model_kwargs)

        # begin training loop
        for epoch in range(num_epochs):

            for i, data in enumerate(train_loader):

                input_tensor, target_tensor = data

                optimizer.zero_grad()

                output_tensor, hidden_tensor = model(input_tensor)
                loss = loss_fcn(output_tensor, target_tensor)
                loss.backward()
                optimizer.step()

            scheduler.step()