"""
This script provides an example of how to call experiment.py
"""

from importlib import reload

import src.experiment
reload(src.experiment)
from src.experiment import ex  # importing experiment here is crucial (why?)

from sacred.observers import FileStorageObserver

import os

# custom configuration
config_updates = {
                  'training': {
                              'dataset': "MuseData",
                              'num_epochs': 150,
                              'batch_size': 128,
                              'lr': 0.001,
                              'decay': 0.96,
                              'optimizer': "SGD"
                              },
                  'initializer': {
                                 'init': 'identity',
                                 'scale': 1.0,
                                 'min_angle': 0.0,
                                 'max_angle': 2.0
                                 },
                   'model_dict': {
                                  'architecture': 'LINEAR',
                                  'gradient_clipping': 1,
                                  'jit': False,
                                  'input_size': 88,
                                  'hidden_size': 300,
                                  'num_layers': 1,
                                  'output_size': 88
                                 }

                }

# store in local directory for now
base_dir = os.getcwd()
ex.observers.append(FileStorageObserver(base_dir + '/results'))

ex.run(config_updates={**config_updates})
result = ex.current_run.result