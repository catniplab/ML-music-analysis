"""
This script provides an example of how to call experiment.py
"""

from importlib import reload

import torch
import numpy as np

import src.experiment
reload(src.experiment)
from src.experiment import ex  # importing experiment here is crucial (why?)

from sacred.observers import FileStorageObserver, RunObserver

import os

# don't record information in the file system, just investigate where the program fails
debug_mode = False

# custom configuration
config_updates = {
                  'architecture': "LINEAR",
                  'optmzr': "SGD",
                  'init': "identity",
                  'parity': "rotate",

                  #'num_epochs': 300,
                  #'hps_epochs': 100,
                  'hidden_size': 88,

                  'lag': 0,
                  'window': 7,

                  'decay': 0.93,
                  'lr': 0.001,

                  'ema_decay': 0.999,
                  'damping': 0.001,

                  'do_hpsearch': False,

                  'save_init_model': True,
                  'save_final_model': True
                 }


if __name__ == "__main__":

   base_dir = os.getcwd()

   if debug_mode:

      # run the experiment without an observer
      ex.run(config_updates={**config_updates})
      result = ex.current_run.result

   else:

      # store in local directory for now
      ex.observers.append(FileStorageObserver(base_dir + '/results'))

      # run the experiment after adding observer
      ex.run(config_updates={**config_updates})
      result = ex.current_run.result