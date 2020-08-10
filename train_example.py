"""
This script provides an example of how to call model_trainer.py
"""

from importlib import reload

import math
import torch
import torch.distributions as distribs
import numpy as np

import src.model_trainer
reload(src.model_trainer)
from src.model_trainer import ex  # importing experiment here is crucial (why?)

from sacred.observers import FileStorageObserver, RunObserver

import os

# don't record information in the file system, just investigate where the program fails
debug_mode = False


# this is a custom distribution that I use for some the experiments with block orthogonal initialization
class MyDistrib(distribs.distribution.Distribution):

   def __init__(self, angle: float, variance: float):

      super(MyDistrib, self).__init__()

      self.bern = distribs.bernoulli.Bernoulli(torch.tensor([0.5]))
      self.normal = distribs.normal.Normal(torch.zeros((1)), torch.tensor([variance]))

      self.angle = angle

   def sample(self):

      result = self.angle*(2.0*self.bern.sample() - 1.0)
      result += self.normal.sample()
      return result


# custom configuration
# it should be noted that when architecture is REGRESSION
# most of the optimization is just handled by sklearn
# all that matters is lag and window
config_updates = {
                  'architecture': "GRU",
                  'readout': "linear",
                  'optmzr': "Adam",
                  'init': "blockortho",
                  #'parity': "rotate",
                  #'t_distrib': MyDistrib(0.25*math.pi, 0.01),
                  'path': "models/209/final_state_dict.pt",

                  'dataset': "Nottingham",

                  'low_off_notes': 10,
                  'high_off_notes': 72,

                  'num_epochs': 300,
                  #'hps_epochs': 100,
                  'hidden_size': 100,
                  'scale': 0.01,

                  #'lag': 1,
                  #'window': 1,

                  'decay': 1,
                  'lr': 0.001,
                  'regularization': 0.0,

                  'do_hpsearch': False,
                  'decays': [1.0],
                  #'regularizations': [0.0001],
                  #'learning_rates': 10**np.linspace(-1, -3, num=5),

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
      ex.observers.append(FileStorageObserver(base_dir + '/models'))

      # run the experiment after adding observer
      ex.run(config_updates={**config_updates})
      result = ex.current_run.result