"""
This script provides an example of how to call experiment.py
"""

from importlib import reload

import math
import torch
import torch.distributions as distribs
import numpy as np

import src.experiment
reload(src.experiment)
from src.experiment import ex  # importing experiment here is crucial (why?)

from sacred.observers import FileStorageObserver, RunObserver

import os

# don't record information in the file system, just investigate where the program fails
debug_mode = False


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
                  'architecture': "REGRESSION",
                  'readout': None,
                  'optmzr': "SecondOrder",
                  'init': "regression",
                  #'parity': "rotate",
                  #'t_distrib': MyDistrib(0.25*math.pi, 0.01),
                  'path': "results/116/final_state_dict.pt",

                  #'num_epochs': 300,
                  #'hps_epochs': 100,
                  'hidden_size': 300,
                  'scale': 0.1,

                  'lag': 1,
                  'window': 1,

                  'decay': 1,
                  'lr': 0.00316,

                  'ema_decay': 0.999,
                  'damping': 0.001,
                  #'regularization': 0.01,

                  'do_hpsearch': False,
                  #'ema_decays': [0.999],
                  #'dampings': [0.001],
                  'decays': [1.0],
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
      ex.observers.append(FileStorageObserver(base_dir + '/results'))

      # run the experiment after adding observer
      ex.run(config_updates={**config_updates})
      result = ex.current_run.result